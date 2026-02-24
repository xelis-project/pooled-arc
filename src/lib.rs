use std::{
    borrow::Borrow,
    collections::HashMap,
    fmt::{self, Debug, Display},
    hash::{Hash, Hasher},
    ops::Deref,
    sync::{Arc, Mutex, Weak},
};

/// Wrapper for raw pointers to make them thread-safe when T is Send + Sync.
///
/// # Safety
///
/// This is safe because:
/// 1. We only store pointers derived from `Arc::as_ptr()` on live Arc allocations
/// 2. Each Arc has a unique pointer that never changes during its lifetime
/// 3. Pointers are only used as HashMap keys for identity-based lookup, never dereferenced
/// 4. The pool's Weak<T> references prevent use-after-free (we verify via strong_count())
/// 5. Pointers are removed from the pool before the Arc is freed (via Drop)
///
/// The pointer is never used to access data directly; it only enables O(1) removal.
#[derive(PartialEq, Eq, Hash)]
pub struct PoolPtr<T: ?Sized>(*const T);

unsafe impl<T: ?Sized + Send + Sync> Send for PoolPtr<T> {}
unsafe impl<T: ?Sized + Send + Sync> Sync for PoolPtr<T> {}

pub type SharedPool<T> = Mutex<HashMap<PoolPtr<T>, Weak<T>>>;

/// Trait for types that can be interned in a global static pool.
///
/// Each type implementing `Internable` gets its own static pool.
/// Use the [`impl_internable!`] macro to implement this trait.
pub trait Internable: Hash + Eq + Send + Sync + 'static {
    /// Returns a reference to the global static pool for this type.
    fn pool() -> &'static SharedPool<Self>;
}

/// Macro to implement [`Internable`] for one or more types,
/// creating a dedicated static pool for each.
#[macro_export]
macro_rules! impl_internable {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl $crate::Internable for $ty {
                #[inline(always)]
                fn pool() -> &'static $crate::SharedPool<Self> {
                    use std::{
                        collections::HashMap,
                        sync::{LazyLock, Mutex}
                    };
                    static POOL: LazyLock<$crate::SharedPool<$ty>> = LazyLock::new(|| Mutex::new(HashMap::new()));
                    &POOL
                }
            }
        )+
    };
}

/// A shared, interned reference-counted pointer that deduplicates equal values.
///
/// `PooledArc<T>` wraps an `Arc<T>` and ensures that equal values share
/// the same allocation via a global static pool (one per type `T`).
///
/// When the last `PooledArc` referencing a value is dropped (only the pool's
/// copy remains), the entry is automatically removed from the pool.
pub struct PooledArc<T: Internable>(Arc<T>);

impl<T: Internable> PooledArc<T> {
    /// Helper to find an existing `Arc<T>` in the pool that matches the given value.
    fn find_matching_arc(value: &T, map: &HashMap<PoolPtr<T>, Weak<T>>) -> Option<Arc<T>> {
        for weak in map.values() {
            if let Some(arc) = weak.upgrade() {
                if *arc == *value {
                    return Some(arc);
                }
            }
        }
        None
    }

    /// Create or retrieve an interned `PooledArc` for the given value.
    ///
    /// If an equal value already exists in the pool, the existing
    /// `Arc<T>` is reused. Otherwise, a new one is created and stored.
    pub fn new(value: T) -> Self {
        let pool = T::pool();
        let mut map = pool.lock().expect("Failed to lock pool");

        if let Some(arc) = Self::find_matching_arc(&value, &map) {
            return Self(arc);
        }

        let arc = Arc::new(value);
        let ptr = PoolPtr(Arc::as_ptr(&arc));
        map.insert(ptr, Arc::downgrade(&arc));
        Self(arc)
    }

    /// Get a reference to the inner `Arc<T>`.
    pub fn inner(&self) -> &Arc<T> {
        &self.0
    }

    /// Number of strong references to this value
    /// (includes the pool's own reference).
    pub fn strong_count(&self) -> usize {
        Arc::strong_count(&self.0)
    }

    /// Get a raw pointer to the inner value for fast identity checks.
    pub fn as_ptr(&self) -> *const T {
        Arc::as_ptr(&self.0)
    }
}

impl<T: Internable> From<T> for PooledArc<T> {
    fn from(value: T) -> Self {
        Self::new(value)
    }
}

impl<T: Internable + Clone> PooledArc<T> {
    pub fn from_ref(value: &T) -> Self {
        let pool = T::pool();
        let mut map = pool.lock().expect("Failed to lock pool");

        if let Some(arc) = Self::find_matching_arc(value, &map) {
            return Self(arc);
        }

        let arc = Arc::new(value.clone());
        let ptr = PoolPtr(Arc::as_ptr(&arc));
        map.insert(ptr, Arc::downgrade(&arc));
        Self(arc)
    }

    /// Get an owned clone of the inner value.
    pub fn to_owned_inner(&self) -> T {
        self.0.as_ref().clone()
    }
}

impl<T: Internable> Drop for PooledArc<T> {
    fn drop(&mut self) {
        // strong_count == 1 means only this PooledArc holds it;
        // after this drop the Arc will be freed, so clean up the pool.
        if Arc::strong_count(&self.0) <= 1 {
            let ptr = PoolPtr(Arc::as_ptr(&self.0));
            if let Ok(mut map) = T::pool().lock() {
                map.remove(&ptr);
            }
        }
    }
}

impl<T: Internable> Clone for PooledArc<T> {
    fn clone(&self) -> Self {
        PooledArc(Arc::clone(&self.0))
    }
}

impl<T: Internable> Deref for PooledArc<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: Internable> AsRef<T> for PooledArc<T> {
    fn as_ref(&self) -> &T {
        &self.0
    }
}

impl<T: Internable> Borrow<T> for PooledArc<T> {
    fn borrow(&self) -> &T {
        &self.0
    }
}

impl<T: Internable> Hash for PooledArc<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl<T: Internable> PartialEq for PooledArc<T> {
    fn eq(&self, other: &Self) -> bool {
        // Fast path: same allocation means equal
        Arc::ptr_eq(&self.0, &other.0) || *self.0 == *other.0
    }
}

impl<T: Internable> Eq for PooledArc<T> {}

impl<T: Internable> PartialEq<T> for PooledArc<T> {
    fn eq(&self, other: &T) -> bool {
        *self.0 == *other
    }
}

impl<T: Internable + PartialOrd> PartialOrd for PooledArc<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl<T: Internable + Ord> Ord for PooledArc<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.cmp(&other.0)
    }
}

impl<T: Internable + Debug> Debug for PooledArc<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl<T: Internable + Display> Display for PooledArc<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[cfg(feature = "serde")]
impl<T: Internable + serde::Serialize> serde::Serialize for PooledArc<T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.0.serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de, T: Internable + serde::Deserialize<'de>> serde::Deserialize<'de> for PooledArc<T> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        T::deserialize(deserializer).map(PooledArc::new)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Hash, Eq, PartialEq, Clone, Debug)]
    struct Foo {
        id: u64,
        name: String,
    }

    // Test type with its own static pool
    impl_internable!(Foo);

    #[test]
    fn test_deduplication() {
        // Use unique values per test to avoid interference from parallel tests
        let a = PooledArc::new(Foo { id: 100, name: "test".to_string() });
        let b = PooledArc::new(Foo { id: 100, name: "test".to_string() });

        // Same allocation
        assert!(Arc::ptr_eq(a.inner(), b.inner()));
        // a + b = 2 (pool only holds a Weak)
        assert_eq!(a.strong_count(), 2);
    }

    #[test]
    fn test_different_values() {
        let a = PooledArc::new(Foo { id: 200, name: "a".to_string() });
        let b = PooledArc::new(Foo { id: 201, name: "b".to_string() });

        assert!(!Arc::ptr_eq(a.inner(), b.inner()));
    }

    #[test]
    fn test_auto_cleanup_on_drop() {
        let val = Foo { id: 300, name: "test".to_string() };
        let a = PooledArc::new(val.clone());
        // a = 1 (pool only holds a Weak)
        assert_eq!(a.strong_count(), 1);

        drop(a);
        // After dropping the only user reference, pool entry is removed
        let map = Foo::pool().lock().unwrap();
        assert!(
            map.is_empty(),
            "Pool entry should be cleaned up after last PooledArc is dropped"
        );
    }

    #[test]
    fn test_no_cleanup_while_others_alive() {
        let a = PooledArc::new(Foo { id: 400, name: "test".to_string() });
        let b = a.clone();
        // a + b = 2 (pool only holds a Weak)
        assert_eq!(a.strong_count(), 2);

        drop(a);
        // b is still alive, pool entry must remain
        assert_eq!(b.strong_count(), 1); // only b
    }

    #[test]
    fn test_reuse_after_full_drop() {
        let val = Foo { id: 500, name: "test".to_string() };
        {
            let a = PooledArc::new(val.clone());
            assert_eq!(*a, val);
        }
        // Pool entry was cleaned up, creating again works
        let b = PooledArc::new(val.clone());
        assert_eq!(*b, val);
        // b = 1 (pool only holds a Weak)
        assert_eq!(b.strong_count(), 1);
    }

    #[test]
    fn test_shared_arc_eq() {
        let a = PooledArc::new(Foo { id: 600, name: "test".to_string() });
        let b = PooledArc::new(Foo { id: 600, name: "test".to_string() });
        assert_eq!(a, b);
    }

    #[test]
    fn test_shared_arc_hash() {
        use std::collections::hash_map::DefaultHasher;

        let a = PooledArc::new(Foo { id: 700, name: "test".to_string() });
        let b = PooledArc::new(Foo { id: 700, name: "test".to_string() });

        let mut h1 = DefaultHasher::new();
        a.hash(&mut h1);

        let mut h2 = DefaultHasher::new();
        b.hash(&mut h2);

        assert_eq!(h1.finish(), h2.finish());
    }

    #[test]
    fn test_shared_arc_in_hashmap() {
        let mut map = HashMap::new();

        let key = PooledArc::new(Foo { id: 800, name: "hello".to_string() });
        map.insert(key.clone(), 42);

        let lookup = PooledArc::new(Foo { id: 800, name: "hello".to_string() });
        assert_eq!(map.get(&lookup), Some(&42));
        assert!(Arc::ptr_eq(key.inner(), lookup.inner()));
    }

    #[test]
    fn test_cross_type_eq() {
        let a = PooledArc::new(Foo { id: 900, name: "test".to_string() });
        let b = Foo { id: 900, name: "test".to_string() };
        assert!(a == b);
    }

    /// A type where `Hash` always produces the same value,
    /// but `Eq` compares by actual content. This forces every
    /// insert into the same hash bucket, exercising the collision path.
    #[derive(Eq, PartialEq, Clone, Debug)]
    struct Collider {
        value: u64,
    }

    impl Hash for Collider {
        fn hash<H: Hasher>(&self, state: &mut H) {
            // Constant hash: every Collider lands in the same bucket
            0u64.hash(state);
        }
    }

    impl_internable!(Collider);

    #[test]
    fn test_hash_collision_different_values() {
        let a = PooledArc::new(Collider { value: 1 });
        let b = PooledArc::new(Collider { value: 2 });

        // Same hash, but different values must NOT share an Arc
        assert!(!Arc::ptr_eq(a.inner(), b.inner()));
        assert_ne!(a, b);

        // Each should still deduplicate with its own equal value
        let a2 = PooledArc::new(Collider { value: 1 });
        let b2 = PooledArc::new(Collider { value: 2 });
        assert!(Arc::ptr_eq(a.inner(), a2.inner()));
        assert!(Arc::ptr_eq(b.inner(), b2.inner()));
        assert!(a == a2);
        assert!(b == b2);
    }

    #[test]
    fn test_hash_collision_cleanup() {
        let c1 = PooledArc::new(Collider { value: 10 });
        let c2 = PooledArc::new(Collider { value: 20 });
        let c3 = PooledArc::new(Collider { value: 30 });

        // All three are in the pool
        {
            let map = Collider::pool().lock().unwrap();
            assert_eq!(map.len(), 3);
        }

        // Drop one — only its entry should be removed from the pool
        drop(c2);
        {
            let map = Collider::pool().lock().unwrap();
            assert_eq!(map.len(), 2);
            // c1 and c3 must still be alive
            let alive: Vec<_> = map.values().filter_map(|w| w.upgrade()).collect();
            assert_eq!(alive.len(), 2);
        }

        // Drop the rest — pool should be fully empty
        drop(c1);
        drop(c3);
        {
            let map = Collider::pool().lock().unwrap();
            assert!(map.is_empty());
        }
    }

    #[test]
    fn test_hash_collision_many_values() {
        // Insert many distinct values that all collide
        let arcs: Vec<_> = (0..100)
            .map(|i| PooledArc::new(Collider { value: i }))
            .collect();

        // All should be distinct allocations
        for i in 0..arcs.len() {
            for j in (i + 1)..arcs.len() {
                assert!(!Arc::ptr_eq(arcs[i].inner(), arcs[j].inner()));
            }
        }

        // Deduplication still works within the collision bucket
        for i in 0..100u64 {
            let dup = PooledArc::new(Collider { value: i });
            assert!(Arc::ptr_eq(arcs[i as usize].inner(), dup.inner()));
        }
    }
}
