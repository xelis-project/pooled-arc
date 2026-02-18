use std::{
    borrow::Borrow,
    collections::HashMap,
    fmt::{self, Debug, Display},
    hash::{DefaultHasher, Hash, Hasher},
    ops::Deref,
    sync::{Arc, Mutex, Weak},
};

pub type SharedPool<T> = Mutex<HashMap<u64, Vec<Weak<T>>>>;

/// Compute a u64 hash of a value for pool bucketing.
fn compute_hash<T: Hash>(value: &T) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

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
    fn find_matching_arc(value: &T, entries: &mut Vec<Weak<T>>) -> Option<Arc<T>> {
        let mut found = None;
        entries.retain(|w| {
            if let Some(arc) = w.upgrade() {
                if *arc == *value {
                    found = Some(arc);
                }
                true
            } else {
                false // dead weak, remove it
            }
        });

        found
    }

    /// Create or retrieve an interned `PooledArc` for the given value.
    ///
    /// If an equal value already exists in the pool, the existing
    /// `Arc<T>` is reused. Otherwise, a new one is created and stored.
    pub fn new(value: T) -> Self {
        let hash = compute_hash(&value);
        let pool = T::pool();
        let mut map = pool.lock().expect("Failed to lock pool");

        if let Some(entries) = map.get_mut(&hash) {
            if let Some(arc) = Self::find_matching_arc(&value, entries) {
                return Self(arc);
            }
        }

        let arc = Arc::new(value);
        map.entry(hash).or_default().push(Arc::downgrade(&arc));
        Self(arc)
    }

    /// Get a reference to the inner `Arc<T>`.
    pub fn inner(&self) -> &Arc<T> {
        &self.0
    }

    /// Convert into the inner `Arc<T>`, disconnected from the pool.
    pub fn into_arc(self) -> Arc<T> {
        Arc::clone(&self.0)
        // self is dropped here, which may clean up the pool entry
        // if no other PooledArc references this value
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

impl<T: Internable + Clone> PooledArc<T> {
    pub fn from_ref(value: &T) -> Self {
        let hash = compute_hash(&value);
        let pool = T::pool();
        let mut map = pool.lock().expect("Failed to lock pool");

        if let Some(entries) = map.get_mut(&hash) {
            if let Some(arc) = Self::find_matching_arc(value, entries) {
                return Self(arc);
            }
        }

        let arc = Arc::new(value.clone());
        map.entry(hash).or_default().push(Arc::downgrade(&arc));
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
            let hash = compute_hash(self.0.as_ref());
            if let Ok(mut map) = T::pool().lock() {
                if let Some(entries) = map.get_mut(&hash) {
                    // Remove our specific weak (same allocation) and any dead ones
                    let self_ptr = Arc::as_ptr(&self.0);
                    entries.retain(|w| {
                        match w.upgrade() {
                            Some(arc) => !std::ptr::eq(Arc::as_ptr(&arc), self_ptr),
                            None => false, // dead weak, remove it
                        }
                    });
                    if entries.is_empty() {
                        map.remove(&hash);
                    }
                }
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
        let hash = compute_hash(&val);
        assert!(
            !map.contains_key(&hash),
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
}
