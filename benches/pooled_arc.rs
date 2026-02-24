use criterion::{black_box, criterion_group, criterion_main, Criterion};
use pooled_arc::{impl_internable, PooledArc};

// Test type for benchmarking
#[derive(Hash, Eq, PartialEq, Clone)]
struct TestData(u64);

impl_internable!(TestData);

fn benchmark_new(c: &mut Criterion) {
    c.bench_function("PooledArc::new", |b| {
        b.iter(|| {
            let _arc = PooledArc::new(black_box(TestData(42)));
        })
    });
}

fn benchmark_new_duplicate(c: &mut Criterion) {
    c.bench_function("PooledArc::new (duplicate)", |b| {
        // Pre-create an arc with a specific value
        let _existing = PooledArc::new(TestData(100));
        
        b.iter(|| {
            // Creating a new PooledArc with the same value should reuse the existing Arc
            let _arc = PooledArc::new(black_box(TestData(100)));
        })
    });
}

fn benchmark_from_ref(c: &mut Criterion) {
    c.bench_function("PooledArc::from_ref", |b| {
        let value = TestData(42);
        b.iter(|| {
            let _arc = PooledArc::from_ref(black_box(&value));
        })
    });
}

fn benchmark_from_ref_duplicate(c: &mut Criterion) {
    c.bench_function("PooledArc::from_ref (duplicate)", |b| {
        let _existing = PooledArc::new(TestData(200));
        let value = TestData(200);
        
        b.iter(|| {
            let _arc = PooledArc::from_ref(black_box(&value));
        })
    });
}

fn benchmark_drop(c: &mut Criterion) {
    c.bench_function("PooledArc::drop (pool cleanup)", |b| {
        b.iter(|| {
            let arc = PooledArc::new(black_box(TestData(300)));
            drop(arc);
        })
    });
}

fn benchmark_drop_with_clone(c: &mut Criterion) {
    c.bench_function("PooledArc::drop (with clone)", |b| {
        let arc1 = PooledArc::new(TestData(400));
        
        b.iter(|| {
            let arc2 = arc1.clone();
            drop(arc2);
        })
    });
}

fn benchmark_new_large_pool(c: &mut Criterion) {
    let mut group = c.benchmark_group("large pool (10k entries)");

    // Pre-fill the pool with 10,000 unique entries
    let pool: Vec<_> = (10_000..20_000)
        .map(|i| PooledArc::new(TestData(i)))
        .collect();

    group.bench_function("new (miss)", |b| {
        b.iter(|| {
            // Value not in pool: must scan hash bucket (empty for this hash) then insert
            let arc = PooledArc::new(black_box(TestData(999_999)));
            drop(arc);
        })
    });

    group.bench_function("new (hit)", |b| {
        b.iter(|| {
            // Value already in pool: must find it via hash bucket lookup
            let _arc = PooledArc::new(black_box(TestData(15_000)));
        })
    });

    group.bench_function("from_ref (hit)", |b| {
        let value = TestData(15_000);
        b.iter(|| {
            let _arc = PooledArc::from_ref(black_box(&value));
        })
    });

    group.bench_function("drop (pool cleanup)", |b| {
        b.iter(|| {
            let arc = PooledArc::new(black_box(TestData(999_998)));
            drop(arc);
        })
    });

    group.bench_function("drop (with clone)", |b| {
        b.iter(|| {
            let arc = pool[5000].clone();
            drop(arc);
        })
    });

    drop(pool);
    group.finish();
}

fn benchmark_new_large_pool_collisions(c: &mut Criterion) {
    // Type with constant hash to force all entries into the same bucket
    #[derive(Eq, PartialEq, Clone)]
    struct Collider(u64);

    impl std::hash::Hash for Collider {
        fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
            0u64.hash(state);
        }
    }

    impl_internable!(Collider);

    let mut group = c.benchmark_group("large pool collisions (1k same bucket)");

    // Pre-fill with 1,000 entries all in the same hash bucket
    let pool: Vec<_> = (0..1_000)
        .map(|i| PooledArc::new(Collider(i)))
        .collect();

    group.bench_function("new (hit, worst case)", |b| {
        b.iter(|| {
            // Last entry: must scan through all 1,000 pointers in the bucket
            let _arc = PooledArc::new(black_box(Collider(999)));
        })
    });

    group.bench_function("new (hit, best case)", |b| {
        b.iter(|| {
            // First entry: found immediately
            let _arc = PooledArc::new(black_box(Collider(0)));
        })
    });

    group.bench_function("drop (pool cleanup)", |b| {
        b.iter(|| {
            let arc = PooledArc::new(black_box(Collider(9999)));
            drop(arc);
        })
    });

    // Keep the pool alive until the end of the benchmark to avoid measuring cleanup time in the drop benchmarks
    drop(pool);
    group.finish();
}

criterion_group!(
    benches,
    benchmark_new,
    benchmark_new_duplicate,
    benchmark_from_ref,
    benchmark_from_ref_duplicate,
    benchmark_drop,
    benchmark_drop_with_clone,
    benchmark_new_large_pool,
    benchmark_new_large_pool_collisions
);
criterion_main!(benches);
