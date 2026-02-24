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

criterion_group!(
    benches,
    benchmark_new,
    benchmark_new_duplicate,
    benchmark_from_ref,
    benchmark_from_ref_duplicate,
    benchmark_drop,
    benchmark_drop_with_clone
);
criterion_main!(benches);
