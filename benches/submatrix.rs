use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use kryst::core::traits::SubmatrixExtract;
use kryst::matrix::sparse::CsrMatrix;

#[path = "infra/datasets.rs"]
mod datasets;

fn bench_submatrix(c: &mut Criterion) {
    let (a, blocks) = datasets::blocky_csr(40, 2_000, 1); // N = 80k
    c.bench_function("extract_blocks", |b| {
        b.iter_batched(
            || (),
            |_| {
                let mut out: Vec<CsrMatrix<f64>> = Vec::with_capacity(blocks.len());
                for idx in &blocks {
                    let sub = a.submatrix(idx);
                    out.push(sub);
                }
                black_box(out)
            },
            BatchSize::SmallInput,
        )
    });
}

criterion_group!(benches, bench_submatrix);
criterion_main!(benches);
