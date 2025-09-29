use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use kryst::matrix::sparse::CsrMatrix;
use kryst::solver::block::block_vec::BlockVec;
use kryst::solver::block::kernels::spmm_csr_dense;

fn poisson3d(n: usize) -> CsrMatrix<f64> {
    kryst::matrix::utils::poisson::poisson_7pt_3d(n)
}

fn bench_spmv_vs_spmm(c: &mut Criterion) {
    let a = poisson3d(20); // ~8k unknowns, quick but illustrative
    let n = a.nrows();

    c.bench_function("spmv_x4", |b| {
        b.iter_batched(
            || (vec![1.0f64; n], vec![0.0f64; n]),
            |(x, mut y)| {
                for _ in 0..4 {
                    a.spmv_scaled(1.0, &x, 0.0, &mut y).unwrap();
                }
            },
            BatchSize::SmallInput,
        )
    });

    c.bench_function("spmm_block4", |b| {
        b.iter_batched(
            || {
                let mut x = BlockVec::new(n, 4);
                for col in 0..4 {
                    for row in 0..n {
                        x[(row, col)] = 1.0;
                    }
                }
                let y = BlockVec::new(n, 4);
                (x, y)
            },
            |(x, mut y)| {
                spmm_csr_dense(&a, &x, &mut y).unwrap();
            },
            BatchSize::SmallInput,
        )
    });
}

criterion_group!(benches, bench_spmv_vs_spmm);
criterion_main!(benches);
