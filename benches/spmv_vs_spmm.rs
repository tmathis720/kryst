#![cfg_attr(feature = "complex", allow(dead_code))]

#[cfg(not(feature = "complex"))]
use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
#[cfg(not(feature = "complex"))]
use kryst::matrix::sparse::CsrMatrix;
#[cfg(not(feature = "complex"))]
use kryst::solver::block::block_vec::BlockVec;
#[cfg(not(feature = "complex"))]
use kryst::solver::block::kernels::spmm_csr_dense;

#[cfg(not(feature = "complex"))]
fn poisson3d(n: usize) -> CsrMatrix<f64> {
    kryst::matrix::utils::poisson::poisson_7pt_3d(n)
}

#[cfg(not(feature = "complex"))]
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

#[cfg(not(feature = "complex"))]
criterion_group!(benches, bench_spmv_vs_spmm);
#[cfg(not(feature = "complex"))]
criterion_main!(benches);

#[cfg(feature = "complex")]
fn main() {
    eprintln!(
        "spmv_vs_spmm benchmark is disabled when building with the `complex` feature."
    );
}
