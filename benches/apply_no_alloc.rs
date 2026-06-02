#[path = "infra/alloc.rs"]
mod alloc;
#[path = "infra/datasets.rs"]
mod datasets;

use alloc::{alloc_counts, reset_alloc_counts};
use criterion::{Criterion, criterion_group, criterion_main};
use kryst::matrix::op::LinOp; // for trait bounds
use kryst::matrix::sparse::CsrMatrix;
use kryst::preconditioner::{Jacobi, PcSide, Preconditioner};
use std::hint::black_box;

fn no_alloc_apply_jacobi(c: &mut Criterion) {
    let n = 400;
    let a: CsrMatrix<f64> = datasets::poisson2d_csr(n);
    let mut pc = Jacobi::new();
    pc.setup(&a as &dyn LinOp<S = f64>).unwrap();

    let m = a.nrows();
    let x = vec![1.0; m];
    let mut y = vec![0.0; m];

    // Warmup
    for _ in 0..10 {
        pc.apply(PcSide::Left, &x, &mut y).unwrap();
    }

    c.bench_function("apply_no_alloc_jacobi", |b| {
        b.iter(|| {
            reset_alloc_counts();
            pc.apply(PcSide::Left, black_box(&x), black_box(&mut y))
                .unwrap();
            let (a, d) = alloc_counts();
            assert_eq!(a - d, 0, "heap activity detected in apply()");
        })
    });
}

criterion_group!(benches, no_alloc_apply_jacobi);
criterion_main!(benches);
