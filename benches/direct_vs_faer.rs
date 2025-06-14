use criterion::{black_box, Criterion, criterion_group, criterion_main};
use faer::Mat;
use faer::linalg::solvers::SolveCore;
use kryst::solver::{LuSolver, LinearSolver};

fn bench_lu_vs_faer(c: &mut Criterion) {
    let n = 200;
    let data: Vec<f64> = (0..n*n).map(|i| (i as f64).sin()).collect();
    let a = Mat::from_fn(n, n, |i, j| data[j * n + i]);
    let b: Vec<f64> = (0..n).map(|i| (i as f64).cos()).collect();
    let mut x = vec![0.0; n];

    c.bench_function("kryst LU", |ben| {
        let mut solver = LuSolver::new();
        ben.iter(|| {
            let _stats = solver.solve(black_box(&a), None, black_box(&b), black_box(&mut x)).unwrap();
        })
    });

    c.bench_function("faer raw LU", |ben| {
        ben.iter(|| {
            let factor = faer::linalg::solvers::FullPivLu::new(a.as_ref());
            let mut y = b.clone();
            let n = y.len();
            let y_mat = faer::MatMut::from_column_major_slice_mut(&mut y, n, 1);
            factor.solve_in_place_with_conj(faer::Conj::No, y_mat);
        })
    });
}

criterion_group!(benches, bench_lu_vs_faer);
criterion_main!(benches);
