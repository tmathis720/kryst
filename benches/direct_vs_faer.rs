use criterion::{Criterion, criterion_group, criterion_main};
use faer::Mat;
#[cfg(feature = "dense-direct")]
use kryst::solver::LuSolver;

fn bench_lu_vs_faer(_c: &mut Criterion) {
    let _comm = kryst::parallel::UniverseComm::NoComm(kryst::parallel::NoComm);
    let n = 200;
    let data: Vec<f64> = (0..n * n).map(|i| (i as f64).sin()).collect();
    let _a = Mat::from_fn(n, n, |i, j| data[j * n + i]);
    let _b: Vec<f64> = (0..n).map(|i| (i as f64).cos()).collect();
    let _x = vec![0.0; n];

    #[cfg(not(feature = "dense-direct"))]
    {
        println!("Skipping LU benchmark: 'dense-direct' feature not enabled.");
        return;
    }
    #[cfg(feature = "dense-direct")]
    c.bench_function("kryst LU", |ben| {
        let mut solver = LuSolver::new();
        ben.iter(|| {
            let _stats = solver
                .solve(
                    black_box(&a),
                    None,
                    black_box(&b),
                    black_box(&mut x),
                    PcSide::Left,
                    &comm,
                    None,
                    None,
                )
                .unwrap();
        })
    });

    #[cfg(feature = "dense-direct")]
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
