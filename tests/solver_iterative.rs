use faer::Mat;
use faer::linalg::solvers::SolveCore;
use kryst::solver::{CgSolver, GmresSolver, LinearSolver};
use rand::Rng;
use approx::assert_abs_diff_eq;

// Helper: small SPD matrix A = Mᵀ M + I
fn random_spd(n: usize) -> (faer::Mat<f64>, Vec<f64>) {
    let mut rng = rand::thread_rng();
    let data: Vec<f64> = (0..n*n).map(|_| rng.gen()).collect();
    let m = Mat::from_fn(n, n, |i, j| data[j * n + i]);
    let m_t = m.transpose();
    let a = &m_t * &m + Mat::<f64>::identity(n, n);
    let b: Vec<f64> = (0..n).map(|_| rng.gen()).collect();
    (a, b)
}

#[test]
fn cg_vs_direct_on_spd() {
    let n = 10;
    let (a, b) = random_spd(n);
    let mut x_cg = vec![0.0; n];
    let mut solver = CgSolver::new(1e-8, 1000);
    let stats = solver.solve(&a, None, &b, &mut x_cg).unwrap();
    assert!(stats.converged);
    // direct solve
    let mut x_direct = b.clone();
    let lus = faer::linalg::solvers::FullPivLu::new(a.as_ref());
    let x_mat = faer::MatMut::from_column_major_slice_mut(&mut x_direct, n, 1);
    lus.solve_in_place_with_conj(faer::Conj::No, x_mat);
    for i in 0..n {
        assert_abs_diff_eq!(x_cg[i], x_direct[i], epsilon = 1e-6);
    }
}

#[test]
fn gmres_vs_direct_on_nonsymmetric() {
    let n = 10;
    let mut rng = rand::thread_rng();
    let data: Vec<f64> = (0..n*n).map(|_| rng.gen()).collect();
    let a = Mat::from_fn(n, n, |i, j| data[j * n + i]);
    let b: Vec<f64> = (0..n).map(|_| rng.gen()).collect();
    let mut x_gmres = vec![0.0; n];
    let mut solver = GmresSolver::new(100, 1e-8, 1000);
    let stats = solver.solve(&a, None, &b, &mut x_gmres).unwrap();
    assert!(stats.converged);
    // direct solve
    let mut x_direct = b.clone();
    let qr = faer::linalg::solvers::Qr::new(a.as_ref());
    let x_mat = faer::MatMut::from_column_major_slice_mut(&mut x_direct, n, 1);
    qr.solve_in_place_with_conj(faer::Conj::No, x_mat);
    for i in 0..n {
        assert_abs_diff_eq!(x_gmres[i], x_direct[i], epsilon = 1e-6);
    }
}
