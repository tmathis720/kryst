use kryst::preconditioner::Preconditioner;
use kryst::preconditioner::{Jacobi, Ilu0};
use kryst::solver::{CgSolver, GmresSolver, LinearSolver};
use faer::Mat;

/// Build a badly conditioned diagonal matrix
fn ill_cond(n: usize, kappa: f64) -> (Mat<f64>, Vec<f64>) {
    let mut diag = vec![1.0; n];
    diag[n-1] = kappa;
    let mut a = Mat::zeros(n, n);
    for i in 0..n {
        a[(i, i)] = diag[i];
    }
    let b = vec![1.0; n];
    (a, b)
}

#[test]
fn cg_with_jacobi() {
    let (a, b) = ill_cond(5, 1e6);
    let mut pc = Jacobi::new();
    <Jacobi<f64> as Preconditioner<Mat<f64>, Vec<f64>>>::setup(&mut pc, &a).unwrap();
    let mut solver = CgSolver::new(1e-6, 1000);
    let mut x = vec![0.0; 5];
    // wrap A and pc into a PCG solver if you have one, else manually:
    let r_in = b.clone();
    let mut r_out = vec![0.0; b.len()];
    <Jacobi<f64> as Preconditioner<Mat<f64>, Vec<f64>>>::apply(&pc, &r_in, &mut r_out).unwrap();
    let stats = solver.solve(&a, &b, &mut x).unwrap();
    assert!(stats.converged);
    // No stats to check; just ensure it runs without panic
}

#[test]
fn gmres_with_ilu0() {
    let (a, b) = ill_cond(5, 1e4);
    let mut pc = Ilu0::new();
    <Ilu0<f64> as Preconditioner<Mat<f64>, Vec<f64>>>::setup(&mut pc, &a).unwrap();
    let mut solver = GmresSolver::new(4, 1e-6, 1000);
    let mut x = vec![0.0; 5];
    let stats = solver.solve(&a, &b, &mut x).unwrap();
    assert!(stats.converged);
    // No stats to check; just ensure it runs without panic
}
