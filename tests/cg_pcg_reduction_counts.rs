mod support;
use support::reduce_counter::CountingComm;
use kryst::solver::LinearSolver;
use kryst::parallel::{NoComm, UniverseComm};
use kryst::solver::{CgSolver, PcgSolver};
use kryst::preconditioner::PcSide;
use kryst::context::ksp_context::Workspace;
use std::sync::atomic::Ordering;
use faer::Mat;

fn build_spd(n: usize) -> Mat<f64> {
    let mut a = Mat::<f64>::zeros(n, n);
    for i in 0..n {
        a[(i, i)] = 2.0;
        if i + 1 < n {
            a[(i, i + 1)] = 1.0;
            a[(i + 1, i)] = 1.0;
        }
    }
    a
}

#[test]
fn cg_reduction_counts() {
    let n = 5;
    let a = build_spd(n);
    let b = vec![1.0; n];
    let mut x = vec![0.0; n];

    let base = UniverseComm::NoComm(NoComm);
    let comm = CountingComm::new(base.clone());
    let mut solver = CgSolver::new(1e-12, 20);
    solver.set_single_reduction(true);
    let mut wk = Workspace::default();
    solver.setup_workspace(&mut wk);
    let stats = solver
        .solve_with_comm(&a, None, &b, &mut x, PcSide::Left, &comm, None, Some(&mut wk))
        .unwrap();
    let expected = 1 + stats.iterations;
    assert_eq!(comm.reduces.load(Ordering::Relaxed), expected);

}

#[test]
fn pcg_reduction_counts() {
    let n = 5;
    let a = build_spd(n);
    let b = vec![1.0; n];
    let mut x = vec![0.0; n];

    let base = UniverseComm::NoComm(NoComm);
    let comm = CountingComm::new(base.clone());
    let mut solver = PcgSolver::new(1e-12, 20);
    solver.set_single_reduction(true);
    let mut wk = Workspace::default();
    solver.setup_workspace(&mut wk);
    let stats = solver
        .solve_with_comm(&a, None, &b, &mut x, PcSide::Left, &comm, None, Some(&mut wk))
        .unwrap();
    let expected = 2 + stats.iterations;
    assert_eq!(comm.reduces.load(Ordering::Relaxed), expected);

}

#[test]
fn cg_classic_reduction_counts() {
    let n = 5;
    let a = build_spd(n);
    let b = vec![1.0; n];
    let mut x = vec![0.0; n];

    let base = UniverseComm::NoComm(NoComm);
    let comm = CountingComm::new(base.clone());
    let mut solver = CgSolver::new(1e-12, 20);
    solver.set_single_reduction(false);
    let mut wk = Workspace::default();
    solver.setup_workspace(&mut wk);
    let stats = solver
        .solve_with_comm(&a, None, &b, &mut x, PcSide::Left, &comm, None, Some(&mut wk))
        .unwrap();
    let expected = 1 + 2 * stats.iterations;
    assert_eq!(comm.reduces.load(Ordering::Relaxed), expected);
}

#[test]
fn pcg_classic_reduction_counts() {
    let n = 5;
    let a = build_spd(n);
    let b = vec![1.0; n];
    let mut x = vec![0.0; n];

    let base = UniverseComm::NoComm(NoComm);
    let comm = CountingComm::new(base.clone());
    let mut solver = PcgSolver::new(1e-12, 20);
    solver.set_single_reduction(false);
    let mut wk = Workspace::default();
    solver.setup_workspace(&mut wk);
    let stats = solver
        .solve_with_comm(&a, None, &b, &mut x, PcSide::Left, &comm, None, Some(&mut wk))
        .unwrap();
    let expected = 2 + 2 * stats.iterations;
    assert_eq!(comm.reduces.load(Ordering::Relaxed), expected);
}

#[test]
fn cg_single_reduction_numeric_parity() {
    let n = 5;
    let a = build_spd(n);
    let b = vec![1.0; n];
    let base = UniverseComm::NoComm(NoComm);

    // Classic two-reduction CG
    let mut x_classic = vec![0.0; n];
    let mut solver_classic = CgSolver::new(1e-12, 20);
    solver_classic.set_single_reduction(false);
    let mut wk_classic = Workspace::default();
    solver_classic.setup_workspace(&mut wk_classic);
    let stats_classic = solver_classic
        .solve_with_comm(&a, None, &b, &mut x_classic, PcSide::Left, &base, None, Some(&mut wk_classic))
        .unwrap();

    // Single-reduction CG
    let mut x_single = vec![0.0; n];
    let mut solver_single = CgSolver::new(1e-12, 20);
    solver_single.set_single_reduction(true);
    let mut wk_single = Workspace::default();
    solver_single.setup_workspace(&mut wk_single);
    let stats_single = solver_single
        .solve_with_comm(&a, None, &b, &mut x_single, PcSide::Left, &base, None, Some(&mut wk_single))
        .unwrap();

    assert_eq!(stats_classic.iterations, stats_single.iterations);
    assert!((stats_classic.final_residual - stats_single.final_residual).abs() < 1e-3);

    let max_diff = x_classic
        .iter()
        .zip(x_single.iter())
        .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
    assert!(max_diff < 1e-3);
}

#[test]
fn pcg_single_reduction_numeric_parity() {
    let n = 5;
    let a = build_spd(n);
    let b = vec![1.0; n];
    let base = UniverseComm::NoComm(NoComm);

    // Classic two-reduction PCG
    let mut x_classic = vec![0.0; n];
    let mut solver_classic = PcgSolver::new(1e-12, 20);
    solver_classic.set_single_reduction(false);
    let mut wk_classic = Workspace::default();
    solver_classic.setup_workspace(&mut wk_classic);
    let stats_classic = solver_classic
        .solve_with_comm(&a, None, &b, &mut x_classic, PcSide::Left, &base, None, Some(&mut wk_classic))
        .unwrap();

    // Single-reduction PCG
    let mut x_single = vec![0.0; n];
    let mut solver_single = PcgSolver::new(1e-12, 20);
    solver_single.set_single_reduction(true);
    let mut wk_single = Workspace::default();
    solver_single.setup_workspace(&mut wk_single);
    let stats_single = solver_single
        .solve_with_comm(&a, None, &b, &mut x_single, PcSide::Left, &base, None, Some(&mut wk_single))
        .unwrap();

    assert_eq!(stats_classic.iterations, stats_single.iterations);
    assert!((stats_classic.final_residual - stats_single.final_residual).abs() < 1e-3);

    let max_diff = x_classic
        .iter()
        .zip(x_single.iter())
        .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
    assert!(max_diff < 1e-3);
}

