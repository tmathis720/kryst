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

