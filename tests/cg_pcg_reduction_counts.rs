#![cfg(all(feature = "backend-faer", not(feature = "complex")))]
mod support;
use faer::Mat;
use kryst::algebra::prelude::*;
use kryst::context::ksp_context::Workspace;
use kryst::parallel::{NoComm, UniverseComm};
use kryst::preconditioner::PcSide;
use kryst::solver::LinearSolver;
use kryst::solver::PcgSolver;
use std::sync::atomic::Ordering;
use support::reduce_counter::CountingComm;

fn build_spd(n: usize) -> Mat<f64> {
    let mut a = Mat::<f64>::zeros(n, n);
    for i in 0..n {
        a[(i, i)] = R::from(2.0);
        if i + 1 < n {
            a[(i, i + 1)] = R::from(1.0);
            a[(i + 1, i)] = R::from(1.0);
        }
    }
    a
}

#[test]
fn pcg_reduction_counts() {
    let n = 5;
    let a = build_spd(n);
    let b = vec![R::from(1.0); n];
    let mut x = vec![R::default(); n];

    let base = UniverseComm::NoComm(NoComm);
    let comm = CountingComm::new(base.clone());
    let mut solver = PcgSolver::new(1e-12, 20);
    let mut wk = Workspace::default();
    solver.setup_workspace(&mut wk);
    let stats = solver
        .solve_with_comm(
            &a,
            None,
            &b,
            &mut x,
            PcSide::Left,
            &comm,
            None,
            Some(&mut wk),
        )
        .unwrap();
    let expected = 2 + 2 * stats.iterations;
    assert_eq!(comm.reduces.load(Ordering::Relaxed), expected);
    assert_eq!(stats.counters.num_global_reductions, 0);
}
