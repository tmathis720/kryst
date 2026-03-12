#![cfg(all(feature = "backend-faer", not(feature = "complex")))]
use faer::Mat;
use fixtures::csr_poisson_1d;
use kryst::algebra::prelude::*;
use kryst::config::options::CgVariant;
use kryst::context::ksp_context::Workspace;
use kryst::ops::wrap::as_s_op;
use kryst::parallel::{NoComm, UniverseComm};
use kryst::preconditioner::PcSide;
use kryst::solver::LinearSolver;
use kryst::solver::cg::{self, CgSolver};

fn build_dense_poisson(n: usize) -> Mat<f64> {
    let mut a = Mat::<f64>::zeros(n, n);
    for i in 0..n {
        a[(i, i)] = 2.0;
        if i > 0 {
            a[(i, i - 1)] = -1.0;
        }
        if i + 1 < n {
            a[(i, i + 1)] = -1.0;
        }
    }
    a
}

#[test]
fn pipelined_matches_classic_solution() {
    let n = 32;
    let a = build_dense_poisson(n);
    let b: Vec<S> = vec![S::from_real(1.0); n];

    let comm = UniverseComm::NoComm(NoComm);

    let mut classic = CgSolver::new(1e-10, 200);
    let mut pipelined = CgSolver::new(1e-10, 200).with_variant(CgVariant::Pipelined);

    let mut x_classic = vec![S::zero(); n];
    let mut x_pipe = vec![S::zero(); n];

    let mut wk_classic = Workspace::default();
    classic.setup_workspace(&mut wk_classic);
    let op = as_s_op(&a);
    let stats_classic = classic
        .solve_with_comm(
            &op,
            None,
            &b,
            &mut x_classic,
            PcSide::Left,
            &comm,
            None,
            Some(&mut wk_classic),
        )
        .expect("classic CG converged");

    let mut wk_pipe = Workspace::default();
    pipelined.setup_workspace(&mut wk_pipe);
    let op_pipe = as_s_op(&a);
    let stats_pipe = pipelined
        .solve_with_comm(
            &op_pipe,
            None,
            &b,
            &mut x_pipe,
            PcSide::Left,
            &comm,
            None,
            Some(&mut wk_pipe),
        )
        .expect("pipelined CG converged");

    assert!(
        stats_pipe.iterations <= stats_classic.iterations + 2,
        "classic iterations: {}, pipelined iterations: {}",
        stats_classic.iterations,
        stats_pipe.iterations
    );

    let mut max_diff = R::zero();
    for (xc, xp) in x_classic.iter().zip(&x_pipe) {
        max_diff = max_diff.max((*xc - *xp).abs());
    }
    assert!(max_diff < R::from(1e-8));

    let rel_res = stats_pipe.final_residual / stats_classic.final_residual.max(R::from(1e-30));
    assert!(rel_res < R::from(2.0));
}

#[test]
fn pipelined_handles_complex_drift() {
    let n = 16;
    let a = csr_poisson_1d(n);
    let b: Vec<S> = vec![S::from_real(1.0); n];
    let comm = UniverseComm::NoComm(NoComm);

    cg::debug::reset_counters();

    let mut solver = CgSolver::new(1e-10, 100).with_variant(CgVariant::Pipelined);
    let mut wk = Workspace::default();
    solver.setup_workspace(&mut wk);
    let mut x = vec![S::zero(); n];
    let op = as_s_op(&a);

    let stats = solver
        .solve_with_comm(
            &op,
            None,
            &b,
            &mut x,
            PcSide::Left,
            &comm,
            None,
            Some(&mut wk),
        )
        .expect("pipelined CG converged");

    assert!(stats.final_residual < R::from(1e-6));
    assert_eq!(cg::debug::large_imag_count(), 0);
}

#[test]
fn pipelined_periodic_refresh_bounds_residual_drift() {
    let n = 128;
    let a = build_dense_poisson(n);
    let b: Vec<S> = vec![S::from_real(1.0); n];
    let comm = UniverseComm::NoComm(NoComm);

    let mut plain = CgSolver::new(1e-10, 300).with_variant(CgVariant::Pipelined);
    let mut refreshed = CgSolver::new(1e-10, 300).with_variant(CgVariant::Pipelined);
    refreshed.set_pipelined_residual_refresh_every(Some(8));

    let mut x_plain = vec![S::zero(); n];
    let mut ws_plain = Workspace::default();
    plain.setup_workspace(&mut ws_plain);
    let stats_plain = plain
        .solve_with_comm(
            &as_s_op(&a),
            None,
            &b,
            &mut x_plain,
            PcSide::Left,
            &comm,
            None,
            Some(&mut ws_plain),
        )
        .expect("plain pipelined CG converged");

    let mut x_refresh = vec![S::zero(); n];
    let mut ws_refresh = Workspace::default();
    refreshed.setup_workspace(&mut ws_refresh);
    let stats_refresh = refreshed
        .solve_with_comm(
            &as_s_op(&a),
            None,
            &b,
            &mut x_refresh,
            PcSide::Left,
            &comm,
            None,
            Some(&mut ws_refresh),
        )
        .expect("refreshed pipelined CG converged");

    assert!(stats_refresh.final_residual <= stats_plain.final_residual * R::from(1.5));
    assert!(stats_refresh.iterations <= 300);
}

mod fixtures;
