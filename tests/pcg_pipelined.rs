#![cfg(feature = "backend-faer")]
use faer::Mat;
use fixtures::csr_poisson_1d;
use kryst::algebra::prelude::*;
use kryst::context::ksp_context::Workspace;
use kryst::error::KError;
use kryst::matrix::op::LinOp;
use kryst::parallel::{NoComm, UniverseComm};
use kryst::preconditioner::PcSide;
use kryst::preconditioner::Preconditioner;
use kryst::preconditioner::jacobi::Jacobi;
use kryst::solver::LinearSolver;
use kryst::solver::pcg::{PcgSolver, PcgVariant};
use kryst::utils::reduction::{install_test_counter, test_hooks};

fn build_dense_poisson(n: usize) -> Mat<R> {
    let mut a = Mat::<R>::zeros(n, n);
    for i in 0..n {
        a[(i, i)] = R::from(2.0);
        if i > 0 {
            a[(i, i - 1)] = R::from(-1.0);
        }
        if i + 1 < n {
            a[(i, i + 1)] = R::from(-1.0);
        }
    }
    a
}

#[test]
fn pipelined_matches_classic_solution() {
    let n = 32;
    let a = build_dense_poisson(n);
    let b: Vec<R> = vec![R::from(1.0); n];

    let comm = UniverseComm::NoComm(NoComm);

    let mut classic = PcgSolver::new(1e-10, 200);
    let mut pipeline = PcgSolver::new(1e-10, 200);
    pipeline.set_variant(PcgVariant::Pipelined { replace_every: 0 });

    let mut x_classic: Vec<R> = vec![R::default(); n];
    let mut x_pipe: Vec<R> = vec![R::default(); n];

    let mut wk_classic = Workspace::default();
    classic.setup_workspace(&mut wk_classic);
    let stats_classic = classic
        .solve_with_comm(
            &a,
            None,
            &b,
            &mut x_classic,
            PcSide::Left,
            &comm,
            None,
            Some(&mut wk_classic),
        )
        .expect("classic PCG converged");

    let mut wk_pipe = Workspace::default();
    pipeline.setup_workspace(&mut wk_pipe);
    let stats_pipe = pipeline
        .solve_with_comm(
            &a,
            None,
            &b,
            &mut x_pipe,
            PcSide::Left,
            &comm,
            None,
            Some(&mut wk_pipe),
        )
        .expect("pipelined PCG converged");

    assert!(
        stats_pipe.iterations <= stats_classic.iterations + 2,
        "classic iterations: {}, pipelined iterations: {}",
        stats_classic.iterations,
        stats_pipe.iterations
    );
    let mut diff = R::default();
    for (xc, xp) in x_classic.iter().zip(&x_pipe) {
        diff = diff.max((xc - xp).abs());
    }
    assert!(diff < R::from(1e-8));
    let rel_res = stats_pipe.final_residual / stats_classic.final_residual.max(R::from(1e-30));
    assert!(rel_res < R::from(2.0));
}

#[test]
fn pipelined_reports_reduction_counts() -> Result<(), KError> {
    let n = 32;
    let a = csr_poisson_1d(n);
    let b: Vec<R> = vec![R::from(1.0); n];

    let comm = UniverseComm::NoComm(NoComm);
    install_test_counter(true);
    let mut solver =
        PcgSolver::new(1e-12, 100).with_variant(PcgVariant::Pipelined { replace_every: 0 });
    debug_assert!(matches!(
        solver.variant(),
        PcgVariant::Pipelined { replace_every: 0 }
    ));

    let mut wk = Workspace::default();
    solver.setup_workspace(&mut wk);
    let mut x: Vec<R> = vec![R::default(); n];

    let op: &dyn LinOp<S = f64> = &a;
    let mut pc = Jacobi::new();
    pc.setup(op)?;

    let stats = solver
        .solve_with_comm(
            op,
            Some(&mut pc),
            &b,
            &mut x,
            PcSide::Left,
            &comm,
            None,
            Some(&mut wk),
        )
        .expect("pipelined PCG converged");

    let (pair_count, vec_count) = test_hooks::wait_counters();
    install_test_counter(false);

    assert_eq!(vec_count, 0);

    // At least two asynchronous reductions per iteration plus the initial setup
    // reductions should be observed.
    let expected_async = stats.iterations * 2;

    assert!(
        pair_count >= expected_async,
        "pair_count {} smaller than expected {}",
        pair_count,
        expected_async
    );
    assert!(
        stats.counters.num_global_reductions >= pair_count,
        "reported {} reductions, less than observed {} async waits",
        stats.counters.num_global_reductions,
        pair_count
    );
    assert!(
        stats.counters.num_global_reductions > 0,
        "serialized solver reported zero reductions"
    );
    Ok(())
}

mod fixtures;
