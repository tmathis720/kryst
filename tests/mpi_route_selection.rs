#![cfg(all(feature = "backend-faer", feature = "mpi", not(feature = "complex")))]

mod fixtures;

use std::sync::Arc;

use kryst::config::options::{KspOptions, PcOptions};
use kryst::context::ksp_context::{KspContext, SolverType};
use kryst::matrix::dist_csr::DistCsrOp;
use kryst::matrix::sparse::CsrMatrix;
use kryst::parallel::{Comm, MpiComm, UniverseComm};
use kryst::utils::convergence::ConvergedReason;

fn local_rows_from_global(
    global: &CsrMatrix<f64>,
    row_start: usize,
    n_local: usize,
) -> CsrMatrix<f64> {
    let mut row_ptr = Vec::with_capacity(n_local + 1);
    let mut col_idx = Vec::new();
    let mut values = Vec::new();
    row_ptr.push(0);
    for i in 0..n_local {
        let (cols, vals) = global.row(row_start + i);
        col_idx.extend_from_slice(cols);
        values.extend_from_slice(vals);
        row_ptr.push(col_idx.len());
    }
    CsrMatrix::from_csr(n_local, global.ncols(), row_ptr, col_idx, values)
}

fn make_dist_poisson(comm: &UniverseComm, n_per: usize) -> DistCsrOp {
    let rank = comm.rank();
    let size = comm.size();
    let n_global = n_per * size;
    let row_start = rank * n_per;
    let global = fixtures::csr_poisson_1d(n_global);
    let local = local_rows_from_global(&global, row_start, n_per);
    let part_prefix: Vec<usize> = (0..=size).map(|p| p * n_per).collect();
    DistCsrOp::from_local_rows(n_global, row_start, &local, &part_prefix, comm.clone())
        .expect("dist csr")
}

fn solve_with_pc(dist: Arc<DistCsrOp>, rhs: &[f64], pc_opts: &PcOptions) -> ConvergedReason {
    let mut ksp = KspContext::new();
    let ksp_opts = KspOptions {
        ksp_type: Some("gmres".to_string()),
        rtol: Some(1e-10),
        atol: Some(1e-12),
        maxits: Some(300),
        ..Default::default()
    };
    ksp.set_type(SolverType::Gmres).expect("set gmres");
    ksp.set_operators(dist, None);
    ksp.set_from_all_options(&ksp_opts, pc_opts)
        .expect("set options");
    ksp.setup().expect("ksp setup");

    let mut x = vec![0.0; rhs.len()];
    let stats = ksp.solve(rhs, &mut x).expect("solve");
    stats.reason
}

#[test]
fn mpi_jacobi_runs_without_pc_global_fallback() {
    let comm = UniverseComm::Mpi(Arc::new(MpiComm::new()));
    if comm.size() <= 1 {
        return;
    }
    let n_per = 6;
    let dist = Arc::new(make_dist_poisson(&comm, n_per));
    let rhs = vec![1.0; n_per];

    let pc_opts = PcOptions {
        pc_type: Some("jacobi".to_string()),
        ..Default::default()
    };

    let reason = solve_with_pc(dist, &rhs, &pc_opts);
    assert!(
        matches!(
            reason,
            ConvergedReason::ConvergedRtol | ConvergedReason::ConvergedAtol
        ),
        "jacobi should use native distributed route without requiring -pc_global, got {reason:?}"
    );
}

#[test]
fn mpi_block_jacobi_runs_without_pc_global_fallback() {
    let comm = UniverseComm::Mpi(Arc::new(MpiComm::new()));
    if comm.size() <= 1 {
        return;
    }
    let n_per = 6;
    let dist = Arc::new(make_dist_poisson(&comm, n_per));
    let rhs = vec![1.0; n_per];

    let pc_opts = PcOptions {
        pc_type: Some("block_jacobi".to_string()),
        ..Default::default()
    };

    let reason = solve_with_pc(dist, &rhs, &pc_opts);
    assert!(
        matches!(
            reason,
            ConvergedReason::ConvergedRtol | ConvergedReason::ConvergedAtol
        ),
        "block-jacobi should use native distributed route without requiring -pc_global, got {reason:?}"
    );
}

#[test]
fn mpi_local_only_pc_auto_promotes_to_native_unless_adapted_route_forced() {
    let comm = UniverseComm::Mpi(Arc::new(MpiComm::new()));
    if comm.size() <= 1 {
        return;
    }
    let n_per = 6;
    let dist = Arc::new(make_dist_poisson(&comm, n_per));
    let rhs = vec![1.0; n_per];

    let base_opts = KspOptions {
        ksp_type: Some("gmres".to_string()),
        maxits: Some(20),
        ..Default::default()
    };

    let promoted_reason = solve_with_pc(
        dist.clone(),
        &rhs,
        &PcOptions {
            pc_type: Some("mg".to_string()),
            ..Default::default()
        },
    );
    assert!(
        matches!(
            promoted_reason,
            ConvergedReason::ConvergedRtol | ConvergedReason::ConvergedAtol
        ),
        "MG should auto-promote to native distributed route by default, got {promoted_reason:?}"
    );

    let mut ksp = KspContext::new();
    ksp.set_type(SolverType::Gmres).expect("set gmres");
    ksp.set_operators(dist.clone(), None);
    let adapted_only = PcOptions {
        pc_type: Some("mg".to_string()),
        pc_dist_route: Some("adapted".to_string()),
        ..Default::default()
    };
    ksp.set_from_all_options(&base_opts, &adapted_only)
        .expect("set opts");
    let err = ksp
        .setup()
        .expect_err("MG adapted route should require explicit distributed fallback");
    assert!(
        err.to_string().contains("pc_dist_route native"),
        "unexpected error: {err}"
    );

    let fallback_opts = PcOptions {
        pc_type: Some("mg".to_string()),
        pc_global: Some("block_jacobi".to_string()),
        pc_local: Some("ilu".to_string()),
        ..Default::default()
    };
    let reason = solve_with_pc(dist, &rhs, &fallback_opts);
    assert!(
        matches!(
            reason,
            ConvergedReason::ConvergedRtol | ConvergedReason::ConvergedAtol
        ),
        "local-only MG preconditioner should converge when explicit pc_global fallback is enabled, got {reason:?}"
    );
}

#[cfg(feature = "backend-faer")]
#[test]
fn mpi_gamg_route_hint_accepts_explicit_choice() {
    let comm = UniverseComm::Mpi(Arc::new(MpiComm::new()));
    if comm.size() <= 1 {
        return;
    }
    let n_per = 6;
    let dist = Arc::new(make_dist_poisson(&comm, n_per));
    let rhs = vec![1.0; n_per];
    let reason = solve_with_pc(
        dist,
        &rhs,
        &PcOptions {
            pc_type: Some("gamg".to_string()),
            amg_dist_coarse_solver_route: Some("root,local".to_string()),
            ..Default::default()
        },
    );
    assert!(
        matches!(
            reason,
            ConvergedReason::ConvergedRtol | ConvergedReason::ConvergedAtol
        ),
        "GAMG with explicit coarse route hint should converge, got {reason:?}"
    );
}

#[test]
fn mpi_route_diagnostics_include_preflight_metadata() {
    let comm = UniverseComm::Mpi(Arc::new(MpiComm::new()));
    if comm.size() <= 1 {
        return;
    }
    let n_per = 4;
    let dist = Arc::new(make_dist_poisson(&comm, n_per));

    let mut ksp = KspContext::new();
    let ksp_opts = KspOptions {
        ksp_type: Some("gmres".to_string()),
        maxits: Some(20),
        ..Default::default()
    };
    let pc_opts = PcOptions {
        pc_type: Some("mg".to_string()),
        pc_global: Some("block_jacobi".to_string()),
        pc_local: Some("ilu".to_string()),
        ..Default::default()
    };
    ksp.set_type(SolverType::Gmres).expect("set gmres");
    ksp.set_operators(dist, None);
    ksp.set_from_all_options(&ksp_opts, &pc_opts)
        .expect("set options");
    ksp.setup().expect("setup");

    let diag = ksp.view();
    assert!(diag.solver_config.contains_key("pc_dist_preflight_outcome"));
    assert!(
        diag.solver_config
            .contains_key("pc_dist_preflight_reason_codes")
    );
    assert!(
        diag.solver_config
            .contains_key("pc_dist_preflight_native_ready")
    );
}
