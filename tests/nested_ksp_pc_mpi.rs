#![cfg(all(feature = "mpi", not(feature = "complex")))]

mod fixtures;

use std::sync::Arc;

use kryst::config::options::{KspOptions, PcOptions};
use kryst::context::ksp_context::{KspContext, SolverType};
use kryst::matrix::dist_csr::DistCsrOp;
use kryst::matrix::sparse::CsrMatrix;
use kryst::parallel::{Comm, MpiComm, UniverseComm};

fn solve_with_nested_policy(
    mode: &str,
    threads: usize,
) -> kryst::utils::convergence::SolveStats<f64> {
    let comm = UniverseComm::Mpi(Arc::new(MpiComm::new()));
    let n_local = 4;
    let a = Arc::new(make_dist_poisson(&comm, n_local));

    let mut ksp = KspContext::new();
    ksp.set_type(SolverType::Gmres).expect("outer type");
    let ksp_opts = KspOptions {
        maxits: Some(40),
        rtol: Some(1e-10),
        threads_mode: Some("context".into()),
        threads: Some(threads),
        ..Default::default()
    };
    let pc_opts = PcOptions {
        pc_type: Some("ksp".into()),
        pc_ksp_ksp_type: Some("gmres".into()),
        pc_ksp_pc_type: Some("jacobi".into()),
        pc_ksp_threads_mode: Some(mode.into()),
        pc_ksp_threads: Some(threads),
        pc_ksp_maxits: Some(2),
        pc_ksp_rtol: Some(1e-2),
        ..Default::default()
    };

    ksp.set_from_all_options(&ksp_opts, &pc_opts).expect("opts");
    ksp.set_operators(a, None);
    let rhs = vec![1.0; n_local];
    let mut x = vec![0.0; n_local];
    ksp.solve(&rhs, &mut x).expect("solve")
}

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

#[test]
fn nested_ksp_pc_mpi_uses_scoped_inner_options_and_side() {
    let comm = UniverseComm::Mpi(Arc::new(MpiComm::new()));
    let n_local = 4;
    let a = Arc::new(make_dist_poisson(&comm, n_local));

    let mut ksp = KspContext::new();
    ksp.set_type(SolverType::Gmres).expect("outer type");

    let ksp_opts = KspOptions {
        pc_side: Some("right".into()),
        maxits: Some(40),
        rtol: Some(1e-10),
        ..Default::default()
    };
    let pc_opts = PcOptions {
        pc_type: Some("ksp".into()),
        pc_ksp_ksp_type: Some("gmres".into()),
        pc_ksp_pc_type: Some("jacobi".into()),
        pc_ksp_maxits: Some(2),
        pc_ksp_rtol: Some(1e-2),
        pc_ksp_pc_side: Some("left".into()),
        ..Default::default()
    };

    ksp.set_from_all_options(&ksp_opts, &pc_opts).expect("opts");
    ksp.set_operators(a, None);

    let rhs = vec![1.0; n_local];
    let mut x = vec![0.0; n_local];
    let stats = ksp.solve(&rhs, &mut x).expect("solve");

    assert!(
        stats.reason.is_converged()
            || matches!(
                stats.reason,
                kryst::utils::convergence::ConvergedReason::DivergedMaxIts
            )
    );
    assert!(stats.nested_pc_failure.is_none());
    assert!(stats.final_residual.is_finite());
}

#[test]
fn nested_ksp_pc_mpi_fgmres_inner_gmres_variant_path() {
    let comm = UniverseComm::Mpi(Arc::new(MpiComm::new()));
    let n_local = 3;
    let a = Arc::new(make_dist_poisson(&comm, n_local));

    let mut ksp = KspContext::new();
    ksp.set_type(SolverType::Fgmres).expect("outer type");

    let ksp_opts = KspOptions {
        pc_side: Some("right".into()),
        maxits: Some(30),
        rtol: Some(1e-8),
        ..Default::default()
    };
    let pc_opts = PcOptions {
        pc_type: Some("ksp".into()),
        pc_ksp_ksp_options: Some(KspOptions {
            ksp_type: Some("gmres".into()),
            pc_side: Some("left".into()),
            gmres_variant: Some("classical".into()),
            gmres_restart: Some(2),
            maxits: Some(2),
            rtol: Some(1e-2),
            ..Default::default()
        }),
        pc_ksp_pc_type: Some("jacobi".into()),
        ..Default::default()
    };

    ksp.set_from_all_options(&ksp_opts, &pc_opts).expect("opts");
    ksp.set_operators(a, None);

    let rhs = vec![1.0; n_local];
    let mut x = vec![0.0; n_local];
    let stats = ksp.solve(&rhs, &mut x).expect("solve");

    assert!(
        stats.reason.is_converged()
            || matches!(
                stats.reason,
                kryst::utils::convergence::ConvergedReason::DivergedMaxIts
            )
    );
    assert!(stats.nested_pc_failure.is_none());
}

#[test]
fn nested_ksp_pc_mpi_inner_maxits_failure_maps_consistently() {
    let comm = UniverseComm::Mpi(Arc::new(MpiComm::new()));
    let n_local = 4;
    let a = Arc::new(make_dist_poisson(&comm, n_local));

    let mut ksp = KspContext::new();
    ksp.set_type(SolverType::Gmres).expect("outer type");
    let ksp_opts = KspOptions {
        maxits: Some(8),
        rtol: Some(1e-12),
        ..Default::default()
    };
    let pc_opts = PcOptions {
        pc_type: Some("ksp".into()),
        pc_ksp_ksp_type: Some("gmres".into()),
        pc_ksp_maxits: Some(1),
        pc_ksp_rtol: Some(0.0),
        pc_ksp_atol: Some(0.0),
        pc_ksp_allow_maxits: Some(false),
        pc_ksp_propagate_converged_reason: Some(true),
        pc_ksp_pc_type: Some("jacobi".into()),
        ..Default::default()
    };

    ksp.set_from_all_options(&ksp_opts, &pc_opts).expect("opts");
    ksp.set_operators(a, None);
    let rhs = vec![1.0; n_local];
    let mut x = vec![0.0; n_local];
    let stats = ksp.solve(&rhs, &mut x).expect("solve");

    if stats.reason.is_converged() {
        assert!(stats.nested_pc_failure.is_none());
    } else {
        assert!(stats.reason.is_diverged());
        if let Some(inner) = stats.nested_pc_failure.as_ref() {
            assert!(matches!(
                inner.reason,
                kryst::utils::convergence::ConvergedReason::DivergedMaxIts
                    | kryst::utils::convergence::ConvergedReason::DivergedPcFailed
                    | kryst::utils::convergence::ConvergedReason::DivergedBreakdown
            ));
        }
    }
}

#[test]
fn nested_ksp_pc_mpi_inner_tol_policy_allow_maxits_overrides_compat_flag() {
    let comm = UniverseComm::Mpi(Arc::new(MpiComm::new()));
    let n_local = 4;
    let a = Arc::new(make_dist_poisson(&comm, n_local));

    let mut ksp = KspContext::new();
    ksp.set_type(SolverType::Gmres).expect("outer type");
    let ksp_opts = KspOptions {
        maxits: Some(8),
        rtol: Some(1e-12),
        ..Default::default()
    };
    let pc_opts = PcOptions {
        pc_type: Some("ksp".into()),
        pc_ksp_ksp_type: Some("gmres".into()),
        pc_ksp_maxits: Some(1),
        pc_ksp_rtol: Some(0.0),
        pc_ksp_atol: Some(0.0),
        pc_ksp_allow_maxits: Some(false),
        pc_ksp_inner_tol_policy: Some("allow_maxits".into()),
        pc_ksp_pc_type: Some("jacobi".into()),
        ..Default::default()
    };

    ksp.set_from_all_options(&ksp_opts, &pc_opts).expect("opts");
    ksp.set_operators(a, None);
    let rhs = vec![1.0; n_local];
    let mut x = vec![0.0; n_local];
    let stats = ksp.solve(&rhs, &mut x).expect("solve");

    assert!(stats.nested_pc_failure.is_none());
}

#[test]
fn nested_ksp_pc_mpi_symmetric_outer_aligns_with_inner_fgmres_side() {
    let comm = UniverseComm::Mpi(Arc::new(MpiComm::new()));
    let n_local = 4;
    let a = Arc::new(make_dist_poisson(&comm, n_local));

    let mut ksp = KspContext::new();
    ksp.set_type(SolverType::Fgmres).expect("outer type");
    let ksp_opts = KspOptions {
        pc_side: Some("symmetric".into()),
        maxits: Some(10),
        rtol: Some(1e-8),
        ..Default::default()
    };
    let pc_opts = PcOptions {
        pc_type: Some("ksp".into()),
        pc_ksp_ksp_options: Some(KspOptions {
            ksp_type: Some("fgmres".into()),
            maxits: Some(2),
            rtol: Some(1e-2),
            ..Default::default()
        }),
        pc_ksp_pc_type: Some("jacobi".into()),
        ..Default::default()
    };

    ksp.set_from_all_options(&ksp_opts, &pc_opts).expect("opts");
    ksp.set_operators(a, None);
    let rhs = vec![1.0; n_local];
    let mut x = vec![0.0; n_local];
    let stats = ksp.solve(&rhs, &mut x).expect("solve");
    assert!(stats.nested_pc_failure.is_none());
}

#[test]
fn nested_ksp_pc_mpi_symmetric_outer_reports_inner_side_mismatch() {
    let comm = UniverseComm::Mpi(Arc::new(MpiComm::new()));
    let n_local = 4;
    let a = Arc::new(make_dist_poisson(&comm, n_local));

    let mut ksp = KspContext::new();
    ksp.set_type(SolverType::Gmres).expect("outer type");
    let ksp_opts = KspOptions {
        pc_side: Some("symmetric".into()),
        // Keep this test bounded to a single outer iteration so it cannot linger
        // if the mismatch guard regresses and falls through to solve-time behavior.
        maxits: Some(1),
        rtol: Some(0.0),
        ..Default::default()
    };
    let pc_opts = PcOptions {
        pc_type: Some("ksp".into()),
        pc_ksp_ksp_options: Some(KspOptions {
            ksp_type: Some("gmres".into()),
            pc_side: Some("right".into()),
            maxits: Some(2),
            rtol: Some(1e-2),
            ..Default::default()
        }),
        pc_ksp_pc_type: Some("jacobi".into()),
        ..Default::default()
    };

    ksp.set_from_all_options(&ksp_opts, &pc_opts).expect("opts");
    ksp.set_operators(a, None);
    let rhs = vec![1.0; n_local];
    let mut x = vec![0.0; n_local];
    let stats = ksp.solve(&rhs, &mut x).expect("solve");

    assert_eq!(
        stats.reason,
        kryst::utils::convergence::ConvergedReason::DivergedPcFailed
    );
    let inner = stats.nested_pc_failure.as_ref().expect("nested failure");
    assert!(inner.detail.contains("compatibility=mismatch"));
    assert!(inner.detail.contains("stage=preflight"));
}

#[test]
fn nested_ksp_pc_mpi_threads_mode_hybrid_matches_serial_convergence() {
    let serial_stats = solve_with_nested_policy("serial", 4);
    let hybrid_stats = solve_with_nested_policy("hybrid", 4);

    assert!(serial_stats.reason.is_converged() || serial_stats.reason.is_diverged());
    assert!(hybrid_stats.reason.is_converged() || hybrid_stats.reason.is_diverged());
    let s = serial_stats.final_residual;
    let h = hybrid_stats.final_residual;
    let scale = s.abs().max(h.abs()).max(1.0);
    assert!((s - h).abs() <= 1e-6 * scale);
}

#[test]
fn nested_ksp_pc_mpi_context_mode_rejects_multithread_inner() {
    let comm = UniverseComm::Mpi(Arc::new(MpiComm::new()));
    let n_local = 4;
    let a = Arc::new(make_dist_poisson(&comm, n_local));

    let mut ksp = KspContext::new();
    ksp.set_type(SolverType::Gmres).expect("outer type");
    let ksp_opts = KspOptions {
        maxits: Some(10),
        rtol: Some(1e-8),
        ..Default::default()
    };
    let pc_opts = PcOptions {
        pc_type: Some("ksp".into()),
        pc_ksp_ksp_type: Some("richardson".into()),
        pc_ksp_pc_type: Some("jacobi".into()),
        pc_ksp_threads_mode: Some("context".into()),
        pc_ksp_threads: Some(2),
        ..Default::default()
    };

    ksp.set_from_all_options(&ksp_opts, &pc_opts).expect("opts");
    ksp.set_operators(a, None);
    let rhs = vec![1.0; n_local];
    let mut x = vec![0.0; n_local];
    if comm.size() > 1 {
        let err = ksp.solve(&rhs, &mut x).unwrap_err();
        assert!(format!("{err}").contains("requires ksp_threads_mode=serial or hybrid"));
    } else {
        let stats = ksp.solve(&rhs, &mut x).expect("single-rank MPI permits context mode");
        assert!(stats.reason.is_converged() || stats.reason.is_diverged());
    }
}
