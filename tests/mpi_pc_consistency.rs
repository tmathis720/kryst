#![cfg(not(feature = "complex"))]
#![cfg(feature = "mpi")]

mod fixtures;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock};

use kryst::config::options::{KspOptions, PcOptions};
use kryst::context::ksp_context::{KspContext, SolverType};
use kryst::matrix::LinOp;
use kryst::matrix::dist_csr::DistCsrOp;
use kryst::matrix::op::CsrOp;
use kryst::matrix::sparse::CsrMatrix;
use kryst::parallel::{Comm, MpiComm, UniverseComm};
use kryst::preconditioner::PcSide;
use kryst::preconditioner::dist::{
    DistLocalApplyMode, DistVec, GlobalPcKind, LocalPcKind, MpiPcOptions, build_block_jacobi_pc,
};
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

fn true_residual_norm(global: &CsrMatrix<f64>, x: &[f64], b: &[f64]) -> f64 {
    let op = CsrOp::new(Arc::new(global.clone()));
    let mut ax = vec![0.0; x.len()];
    op.matvec(x, &mut ax);
    b.iter()
        .zip(ax.iter())
        .map(|(&bi, &axi)| {
            let r = bi - axi;
            r * r
        })
        .sum::<f64>()
        .sqrt()
}

fn reason_id(reason: ConvergedReason) -> f64 {
    match reason {
        ConvergedReason::ConvergedRtol => 1.0,
        ConvergedReason::ConvergedAtol => 2.0,
        ConvergedReason::ConvergedTrustRegion => 3.0,
        ConvergedReason::ConvergedHappyBreakdown => 4.0,
        ConvergedReason::DivergedNan => 5.0,
        ConvergedReason::DivergedInf => 6.0,
        ConvergedReason::DivergedDtol => 7.0,
        ConvergedReason::DivergedMaxIts => 8.0,
        ConvergedReason::DivergedBreakdown => 9.0,
        ConvergedReason::DivergedArnoldiRankLoss => 10.0,
        ConvergedReason::DivergedBreakdownBiCG => 11.0,
        ConvergedReason::DivergedReducedSystemSingular => 12.0,
        ConvergedReason::DivergedIndefiniteMatrix => 13.0,
        ConvergedReason::DivergedIndefinitePC => 14.0,
        ConvergedReason::DivergedPcSetupFailed => 15.0,
        ConvergedReason::DivergedPcFailed => 16.0,
        ConvergedReason::StoppedByMonitor => 17.0,
        ConvergedReason::Continued => 18.0,
    }
}

fn mpi_test_guard() -> MutexGuard<'static, ()> {
    static MPI_TEST_MUTEX: OnceLock<Mutex<()>> = OnceLock::new();
    MPI_TEST_MUTEX
        .get_or_init(|| Mutex::new(()))
        .lock()
        .expect("mpi test mutex poisoned")
}

#[test]
fn mpi_convergence_reason_consistent_across_ranks() {
    let _guard = mpi_test_guard();
    let comm = UniverseComm::Mpi(Arc::new(MpiComm::new()));
    comm.set_reproducible(true);

    let n_per = 4;
    let dist = make_dist_poisson(&comm, n_per);
    let rhs: Vec<f64> = (0..n_per).map(|i| 1.0 + i as f64).collect();
    let mut x = vec![0.0; n_per];

    let mut ksp = KspContext::new();
    let ksp_opts = KspOptions::default();
    let mut pc_opts = PcOptions::default();
    pc_opts.pc_type = Some("jacobi".to_string());
    pc_opts.pc_global = Some("block_jacobi".to_string());
    pc_opts.pc_local = Some("ilu".to_string());
    ksp.set_from_options(&ksp_opts).expect("set options");
    ksp.rtol = 1e-10;
    ksp.atol = 1e-12;
    ksp.maxits = 500;
    ksp.set_type(SolverType::Cg).expect("set cg");
    ksp.set_operators(Arc::new(dist), None);
    ksp.setup().expect("ksp setup");

    let stats = ksp.solve(&rhs, &mut x).expect("ksp solve");
    assert!(stats.final_residual.is_finite());
    assert!(
        matches!(
            stats.reason,
            ConvergedReason::ConvergedRtol | ConvergedReason::ConvergedAtol
        ),
        "expected converged reason, got {:?}",
        stats.reason
    );

    let reason_sum = comm.all_reduce_f64(reason_id(stats.reason));
    let iter_sum = comm.all_reduce_f64(stats.iterations as f64);
    let size = comm.size() as f64;
    assert!(
        (reason_sum - reason_id(stats.reason) * size).abs() < 1e-12,
        "convergence reason differs across ranks"
    );
    assert!(
        (iter_sum - (stats.iterations as f64) * size).abs() < 1e-12,
        "iteration count differs across ranks"
    );
}

#[test]
fn mpi_pcg_matches_serial_residual_iterations_and_rank0_monitor_policy() {
    let _guard = mpi_test_guard();
    let comm = UniverseComm::Mpi(Arc::new(MpiComm::new()));
    comm.set_reproducible(true);

    let n_per = 4usize;
    let size = comm.size();
    let n_global = n_per * size;
    let rank = comm.rank();
    let row_start = rank * n_per;

    let global = fixtures::csr_poisson_1d(n_global);
    let local = local_rows_from_global(&global, row_start, n_per);
    let part_prefix: Vec<usize> = (0..=size).map(|p| p * n_per).collect();
    let dist = DistCsrOp::from_local_rows(n_global, row_start, &local, &part_prefix, comm.clone())
        .expect("dist csr");

    let rhs_global: Vec<f64> = (0..n_global).map(|i| 1.0 + i as f64 * 0.25).collect();
    let rhs_local = rhs_global[row_start..row_start + n_per].to_vec();

    let monitor_hits = Arc::new(AtomicUsize::new(0));
    let mut ksp_dist = KspContext::new();
    let mut ksp_opts = KspOptions::default();
    ksp_opts.ksp_monitor_rank0 = Some(true);
    ksp_dist.set_from_options(&ksp_opts).expect("set options");
    ksp_dist.rtol = 1e-10;
    ksp_dist.atol = 1e-12;
    ksp_dist.maxits = 200;
    ksp_dist.set_type(SolverType::Pcg).expect("set pcg");
    ksp_dist.set_operators(Arc::new(dist), None);
    let monitor_hits_clone = Arc::clone(&monitor_hits);
    ksp_dist.add_monitor(move |_, _, _| {
        monitor_hits_clone.fetch_add(1, Ordering::Relaxed);
        kryst::solver::MonitorAction::Continue
    });
    ksp_dist.setup().expect("dist setup");

    let mut x_local = vec![0.0; n_per];
    let stats_dist = ksp_dist
        .solve(&rhs_local, &mut x_local)
        .expect("dist solve");

    let mut gathered_x = Vec::new();
    comm.gather(&x_local, &mut gathered_x, 0);

    let mut ksp_serial = KspContext::new();
    ksp_serial.rtol = 1e-10;
    ksp_serial.atol = 1e-12;
    ksp_serial.maxits = 200;
    ksp_serial
        .set_type(SolverType::Pcg)
        .expect("set serial pcg");
    ksp_serial.set_operators(Arc::new(CsrOp::new(Arc::new(global.clone()))), None);
    ksp_serial.setup().expect("serial setup");
    let mut x_serial = vec![0.0; n_global];
    let stats_serial = ksp_serial
        .solve(&rhs_global, &mut x_serial)
        .expect("serial solve");

    let residual_pair = if rank == 0 {
        let dist_true = true_residual_norm(&global, &gathered_x, &rhs_global);
        let serial_true = true_residual_norm(&global, &x_serial, &rhs_global);
        vec![dist_true, serial_true]
    } else {
        vec![0.0, 0.0]
    };
    let dist_true = comm.all_reduce_f64(residual_pair[0]);
    let serial_true = comm.all_reduce_f64(residual_pair[1]);

    assert!(
        (dist_true - serial_true).abs() <= 1e-8,
        "distributed true residual {dist_true:.3e} differs from serial {serial_true:.3e}"
    );
    assert!(
        (stats_dist.final_residual - stats_serial.final_residual).abs() <= 1e-8,
        "reported final residual differs: dist {:.3e}, serial {:.3e}",
        stats_dist.final_residual,
        stats_serial.final_residual
    );
    assert_eq!(
        stats_dist.iterations, stats_serial.iterations,
        "PCG iterations should match between distributed and serial runs"
    );
    assert_eq!(
        stats_dist.reason, stats_serial.reason,
        "convergence reason should match between distributed and serial runs"
    );

    let hits = monitor_hits.load(Ordering::Relaxed);
    if rank == 0 {
        assert!(hits > 0, "rank 0 should receive monitor callbacks");
    } else {
        assert_eq!(hits, 0, "non-root ranks must not emit monitor callbacks");
    }

    let phases = stats_dist
        .reduction_phase_diagnostics()
        .expect("PCG should expose a reduction model");
    assert!(
        phases.startup > 0 && phases.iterative > 0,
        "expected startup/iterative phase reductions to be non-zero"
    );
    let model = stats_dist
        .reduction_model
        .as_ref()
        .expect("PCG should carry reduction model metadata");
    assert_eq!(
        phases.startup, model.startup,
        "startup phase should match reduction model"
    );
    assert_eq!(
        phases.iterative,
        (model.per_iteration * stats_dist.iterations as f64).ceil() as usize,
        "iterative phase should match model-per-iteration estimate"
    );
    if stats_dist.counters.num_global_reductions > 0 {
        assert_eq!(
            phases.startup + phases.iterative + phases.tail,
            stats_dist.counters.num_global_reductions,
            "phase reductions should sum to observed total when counters are enabled"
        );
    }
}

#[test]
fn mpi_gamg_distributed_policy_options_parse() {
    let opts = PcOptions::from_args(&[
        "-pc_type",
        "gamg",
        "-pc_amg_dist_apply_mode",
        "local_prototype",
        "-pc_amg_dist_coarse_repartition",
        "uniform",
        "-pc_gamg_coarse_solver_route",
        "local",
        "-pc_amg_dist_instrumentation",
        "true",
    ])
    .expect("parse distributed GAMG policy options");

    assert_eq!(opts.amg_dist_apply_mode.as_deref(), Some("local_prototype"));
    assert_eq!(opts.amg_dist_coarse_repartition.as_deref(), Some("uniform"));
    assert_eq!(opts.amg_dist_coarse_solver_route.as_deref(), Some("local"));
    assert_eq!(opts.amg_dist_instrumentation, Some(true));
}

#[test]
fn mpi_block_jacobi_native_strict_supported_local_pcs_consistent() {
    let _guard = mpi_test_guard();
    let comm = UniverseComm::Mpi(Arc::new(MpiComm::new()));
    comm.set_reproducible(true);

    let n_per = 5;
    let dist = make_dist_poisson(&comm, n_per);
    let rhs: Vec<f64> = (0..n_per).map(|i| 1.0 + i as f64).collect();

    for local_pc in [LocalPcKind::Fsai, LocalPcKind::Spai] {
        let mut mpi_opts = MpiPcOptions::default();
        mpi_opts.global_pc = GlobalPcKind::BlockJacobi;
        mpi_opts.local_pc = local_pc;
        mpi_opts.local_apply_mode = DistLocalApplyMode::NativeStrict;

        let pc = build_block_jacobi_pc(&dist, &mpi_opts)
            .expect("build block-jacobi PC")
            .expect("pc exists");

        let mut out = DistVec::new(
            comm.clone(),
            dist.local_row_offset(),
            dist.n_global,
            rhs.clone(),
        );
        pc.apply_global(PcSide::Left, &mut out)
            .expect("distributed apply");
        let local_sum: f64 = out.local_view().iter().copied().sum();
        let global_sum = comm.all_reduce_f64(local_sum);
        let size = comm.size() as f64;
        assert!(global_sum.is_finite(), "global sum must be finite");
        assert!(
            (global_sum / size).is_finite(),
            "scaled global sum must remain finite for {local_pc:?}"
        );
    }
}

#[test]
fn mpi_block_jacobi_native_halo_vs_strict_match_for_new_local_pcs() {
    let _guard = mpi_test_guard();
    let comm = UniverseComm::Mpi(Arc::new(MpiComm::new()));
    comm.set_reproducible(true);

    let n_per = 4;
    let dist = make_dist_poisson(&comm, n_per);
    let rhs: Vec<f64> = (0..n_per).map(|i| 0.5 + i as f64).collect();

    for local_pc in [LocalPcKind::Fsai, LocalPcKind::Spai] {
        let mut strict_opts = MpiPcOptions::default();
        strict_opts.global_pc = GlobalPcKind::BlockJacobi;
        strict_opts.local_pc = local_pc;
        strict_opts.local_apply_mode = DistLocalApplyMode::NativeStrict;

        let mut halo_opts = strict_opts.clone();
        halo_opts.local_apply_mode = DistLocalApplyMode::NativeLocalHalo;

        let strict_pc = build_block_jacobi_pc(&dist, &strict_opts)
            .expect("strict build")
            .expect("strict pc");
        let halo_pc = build_block_jacobi_pc(&dist, &halo_opts)
            .expect("halo build")
            .expect("halo pc");

        let mut strict_out = DistVec::new(
            comm.clone(),
            dist.local_row_offset(),
            dist.n_global,
            rhs.clone(),
        );
        let mut halo_out = DistVec::new(
            comm.clone(),
            dist.local_row_offset(),
            dist.n_global,
            rhs.clone(),
        );
        strict_pc
            .apply_global(PcSide::Left, &mut strict_out)
            .expect("strict apply");
        halo_pc
            .apply_global(PcSide::Left, &mut halo_out)
            .expect("halo apply");

        for (a, b) in strict_out.local_view().iter().zip(halo_out.local_view()) {
            assert!((a - b).abs() < 1e-11, "mode mismatch for {local_pc:?}");
        }
    }
}

#[test]
fn mpi_mg_distributed_coarse_route_options_parse() {
    let opts = PcOptions::from_args(&[
        "-pc_type",
        "mg",
        "-pc_mg_coarse_solver_route",
        "root_gather",
        "-pc_mg_levels_policy",
        "level=2,coarse_routes=root_gather|local_prototype|direct",
    ])
    .expect("parse distributed MG route options");

    assert_eq!(opts.pc_type.as_deref(), Some("mg"));
    assert_eq!(
        opts.amg_dist_coarse_solver_route.as_deref(),
        Some("root_gather")
    );
    assert!(opts.pc_mg_level_policies.is_some());
}

#[test]
fn mpi_pc_diagnostics_reports_negotiated_distributed_mode() {
    let _guard = mpi_test_guard();
    let comm = UniverseComm::Mpi(Arc::new(MpiComm::new()));
    comm.set_reproducible(true);
    if comm.size() <= 1 {
        return;
    }

    let dist = Arc::new(make_dist_poisson(&comm, 4));
    let mut ksp = KspContext::new();
    ksp.set_type(SolverType::Gmres).expect("set gmres");
    ksp.set_operators(dist, None);

    let ksp_opts = KspOptions {
        ksp_type: Some("gmres".to_string()),
        maxits: Some(20),
        ..Default::default()
    };
    let pc_opts = PcOptions {
        pc_type: Some("ilu".to_string()),
        pc_global: Some("block_jacobi".to_string()),
        pc_local: Some("ilu".to_string()),
        pc_dist_local_apply: Some("strict".to_string()),
        pc_dist_route: Some("native".to_string()),
        ..Default::default()
    };
    ksp.set_from_all_options(&ksp_opts, &pc_opts)
        .expect("set distributed options");
    ksp.setup().expect("setup with strict native distributed");

    let diag = ksp.view();
    let pc = diag.pc.expect("pc diagnostics");
    assert_eq!(pc.distributed_mode.as_deref(), Some("native_distributed"));
    assert_eq!(pc.native_distributed_supported, Some(true));
    assert_eq!(pc.adapter_distributed_supported, Some(true));
}

#[test]
fn mpi_strict_mode_rejects_adapter_masking() {
    let _guard = mpi_test_guard();
    let comm = UniverseComm::Mpi(Arc::new(MpiComm::new()));
    comm.set_reproducible(true);
    if comm.size() <= 1 {
        return;
    }

    let dist = Arc::new(make_dist_poisson(&comm, 4));
    let mut ksp = KspContext::new();
    ksp.set_type(SolverType::Gmres).expect("set gmres");
    ksp.set_operators(dist, None);

    let ksp_opts = KspOptions {
        ksp_type: Some("gmres".to_string()),
        maxits: Some(20),
        ..Default::default()
    };
    let pc_opts = PcOptions {
        pc_type: Some("ilu".to_string()),
        pc_global: Some("block_jacobi".to_string()),
        pc_local: Some("ilu".to_string()),
        pc_dist_local_apply: Some("strict".to_string()),
        pc_dist_route: Some("adapted".to_string()),
        ..Default::default()
    };
    ksp.set_from_all_options(&ksp_opts, &pc_opts)
        .expect("set distributed options");
    let err = ksp
        .setup()
        .expect_err("strict mode must reject adapted fallback route");
    assert!(
        err.to_string()
            .contains("pc_dist_local_apply=strict forbids local adapter fallback"),
        "unexpected strict rejection error: {err}"
    );
}
