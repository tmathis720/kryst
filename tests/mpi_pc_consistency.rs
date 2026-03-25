#![cfg(not(feature = "complex"))]
#![cfg(feature = "mpi")]

mod fixtures;

use std::sync::Arc;

use kryst::config::options::{KspOptions, PcOptions};
use kryst::context::ksp_context::{KspContext, SolverType};
use kryst::matrix::dist_csr::DistCsrOp;
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
        ConvergedReason::DivergedBreakdownBiCG => 10.0,
        ConvergedReason::DivergedIndefiniteMatrix => 11.0,
        ConvergedReason::DivergedIndefinitePC => 12.0,
        ConvergedReason::DivergedPcSetupFailed => 13.0,
        ConvergedReason::DivergedPcFailed => 14.0,
        ConvergedReason::StoppedByMonitor => 15.0,
        ConvergedReason::Continued => 16.0,
    }
}

#[test]
fn mpi_convergence_reason_consistent_across_ranks() {
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
        opts.pc_mg_coarse_solver_route.as_deref(),
        Some("root_gather")
    );
    assert!(opts.pc_mg_level_policies.is_some());
}
