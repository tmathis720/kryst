#![cfg(all(feature = "backend-faer", feature = "mpi", not(feature = "complex")))]

mod fixtures;

use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

use kryst::config::options::{KspOptions, PcOptions};
use kryst::context::ksp_context::{KspContext, SolverType};
use kryst::matrix::dist_csr::DistCsrOp;
use kryst::matrix::sparse::CsrMatrix;
use kryst::parallel::{Comm, MpiComm, UniverseComm};
use kryst::utils::convergence::ConvergedReason;


fn mpi_test_guard() -> MutexGuard<'static, ()> {
    static GUARD: OnceLock<Mutex<()>> = OnceLock::new();
    GUARD
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn mpi_world() -> Option<UniverseComm> {
    let Some(comm) = MpiComm::try_new() else {
        eprintln!("skipping bddc mpi tests: MPI init failed");
        return None;
    };
    if comm.size() < 2 {
        eprintln!(
            "skipping bddc mpi tests: requires >=2 MPI ranks, found {}",
            comm.size()
        );
        return None;
    }
    Some(UniverseComm::Mpi(Arc::new(comm)))
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

fn solve_with_pc(
    dist: Arc<DistCsrOp>,
    rhs: &[f64],
    pc_opts: &PcOptions,
) -> (usize, f64, ConvergedReason) {
    let mut ksp = KspContext::new();
    let ksp_opts = KspOptions {
        ksp_type: Some("gmres".to_string()),
        rtol: Some(1e-10),
        atol: Some(1e-12),
        maxits: Some(500),
        ..Default::default()
    };
    ksp.set_type(SolverType::Gmres).expect("set gmres");
    ksp.set_operators(dist, None);
    ksp.set_from_all_options(&ksp_opts, pc_opts)
        .expect("set options");
    ksp.setup().expect("ksp setup");

    let mut x = vec![0.0; rhs.len()];
    let stats = ksp.solve(rhs, &mut x).expect("solve");
    (stats.iterations, stats.final_residual, stats.reason)
}

#[test]
fn mpi_bddc_converges_and_is_stable_vs_block_jacobi() {
    let _guard = mpi_test_guard();
    let Some(comm) = mpi_world() else {
        return;
    };
    comm.set_reproducible(true);

    let n_per = 8;
    let dist = Arc::new(make_dist_poisson(&comm, n_per));
    let rhs: Vec<f64> = (0..n_per).map(|i| 1.0 + i as f64).collect();
    let rhs_norm2_local: f64 = rhs.iter().map(|v| v * v).sum();
    let rhs_norm = comm.all_reduce_f64(rhs_norm2_local).sqrt();

    let bj_opts = PcOptions {
        pc_type: Some("jacobi".to_string()),
        pc_global: Some("block_jacobi".to_string()),
        pc_local: Some("ilu".to_string()),
        ..Default::default()
    };
    let (bj_its, _bj_res, bj_reason) = solve_with_pc(dist.clone(), &rhs, &bj_opts);

    let bddc_opts = PcOptions {
        pc_type: Some("bddc".to_string()),
        pc_bddc_coarse_ksp_type: Some("preonly".to_string()),
        pc_bddc_coarse_pc_type: Some("lu".to_string()),
        pc_bddc_constraint_selection: Some("interface".to_string()),
        pc_bddc_scaling: Some("uniform".to_string()),
        ..Default::default()
    };
    let (bddc_its, bddc_res, bddc_reason) = solve_with_pc(dist, &rhs, &bddc_opts);

    assert!(
        matches!(
            bj_reason,
            ConvergedReason::ConvergedRtol | ConvergedReason::ConvergedAtol
        ),
        "block-jacobi baseline did not converge: {bj_reason:?}"
    );
    assert!(
        matches!(
            bddc_reason,
            ConvergedReason::ConvergedRtol | ConvergedReason::ConvergedAtol
        ),
        "bddc did not converge: {bddc_reason:?}"
    );
    assert!(
        bddc_res < rhs_norm,
        "BDDC should reduce residual from RHS norm"
    );
    assert!(
        bddc_its <= bj_its.saturating_mul(2).max(8),
        "BDDC iterations ({bddc_its}) should remain stable against block-jacobi ({bj_its})"
    );
}

#[test]
fn mpi_mg_is_viable_distributed_baseline_for_bddc_matrix() {
    let _guard = mpi_test_guard();
    let Some(comm) = mpi_world() else {
        return;
    };
    comm.set_reproducible(true);

    let n_per = 6;
    let dist = Arc::new(make_dist_poisson(&comm, n_per));
    let rhs: Vec<f64> = (0..n_per).map(|i| 1.0 + i as f64).collect();

    let mg_opts = PcOptions {
        pc_type: Some("mg".to_string()),
        pc_mg_levels: Some(3),
        pc_mg_cycle_type: Some("v".to_string()),
        pc_mg_smoother: Some("jacobi".to_string()),
        pc_mg_smoother_steps: Some(1),
        ..Default::default()
    };
    let (_its, res, reason) = solve_with_pc(dist, &rhs, &mg_opts);
    assert!(
        matches!(
            reason,
            ConvergedReason::ConvergedRtol | ConvergedReason::ConvergedAtol
        ),
        "mg should converge on distributed poisson system: {reason:?}"
    );
    assert!(res.is_finite(), "mg residual must be finite");
}

#[test]
fn mpi_bddc_constraint_and_scaling_variants_converge() {
    let _guard = mpi_test_guard();
    let Some(comm) = mpi_world() else {
        return;
    };
    comm.set_reproducible(true);
    let n_per = 5;
    let dist = Arc::new(make_dist_poisson(&comm, n_per));
    let rhs: Vec<f64> = (0..n_per).map(|i| 0.5 + i as f64).collect();

    let opts_vertices = PcOptions {
        pc_type: Some("bddc".to_string()),
        pc_bddc_constraint_selection: Some("vertices".to_string()),
        pc_bddc_scaling: Some("uniform".to_string()),
        ..Default::default()
    };
    let (_its_v, res_v, reason_v) = solve_with_pc(dist.clone(), &rhs, &opts_vertices);
    assert!(
        matches!(
            reason_v,
            ConvergedReason::ConvergedRtol | ConvergedReason::ConvergedAtol
        ),
        "vertices mode should converge: {reason_v:?}"
    );

    let opts_iface_deluxe = PcOptions {
        pc_type: Some("bddc".to_string()),
        pc_bddc_constraint_selection: Some("interface".to_string()),
        pc_bddc_scaling: Some("deluxe_like".to_string()),
        ..Default::default()
    };
    let (_its_i, res_i, reason_i) = solve_with_pc(dist, &rhs, &opts_iface_deluxe);
    assert!(
        matches!(
            reason_i,
            ConvergedReason::ConvergedRtol | ConvergedReason::ConvergedAtol
        ),
        "interface+deluxe mode should converge: {reason_i:?}"
    );
    assert!(res_v.is_finite() && res_i.is_finite());
}

#[test]
fn mpi_bddc_coarse_backend_combinations_converge() {
    let _guard = mpi_test_guard();
    let Some(comm) = mpi_world() else {
        return;
    };
    comm.set_reproducible(true);
    let n_per = 6;
    let dist = Arc::new(make_dist_poisson(&comm, n_per));
    let rhs: Vec<f64> = (0..n_per).map(|i| 1.0 + 0.25 * i as f64).collect();
    let combos = [("preonly", "lu"), ("cg", "jacobi"), ("gmres", "ilu")];

    for (ksp, pc) in combos {
        let opts = PcOptions {
            pc_type: Some("bddc".to_string()),
            pc_bddc_coarse_ksp_type: Some(ksp.to_string()),
            pc_bddc_coarse_pc_type: Some(pc.to_string()),
            pc_bddc_constraint_selection: Some("interface".to_string()),
            ..Default::default()
        };
        let (_its, res, reason) = solve_with_pc(dist.clone(), &rhs, &opts);
        assert!(
            matches!(
                reason,
                ConvergedReason::ConvergedRtol | ConvergedReason::ConvergedAtol
            ),
            "coarse combo {ksp}+{pc} should converge, got {reason:?}"
        );
        assert!(
            res.is_finite(),
            "coarse combo {ksp}+{pc} residual must be finite"
        );
    }
}
