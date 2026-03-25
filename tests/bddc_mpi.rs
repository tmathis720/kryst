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
    let comm = UniverseComm::Mpi(Arc::new(MpiComm::new()));
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
    let comm = UniverseComm::Mpi(Arc::new(MpiComm::new()));
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
