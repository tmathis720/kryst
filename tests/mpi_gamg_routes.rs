#![cfg(all(feature = "backend-faer", feature = "mpi", not(feature = "complex")))]

mod fixtures;

use std::sync::Arc;

use kryst::config::options::{KspOptions, PcOptions};
use kryst::context::ksp_context::{KspContext, SolverType};
use kryst::matrix::dist_csr::DistCsrOp;
use kryst::matrix::sparse::CsrMatrix;
use kryst::parallel::{Comm, MpiComm, UniverseComm};

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

fn make_dist_poisson(comm: &UniverseComm, n_per: usize) -> Arc<DistCsrOp> {
    let rank = comm.rank();
    let size = comm.size();
    let n_global = n_per * size;
    let row_start = rank * n_per;
    let global = fixtures::csr_poisson_1d(n_global);
    let local = local_rows_from_global(&global, row_start, n_per);
    let part_prefix: Vec<usize> = (0..=size).map(|p| p * n_per).collect();
    Arc::new(
        DistCsrOp::from_local_rows(n_global, row_start, &local, &part_prefix, comm.clone())
            .expect("dist csr"),
    )
}

#[test]
fn mpi_gamg_auto_route_policy_prefers_fallback_chain() {
    let comm = UniverseComm::Mpi(Arc::new(MpiComm::new()));
    if comm.size() <= 1 {
        return;
    }

    let dist = make_dist_poisson(&comm, 6);
    let rhs = vec![1.0; dist.local_nrows()];

    let mut ksp = KspContext::new();
    ksp.set_type(SolverType::Gmres).expect("set solver");
    ksp.set_operators(dist, None);
    ksp.set_from_all_options(
        &KspOptions {
            ksp_type: Some("gmres".into()),
            rtol: Some(1e-9),
            maxits: Some(80),
            ..Default::default()
        },
        &PcOptions {
            pc_type: Some("gamg".into()),
            amg_dist_apply_mode: Some("root".into()),
            amg_dist_coarse_solver_route: Some("auto,superlu_dist,root,local".into()),
            pc_gamg_level_policies: Some(vec![
                "level=2,coarse_routes=auto,superlu_dist,root,local".into(),
            ]),
            ..Default::default()
        },
    )
    .expect("set opts");
    ksp.setup().expect("gamg setup");
    let mut x = vec![0.0; rhs.len()];
    ksp.solve(&rhs, &mut x).expect("solve");
}

#[test]
fn mpi_gamg_forced_unavailable_route_fails_without_fallback() {
    let comm = UniverseComm::Mpi(Arc::new(MpiComm::new()));
    if comm.size() <= 1 {
        return;
    }

    let dist = make_dist_poisson(&comm, 6);

    let mut ksp = KspContext::new();
    ksp.set_type(SolverType::Gmres).expect("set solver");
    ksp.set_operators(dist, None);
    ksp.set_from_all_options(
        &KspOptions {
            ksp_type: Some("gmres".into()),
            maxits: Some(20),
            ..Default::default()
        },
        &PcOptions {
            pc_type: Some("gamg".into()),
            amg_dist_coarse_solver_route: Some("root,local".into()),
            pc_gamg_level_policies: Some(vec!["level=2,coarse_routes=superlu_dist,root".into()]),
            ..Default::default()
        },
    )
    .expect("set opts");

    #[cfg(not(feature = "superlu_dist"))]
    {
        let err = ksp
            .setup()
            .expect_err("forced unavailable route should fail");
        assert!(
            err.to_string().contains("forced distributed coarse route")
                || err.to_string().contains("explicitly requested")
        );
    }

    #[cfg(feature = "superlu_dist")]
    {
        ksp.setup().expect("superlu_dist build should allow route");
    }
}
