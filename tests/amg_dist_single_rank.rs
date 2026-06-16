#![cfg(all(feature = "backend-faer", not(feature = "complex")))]

use faer::Mat;
use kryst::matrix::{CsrMatrix, DistCsrOp};
use kryst::parallel::{NoComm, UniverseComm};
use kryst::preconditioner::amg::{AMGBuilder, RelaxType};
use kryst::preconditioner::dist::DistCoarseStrategy;
use kryst::preconditioner::{PcDistributedSupport, PcSide, Preconditioner};

fn poisson_1d(n: usize) -> CsrMatrix<f64> {
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::new();
    let mut vals = Vec::new();
    row_ptr.push(0);
    for i in 0..n {
        if i > 0 {
            col_idx.push(i - 1);
            vals.push(-1.0);
        }
        col_idx.push(i);
        vals.push(2.0);
        if i + 1 < n {
            col_idx.push(i + 1);
            vals.push(-1.0);
        }
        row_ptr.push(col_idx.len());
    }
    CsrMatrix::from_csr(n, n, row_ptr, col_idx, vals)
}

fn build_amg() -> kryst::preconditioner::amg::AMG {
    AMGBuilder::new()
        .logging_level(0)
        .relaxation_type(RelaxType::Jacobi)
        .grid_relax_type_all(RelaxType::Jacobi)
        .build(&Mat::<f64>::zeros(0, 0))
        .expect("amg build")
}

#[test]
fn single_rank_distcsr_amg_matches_local_amg_apply() {
    let n = 12;
    let local_csr = poisson_1d(n);
    let comm = UniverseComm::NoComm(NoComm);
    let dist =
        DistCsrOp::from_local_rows(n, 0, &local_csr, &[0, n], comm).expect("single-rank DistCsrOp");

    let rhs: Vec<f64> = (0..n)
        .map(|i| ((i + 1) as f64).sin() + 0.25 * ((2 * i + 1) as f64).cos())
        .collect();

    let mut local_amg = build_amg();
    local_amg.setup(&local_csr).expect("local AMG setup");
    let mut local_out = vec![0.0; n];
    local_amg
        .apply(PcSide::Left, &rhs, &mut local_out)
        .expect("local AMG apply");

    let mut dist_amg = build_amg();
    dist_amg
        .setup(&dist)
        .expect("single-rank DistCsrOp AMG setup");
    assert_eq!(
        dist_amg.distributed_support(),
        PcDistributedSupport::LocalOnly
    );
    let mut dist_out = vec![0.0; n];
    dist_amg
        .apply(PcSide::Left, &rhs, &mut dist_out)
        .expect("single-rank DistCsrOp AMG apply");

    for (i, (local, dist)) in local_out.iter().zip(dist_out.iter()).enumerate() {
        let diff = (local - dist).abs();
        let scale = local.abs().max(dist.abs()).max(1.0);
        assert!(
            diff <= 1e-11 * scale,
            "AMG output mismatch at row {i}: local={local}, dist={dist}, diff={diff}"
        );
    }
}

#[test]
fn single_rank_distcsr_amg_numeric_update_matches_local_amg() {
    let n = 12;
    let local_csr = poisson_1d(n);
    let mut updated_vals = local_csr.values().to_vec();
    for row in 0..n {
        for slot in local_csr.row_ptr()[row]..local_csr.row_ptr()[row + 1] {
            if local_csr.col_idx()[slot] == row {
                updated_vals[slot] = 3.0;
            }
        }
    }
    let updated_csr = CsrMatrix::from_csr(
        n,
        n,
        local_csr.row_ptr().to_vec(),
        local_csr.col_idx().to_vec(),
        updated_vals.clone(),
    );

    let comm = UniverseComm::NoComm(NoComm);
    let mut dist =
        DistCsrOp::from_local_rows(n, 0, &local_csr, &[0, n], comm).expect("single-rank DistCsrOp");

    let rhs: Vec<f64> = (0..n)
        .map(|i| 1.0 + 0.125 * ((3 * i + 2) as f64).sin())
        .collect();

    let mut local_amg = build_amg();
    local_amg.setup(&local_csr).expect("local AMG setup");
    let mut before_update = vec![0.0; n];
    local_amg
        .apply(PcSide::Left, &rhs, &mut before_update)
        .expect("local AMG apply before update");
    local_amg
        .update_numeric(&updated_csr)
        .expect("local AMG numeric update");
    let mut local_out = vec![0.0; n];
    local_amg
        .apply(PcSide::Left, &rhs, &mut local_out)
        .expect("local AMG apply after update");

    dist.update_numeric(&updated_vals)
        .expect("DistCsrOp numeric update");
    let mut dist_amg = build_amg();
    dist_amg
        .setup(&local_csr)
        .expect("initial AMG setup before DistCsrOp update");
    dist_amg
        .update_numeric(&dist)
        .expect("single-rank DistCsrOp AMG numeric update");
    let mut dist_out = vec![0.0; n];
    dist_amg
        .apply(PcSide::Left, &rhs, &mut dist_out)
        .expect("single-rank DistCsrOp AMG apply after update");

    let changed = before_update
        .iter()
        .zip(local_out.iter())
        .any(|(before, after)| (before - after).abs() > 1e-9);
    assert!(changed, "numeric update did not change AMG output");

    for (i, (local, dist)) in local_out.iter().zip(dist_out.iter()).enumerate() {
        let diff = (local - dist).abs();
        let scale = local.abs().max(dist.abs()).max(1.0);
        assert!(
            diff <= 1e-11 * scale,
            "updated AMG output mismatch at row {i}: local={local}, dist={dist}, diff={diff}"
        );
    }
}

#[test]
fn single_rank_distributed_csr_route_matches_residual_correction_contract() {
    let n = 12;
    let local_csr = poisson_1d(n);
    let comm = UniverseComm::NoComm(NoComm);
    let dist =
        DistCsrOp::from_local_rows(n, 0, &local_csr, &[0, n], comm).expect("single-rank DistCsrOp");
    let rhs: Vec<f64> = (0..n).map(|i| 0.5 + ((i + 2) as f64).sin()).collect();

    let mut local_amg = build_amg();
    local_amg.setup(&local_csr).expect("local AMG setup");
    let mut expected = vec![0.0; n];
    local_amg
        .apply(PcSide::Left, &rhs, &mut expected)
        .expect("first local AMG correction");
    let mut ax = vec![0.0; n];
    kryst::core::traits::MatVec::matvec(&local_csr, &expected, &mut ax);
    let residual: Vec<f64> = rhs.iter().zip(&ax).map(|(r, az)| r - az).collect();
    let mut correction = vec![0.0; n];
    local_amg
        .apply(PcSide::Left, &residual, &mut correction)
        .expect("second local AMG correction");
    for (value, correction) in expected.iter_mut().zip(correction) {
        *value += correction;
    }

    let mut dist_amg = AMGBuilder::new()
        .logging_level(0)
        .relaxation_type(RelaxType::Jacobi)
        .grid_relax_type_all(RelaxType::Jacobi)
        .dist_coarse_strategy(DistCoarseStrategy::DistributedCsr)
        .dist_apply_instrumentation(true)
        .build(&Mat::<f64>::zeros(0, 0))
        .expect("distributed CSR AMG build");
    dist_amg.setup(&dist).expect("distributed CSR AMG setup");
    assert_eq!(
        dist_amg.distributed_support(),
        PcDistributedSupport::LocalOnly
    );

    let mut actual = vec![0.0; n];
    dist_amg
        .apply(PcSide::Left, &rhs, &mut actual)
        .expect("distributed CSR AMG apply");

    for (i, (expected, actual)) in expected.iter().zip(actual.iter()).enumerate() {
        let scale = expected.abs().max(actual.abs()).max(1.0);
        assert!(
            (expected - actual).abs() <= 1e-11 * scale,
            "distributed CSR correction mismatch at row {i}: expected={expected}, actual={actual}"
        );
    }
    let stats = dist_amg
        .dist_apply_stats()
        .expect("distributed apply stats");
    assert_eq!(stats.mode_label(), "distributed_csr");
    assert_eq!(stats.coarse_solver_route_label(), "distributed_csr");
    assert!(!stats.reports_distributed_support());
    assert!(!stats.uses_root_gather());
    assert_eq!(stats.gather, std::time::Duration::default());
    assert_eq!(stats.scatter, std::time::Duration::default());
}

#[test]
fn single_rank_distributed_csr_route_refreshes_numeric_values() {
    let n = 12;
    let local_csr = poisson_1d(n);
    let mut updated_vals = local_csr.values().to_vec();
    for row in 0..n {
        for slot in local_csr.row_ptr()[row]..local_csr.row_ptr()[row + 1] {
            if local_csr.col_idx()[slot] == row {
                updated_vals[slot] = 3.5;
            }
        }
    }
    let updated_csr = CsrMatrix::from_csr(
        n,
        n,
        local_csr.row_ptr().to_vec(),
        local_csr.col_idx().to_vec(),
        updated_vals.clone(),
    );
    let comm = UniverseComm::NoComm(NoComm);
    let mut dist =
        DistCsrOp::from_local_rows(n, 0, &local_csr, &[0, n], comm).expect("single-rank DistCsrOp");
    let rhs: Vec<f64> = (0..n).map(|i| 0.75 + ((2 * i + 3) as f64).cos()).collect();

    let mut local_amg = build_amg();
    local_amg
        .setup(&updated_csr)
        .expect("updated local AMG setup");
    let mut expected = vec![0.0; n];
    local_amg
        .apply(PcSide::Left, &rhs, &mut expected)
        .expect("first updated local AMG correction");
    let mut ax = vec![0.0; n];
    kryst::core::traits::MatVec::matvec(&updated_csr, &expected, &mut ax);
    let residual: Vec<f64> = rhs.iter().zip(&ax).map(|(r, az)| r - az).collect();
    let mut correction = vec![0.0; n];
    local_amg
        .apply(PcSide::Left, &residual, &mut correction)
        .expect("second updated local AMG correction");
    for (value, correction) in expected.iter_mut().zip(correction) {
        *value += correction;
    }

    let mut dist_amg = AMGBuilder::new()
        .logging_level(0)
        .relaxation_type(RelaxType::Jacobi)
        .grid_relax_type_all(RelaxType::Jacobi)
        .dist_coarse_strategy(DistCoarseStrategy::DistributedCsr)
        .build(&Mat::<f64>::zeros(0, 0))
        .expect("distributed CSR AMG build");
    dist_amg.setup(&dist).expect("distributed CSR AMG setup");
    dist.update_numeric(&updated_vals)
        .expect("DistCsrOp numeric update");
    dist_amg
        .update_numeric(&dist)
        .expect("distributed CSR AMG numeric refresh");

    let mut actual = vec![0.0; n];
    dist_amg
        .apply(PcSide::Left, &rhs, &mut actual)
        .expect("updated distributed CSR AMG apply");
    for (i, (expected, actual)) in expected.iter().zip(actual.iter()).enumerate() {
        let scale = expected.abs().max(actual.abs()).max(1.0);
        assert!(
            (expected - actual).abs() <= 1e-11 * scale,
            "updated distributed CSR correction mismatch at row {i}: expected={expected}, actual={actual}"
        );
    }
}
