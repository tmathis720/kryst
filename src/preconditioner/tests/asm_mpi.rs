#![cfg(feature = "mpi")]
#![cfg(not(feature = "complex"))]

use super::*;
use crate::algebra::prelude::*;
use crate::assert_vec_close;
use crate::matrix::DistCsrOp;
use crate::matrix::op::{CsrOp, LinOp};
use crate::matrix::sparse::CsrMatrix;
use crate::parallel::{Comm, MpiComm, UniverseComm};
use crate::preconditioner::Preconditioner;
use crate::preconditioner::asm::{AsmBlockSolver, AsmInnerPc, AsmPc, Weighting};
use crate::preconditioner::builders::{build_block_jacobi, build_ilu0_with_conditioning};
use crate::preconditioner::dist::{
    DistCoarseStrategy, DistLocalApplyMode, DistPcAdapter, DistPcBuilder, GlobalPcKind,
    LocalPcKind, MpiPcOptions,
};
use crate::utils::conditioning::ConditioningOptions;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::time::Instant;

fn mpi_test_guard() -> MutexGuard<'static, ()> {
    static GUARD: OnceLock<Mutex<()>> = OnceLock::new();
    GUARD
        .get_or_init(|| Mutex::new(()))
        .lock()
        .expect("mpi_test_guard poisoned")
}

fn mpi_world() -> Option<UniverseComm> {
    let Some(comm) = MpiComm::try_new() else {
        eprintln!("skipping asm mpi tests: MPI init failed");
        return None;
    };
    let comm = UniverseComm::Mpi(Arc::new(comm));
    if !(2..=4).contains(&comm.size()) {
        eprintln!(
            "skipping asm mpi tests: expected 2-4 ranks, got {}",
            comm.size()
        );
        return None;
    }
    Some(comm)
}

fn local_rows_from_global(global: &CsrMatrix<R>, row_start: usize, n_local: usize) -> CsrMatrix<R> {
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

fn make_dist_poisson(comm: &UniverseComm, n_per: usize) -> (DistCsrOp, CsrMatrix<R>, usize, usize) {
    let rank = comm.rank();
    let size = comm.size();
    let n_global = n_per * size;
    let row_start = rank * n_per;
    let global = super::asm_amg::poisson_1d(n_global);
    let local = local_rows_from_global(&global, row_start, n_per);
    let part_prefix: Vec<usize> = (0..=size).map(|p| p * n_per).collect();
    let dist = DistCsrOp::from_local_rows(n_global, row_start, &local, &part_prefix, comm.clone())
        .expect("dist csr");
    (dist, global, row_start, n_global)
}

fn make_dist_block_diag(comm: &UniverseComm, n_per: usize) -> DistCsrOp {
    let rank = comm.rank();
    let size = comm.size();
    let n_global = n_per * size;
    let row_start = rank * n_per;
    let mut row_ptr = Vec::with_capacity(n_per + 1);
    let mut col_idx = Vec::new();
    let mut vals = Vec::new();
    row_ptr.push(0);
    for i in 0..n_per {
        let gi = row_start + i;
        if i > 0 {
            col_idx.push(gi - 1);
            vals.push(-1.0);
        }
        col_idx.push(gi);
        vals.push(4.0);
        if i + 1 < n_per {
            col_idx.push(gi + 1);
            vals.push(-1.0);
        }
        row_ptr.push(col_idx.len());
    }
    let local = CsrMatrix::from_csr(n_per, n_global, row_ptr, col_idx, vals);
    let part_prefix: Vec<usize> = (0..=size).map(|p| p * n_per).collect();
    DistCsrOp::from_local_rows(n_global, row_start, &local, &part_prefix, comm.clone())
        .expect("dist block-diag")
}

fn subdomain_from_global(global: &CsrMatrix<R>, subdofs: &[usize]) -> CsrMatrix<R> {
    let mut map = HashMap::with_capacity(subdofs.len());
    for (i, &g) in subdofs.iter().enumerate() {
        map.insert(g, i);
    }
    let mut row_ptr = Vec::with_capacity(subdofs.len() + 1);
    let mut col_idx = Vec::new();
    let mut values = Vec::new();
    row_ptr.push(0);
    for &g in subdofs {
        let (cols, vals) = global.row(g);
        for (&col, &val) in cols.iter().zip(vals.iter()) {
            if let Some(&local_col) = map.get(&col) {
                col_idx.push(local_col);
                values.push(val);
            }
        }
        row_ptr.push(col_idx.len());
    }
    CsrMatrix::from_csr(subdofs.len(), subdofs.len(), row_ptr, col_idx, values)
}

fn distributed_l2_norm(comm: &UniverseComm, v: &[f64]) -> f64 {
    let local_sq: f64 = v.iter().map(|x| x * x).sum();
    comm.all_reduce_f64(local_sq).sqrt()
}

fn stationary_iterations(
    dist: &DistCsrOp,
    pc: &dyn Preconditioner,
    rhs: &[f64],
    iters: usize,
) -> f64 {
    let mut x = vec![0.0; rhs.len()];
    let mut z = vec![0.0; rhs.len()];
    let mut ax = vec![0.0; rhs.len()];
    let mut r = vec![0.0; rhs.len()];
    let comm = dist.comm();
    for _ in 0..iters {
        dist.matvec(&x, &mut ax);
        for i in 0..rhs.len() {
            r[i] = rhs[i] - ax[i];
        }
        pc.apply(PcSide::Left, &r, &mut z).expect("pc apply");
        for i in 0..x.len() {
            x[i] += z[i];
        }
    }
    dist.matvec(&x, &mut ax);
    for i in 0..rhs.len() {
        r[i] = rhs[i] - ax[i];
    }
    distributed_l2_norm(&comm, &r)
}

#[test]
fn mpi_ras_overlap_zero_matches_block_jacobi() {
    let _guard = mpi_test_guard();
    let Some(comm) = mpi_world() else {
        return;
    };
    let n_per = 2;
    let (dist, _global, _row_start, _n_global) = make_dist_poisson(&comm, n_per);

    let mut asm = AsmPc::ras(
        0,
        None,
        AsmBlockSolver::Csr,
        AsmInnerPc::Ilu0,
        Weighting::None,
    );
    asm.setup(&dist).expect("ras asm setup");

    let mut bj = build_block_jacobi(n_per).expect("block jacobi build");
    bj.setup(&dist).expect("block jacobi setup");

    let rhs: Vec<S> = (0..n_per).map(|i| S::from_real((i + 1) as f64)).collect();
    let mut y_asm = vec![S::zero(); n_per];
    let mut y_bj = vec![S::zero(); n_per];
    asm.apply(PcSide::Left, &rhs, &mut y_asm)
        .expect("ras asm apply");
    bj.apply(PcSide::Left, &rhs, &mut y_bj)
        .expect("block jacobi apply");

    assert_vec_close!("ras overlap=0 matches block jacobi", &y_asm, &y_bj);
}

#[test]
fn mpi_ras_overlap_imports_ghost_rows() {
    let _guard = mpi_test_guard();
    let Some(comm) = mpi_world() else {
        return;
    };
    let n_per = 2;
    let (dist, _global, row_start, _n_global) = make_dist_poisson(&comm, n_per);

    let mut asm0 = AsmPc::ras(
        0,
        None,
        AsmBlockSolver::Csr,
        AsmInnerPc::Ilu0,
        Weighting::None,
    );
    asm0.setup(&dist).expect("ras asm overlap=0 setup");

    let mut asm1 = AsmPc::ras(
        1,
        None,
        AsmBlockSolver::Csr,
        AsmInnerPc::Ilu0,
        Weighting::None,
    );
    asm1.setup(&dist).expect("ras asm overlap=1 setup");

    let mut rhs = vec![S::zero(); n_per];
    if comm.rank() == 1 {
        rhs[0] = S::from_real(1.0);
    }

    let mut y0 = vec![S::zero(); n_per];
    let mut y1 = vec![S::zero(); n_per];
    asm0.apply(PcSide::Left, &rhs, &mut y0)
        .expect("ras asm overlap=0 apply");
    asm1.apply(PcSide::Left, &rhs, &mut y1)
        .expect("ras asm overlap=1 apply");

    if row_start == 0 {
        let near_zero = y0.iter().all(|v| v.abs() < 1e-12);
        assert!(near_zero, "overlap=0 should ignore ghost rhs");
        assert!(
            y1[n_per - 1].abs() > 1e-8,
            "overlap=1 should import ghost rhs"
        );
    }
}

#[test]
fn mpi_ras_apply_injects_owned_rows() {
    let _guard = mpi_test_guard();
    let Some(comm) = mpi_world() else {
        return;
    };
    let n_per = 2;
    let (dist, global, row_start, n_global) = make_dist_poisson(&comm, n_per);
    let row_end = row_start + n_per;

    let mut asm = AsmPc::ras(
        1,
        None,
        AsmBlockSolver::Csr,
        AsmInnerPc::Ilu0,
        Weighting::None,
    );
    asm.setup(&dist).expect("ras asm setup");

    let x_global: Vec<S> = (0..n_global)
        .map(|i| S::from_real((i + 1) as f64))
        .collect();
    let rhs = x_global[row_start..row_end].to_vec();

    let mut y_local = vec![S::zero(); n_per];
    asm.apply(PcSide::Left, &rhs, &mut y_local)
        .expect("ras asm apply");

    let sub_start = row_start.saturating_sub(1);
    let sub_end = (row_end + 1).min(n_global);
    let subdofs: Vec<usize> = (sub_start..sub_end).collect();
    let sub_csr = subdomain_from_global(&global, &subdofs);

    let mut ilu =
        build_ilu0_with_conditioning(ConditioningOptions::default()).expect("ilu0 builder");
    ilu.setup(&CsrOp::new(Arc::new(sub_csr)))
        .expect("subdomain ilu0 setup");

    let rhs_sub: Vec<S> = subdofs.iter().map(|&g| x_global[g]).collect();
    let mut sol_sub = vec![S::zero(); subdofs.len()];
    ilu.apply(PcSide::Left, &rhs_sub, &mut sol_sub)
        .expect("subdomain ilu0 apply");

    let owned_offset = row_start - sub_start;
    let expected = &sol_sub[owned_offset..owned_offset + n_per];
    assert_vec_close!("ras injects owned rows", &y_local, expected);
}

#[test]
fn mpi_block_jacobi_strict_rejects_unsupported_local_pc() {
    let _guard = mpi_test_guard();
    let Some(comm) = mpi_world() else {
        return;
    };
    let (dist, _, _, _) = make_dist_poisson(&comm, 2);

    let opts = MpiPcOptions {
        global_pc: GlobalPcKind::BlockJacobi,
        local_pc: LocalPcKind::Fsai,
        local_apply_mode: DistLocalApplyMode::NativeStrict,
        ..MpiPcOptions::default()
    };
    let err = match DistPcAdapter::build(&dist, DistPcBuilder::BlockJacobi { opts }) {
        Ok(_) => panic!("strict mode should reject unsupported local pc"),
        Err(err) => err,
    };
    let msg = format!("{err}");
    assert!(
        msg.contains("err_key=pc_dist_strict_mode_rejected")
            && msg.contains("pc_dist_local_apply=strict")
            && msg.contains("pc_global=BlockJacobi")
            && msg.contains("detail_key=unsupported_local_pc"),
        "unexpected strict-mode error: {msg}"
    );
}

#[test]
fn mpi_block_jacobi_hybrid_differs_from_halo_only() {
    let _guard = mpi_test_guard();
    let Some(comm) = mpi_world() else {
        return;
    };
    let (dist, _, row_start, _) = make_dist_poisson(&comm, 2);

    let halo_opts = MpiPcOptions {
        global_pc: GlobalPcKind::BlockJacobi,
        local_pc: LocalPcKind::Chebyshev,
        local_apply_mode: DistLocalApplyMode::NativeLocalHalo,
        ..MpiPcOptions::default()
    };
    let hybrid_opts = MpiPcOptions {
        local_apply_mode: DistLocalApplyMode::NativeHybrid,
        ..halo_opts.clone()
    };

    let halo = DistPcAdapter::build(&dist, DistPcBuilder::BlockJacobi { opts: halo_opts })
        .expect("halo-only build");
    let hybrid = DistPcAdapter::build(&dist, DistPcBuilder::BlockJacobi { opts: hybrid_opts })
        .expect("hybrid build");

    let mut rhs = vec![0.0; dist.local_nrows()];
    if row_start == 0 {
        rhs[0] = 1.0;
    }
    let mut y_halo = vec![0.0; rhs.len()];
    let mut y_hybrid = vec![0.0; rhs.len()];
    halo.apply(PcSide::Left, &rhs, &mut y_halo)
        .expect("halo apply");
    hybrid
        .apply(PcSide::Left, &rhs, &mut y_hybrid)
        .expect("hybrid apply");

    let l1_local: f64 = y_halo
        .iter()
        .zip(y_hybrid.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    let l1_global = comm.all_reduce_f64(l1_local);
    assert!(
        l1_global > 1e-12,
        "expected hybrid strategy to alter distributed correction"
    );
}

#[test]
fn mpi_block_jacobi_sor_native_matches_wrapped_on_block_diag() {
    let _guard = mpi_test_guard();
    let Some(comm) = mpi_world() else {
        return;
    };
    let dist = make_dist_block_diag(&comm, 3);

    let wrapped_opts = MpiPcOptions {
        global_pc: GlobalPcKind::BlockJacobi,
        local_pc: LocalPcKind::Sor,
        local_apply_mode: DistLocalApplyMode::WrappedLocal,
        ..MpiPcOptions::default()
    };
    let native_opts = MpiPcOptions {
        local_apply_mode: DistLocalApplyMode::NativeLocalHalo,
        ..wrapped_opts.clone()
    };

    let wrapped = DistPcAdapter::build(&dist, DistPcBuilder::BlockJacobi { opts: wrapped_opts })
        .expect("wrapped sor build");
    let native = DistPcAdapter::build(&dist, DistPcBuilder::BlockJacobi { opts: native_opts })
        .expect("native sor build");

    let rhs: Vec<f64> = (0..dist.local_nrows()).map(|i| (i + 1) as f64).collect();
    let mut y_wrapped = vec![0.0; rhs.len()];
    let mut y_native = vec![0.0; rhs.len()];
    wrapped
        .apply(PcSide::Left, &rhs, &mut y_wrapped)
        .expect("wrapped sor apply");
    native
        .apply(PcSide::Left, &rhs, &mut y_native)
        .expect("native sor apply");

    assert_vec_close!(
        "sor native equals wrapped on uncoupled blocks",
        &y_native,
        &y_wrapped
    );
}

#[test]
fn mpi_block_jacobi_sor_native_differs_from_wrapped_with_coupling() {
    let _guard = mpi_test_guard();
    let Some(comm) = mpi_world() else {
        return;
    };
    let (dist, _, row_start, _) = make_dist_poisson(&comm, 2);

    let wrapped_opts = MpiPcOptions {
        global_pc: GlobalPcKind::BlockJacobi,
        local_pc: LocalPcKind::Sor,
        local_apply_mode: DistLocalApplyMode::WrappedLocal,
        ..MpiPcOptions::default()
    };
    let native_opts = MpiPcOptions {
        local_apply_mode: DistLocalApplyMode::NativeLocalHalo,
        ..wrapped_opts.clone()
    };

    let wrapped = DistPcAdapter::build(&dist, DistPcBuilder::BlockJacobi { opts: wrapped_opts })
        .expect("wrapped sor build");
    let native = DistPcAdapter::build(&dist, DistPcBuilder::BlockJacobi { opts: native_opts })
        .expect("native sor build");

    let mut rhs = vec![0.0; dist.local_nrows()];
    if row_start == 0 {
        rhs[0] = 1.0;
    }

    let mut y_wrapped = vec![0.0; rhs.len()];
    let mut y_native = vec![0.0; rhs.len()];
    wrapped
        .apply(PcSide::Left, &rhs, &mut y_wrapped)
        .expect("wrapped sor apply");
    native
        .apply(PcSide::Left, &rhs, &mut y_native)
        .expect("native sor apply");

    let l1_local: f64 = y_wrapped
        .iter()
        .zip(y_native.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    let l1_global = comm.all_reduce_f64(l1_local);
    assert!(
        l1_global > 1e-12,
        "expected native SOR halo correction to differ from wrapper mode"
    );
}

#[test]
fn mpi_dist_builder_constructs_native_ras_and_asm() {
    let _guard = mpi_test_guard();
    let Some(comm) = mpi_world() else {
        return;
    };
    let (dist, _, _, _) = make_dist_poisson(&comm, 2);

    let ras = DistPcAdapter::build(
        &dist,
        DistPcBuilder::Ras {
            overlap: 1,
            subdomain_hint: None,
            block_solver: AsmBlockSolver::Csr,
            inner_pc: AsmInnerPc::Ilu0,
            weighting: Weighting::None,
            coarse_strategy: DistCoarseStrategy::None,
            local_apply_mode: DistLocalApplyMode::NativeLocalHalo,
        },
    )
    .expect("native distributed ras build");

    let asm = DistPcAdapter::build(
        &dist,
        DistPcBuilder::Asm {
            overlap: 1,
            subdomain_hint: None,
            block_solver: AsmBlockSolver::Csr,
            inner_pc: AsmInnerPc::Ilu0,
            weighting: Weighting::None,
            coarse_strategy: DistCoarseStrategy::None,
            local_apply_mode: DistLocalApplyMode::NativeLocalHalo,
        },
    )
    .expect("native distributed asm build");

    let rhs: Vec<f64> = (0..dist.local_nrows()).map(|i| (i + 1) as f64).collect();
    let mut y_ras = vec![0.0; rhs.len()];
    let mut y_asm = vec![0.0; rhs.len()];
    ras.apply(PcSide::Left, &rhs, &mut y_ras)
        .expect("ras apply");
    asm.apply(PcSide::Left, &rhs, &mut y_asm)
        .expect("asm apply");

    assert!(
        y_ras.iter().all(|v| v.is_finite()) && y_asm.iter().all(|v| v.is_finite()),
        "distributed ASM/RAS outputs must be finite"
    );
}

#[test]
fn mpi_asm_and_ras_strict_accept_supported_native_prerequisites() {
    let _guard = mpi_test_guard();
    let Some(comm) = mpi_world() else {
        return;
    };
    let (dist, _, _, _) = make_dist_poisson(&comm, 2);

    DistPcAdapter::build(
        &dist,
        DistPcBuilder::Asm {
            overlap: 1,
            subdomain_hint: None,
            block_solver: AsmBlockSolver::Csr,
            inner_pc: AsmInnerPc::Ilu0,
            weighting: Weighting::None,
            coarse_strategy: DistCoarseStrategy::None,
            local_apply_mode: DistLocalApplyMode::NativeStrict,
        },
    )
    .expect("strict mode should accept ASM distributed builder when prerequisites are met");

    DistPcAdapter::build(
        &dist,
        DistPcBuilder::Ras {
            overlap: 1,
            subdomain_hint: None,
            block_solver: AsmBlockSolver::Csr,
            inner_pc: AsmInnerPc::Ilu0,
            weighting: Weighting::None,
            coarse_strategy: DistCoarseStrategy::None,
            local_apply_mode: DistLocalApplyMode::NativeStrict,
        },
    )
    .expect("strict mode should accept RAS distributed builder when prerequisites are met");
}

#[test]
fn mpi_asm_strict_rejected_on_overlap_mode_with_structured_keys() {
    let _guard = mpi_test_guard();
    let Some(comm) = mpi_world() else {
        return;
    };
    let (dist, _, _, _) = make_dist_poisson(&comm, 2);

    let err = match DistPcAdapter::build(
        &dist,
        DistPcBuilder::Asm {
            overlap: 0,
            subdomain_hint: None,
            block_solver: AsmBlockSolver::Csr,
            inner_pc: AsmInnerPc::Ilu0,
            weighting: Weighting::None,
            coarse_strategy: DistCoarseStrategy::None,
            local_apply_mode: DistLocalApplyMode::NativeStrict,
        },
    ) {
        Ok(_) => panic!("strict mode should reject overlap=0 ASM distributed builder"),
        Err(err) => err,
    };

    let msg = format!("{err}");
    assert!(
        msg.contains("err_key=pc_dist_strict_mode_rejected")
            && msg.contains("pc_dist_local_apply=strict")
            && msg.contains("pc_global=Asm")
            && msg.contains("detail_key=overlap_mode"),
        "unexpected strict-mode error: {msg}"
    );
}

#[test]
fn mpi_ras_strict_rejected_on_local_solver_support_with_structured_keys() {
    let _guard = mpi_test_guard();
    let Some(comm) = mpi_world() else {
        return;
    };
    let (dist, _, _, _) = make_dist_poisson(&comm, 2);

    let err = match DistPcAdapter::build(
        &dist,
        DistPcBuilder::Ras {
            overlap: 1,
            subdomain_hint: None,
            block_solver: AsmBlockSolver::LuDense,
            inner_pc: AsmInnerPc::Ilu0,
            weighting: Weighting::None,
            coarse_strategy: DistCoarseStrategy::None,
            local_apply_mode: DistLocalApplyMode::NativeStrict,
        },
    ) {
        Ok(_) => panic!("strict mode should reject unsupported RAS solver combination"),
        Err(err) => err,
    };

    let msg = format!("{err}");
    assert!(
        msg.contains("err_key=pc_dist_strict_mode_rejected")
            && msg.contains("pc_dist_local_apply=strict")
            && msg.contains("pc_global=Ras")
            && msg.contains("detail_key=local_solver_support"),
        "unexpected strict-mode error: {msg}"
    );
}

#[test]
fn mpi_asm_strict_rejected_on_comm_plan_constraints_with_structured_keys() {
    let _guard = mpi_test_guard();
    let Some(comm) = mpi_world() else {
        return;
    };
    let solo = comm.split(comm.rank() as i32, 0);
    let (dist, _, _, _) = make_dist_poisson(&solo, 2);

    let err = match DistPcAdapter::build(
        &dist,
        DistPcBuilder::Asm {
            overlap: 1,
            subdomain_hint: None,
            block_solver: AsmBlockSolver::Csr,
            inner_pc: AsmInnerPc::Ilu0,
            weighting: Weighting::None,
            coarse_strategy: DistCoarseStrategy::None,
            local_apply_mode: DistLocalApplyMode::NativeStrict,
        },
    ) {
        Ok(_) => panic!("strict mode should reject communicator size=1"),
        Err(err) => err,
    };

    let msg = format!("{err}");
    assert!(
        msg.contains("err_key=pc_dist_strict_mode_rejected")
            && msg.contains("pc_dist_local_apply=strict")
            && msg.contains("pc_global=Asm")
            && msg.contains("detail_key=communication_plan_constraints"),
        "unexpected strict-mode error: {msg}"
    );
}

#[test]
fn mpi_asm_two_level_improves_stationary_convergence() {
    let _guard = mpi_test_guard();
    let Some(comm) = mpi_world() else {
        return;
    };
    let (dist, _, _, _) = make_dist_poisson(&comm, 8);
    let rhs: Vec<f64> = vec![1.0; dist.local_nrows()];

    let mut one_level = AsmPc::ras(
        1,
        None,
        AsmBlockSolver::Csr,
        AsmInnerPc::Ilu0,
        Weighting::None,
    )
    .with_dist_coarse_strategy(DistCoarseStrategy::None);
    one_level.setup(&dist).expect("one-level setup");
    let one_level_res = stationary_iterations(&dist, &one_level, &rhs, 8);

    let mut two_level = AsmPc::ras(
        1,
        None,
        AsmBlockSolver::Csr,
        AsmInnerPc::Ilu0,
        Weighting::None,
    )
    .with_dist_coarse_strategy(DistCoarseStrategy::RootGather);
    two_level.setup(&dist).expect("two-level setup");
    let two_level_res = stationary_iterations(&dist, &two_level, &rhs, 8);

    assert!(
        two_level_res < one_level_res,
        "expected coarse correction to reduce residual: one-level={one_level_res:e}, two-level={two_level_res:e}"
    );
}

#[test]
fn mpi_asm_reports_setup_and_apply_costs() {
    let _guard = mpi_test_guard();
    let Some(comm) = mpi_world() else {
        return;
    };
    let (dist, _, _, _) = make_dist_poisson(&comm, 8);
    let rhs: Vec<f64> = vec![1.0; dist.local_nrows()];
    let mut out = vec![0.0; rhs.len()];

    let mut asm = AsmPc::ras(
        1,
        None,
        AsmBlockSolver::Csr,
        AsmInnerPc::Ilu0,
        Weighting::None,
    )
    .with_dist_coarse_strategy(DistCoarseStrategy::RootGather);

    let t_setup = Instant::now();
    asm.setup(&dist).expect("setup");
    let setup_s = t_setup.elapsed().as_secs_f64();

    let t_apply = Instant::now();
    asm.apply(PcSide::Left, &rhs, &mut out).expect("apply");
    let apply_s = t_apply.elapsed().as_secs_f64();

    let setup_max = comm.all_reduce_f64(setup_s);
    let apply_max = comm.all_reduce_f64(apply_s);
    eprintln!(
        "distributed asm timing: setup_max={setup_max:.6e}s apply_max={apply_max:.6e}s ranks={}",
        comm.size()
    );
    assert!(setup_max > 0.0 && apply_max > 0.0);
    assert!(out.iter().all(|v| v.is_finite()));
}

#[test]
fn mpi_block_jacobi_strict_rejects_wrapped_only_with_structured_keys() {
    let _guard = mpi_test_guard();
    let Some(comm) = mpi_world() else {
        return;
    };
    let (dist, _, _, _) = make_dist_poisson(&comm, 2);

    let opts = MpiPcOptions {
        global_pc: GlobalPcKind::BlockJacobi,
        local_pc: LocalPcKind::Spai,
        local_apply_mode: DistLocalApplyMode::NativeStrict,
        ..MpiPcOptions::default()
    };
    let err = match DistPcAdapter::build(&dist, DistPcBuilder::BlockJacobi { opts }) {
        Ok(_) => panic!("strict mode should reject wrapped-only block-jacobi local pc"),
        Err(err) => err,
    };

    let msg = format!("{err}");
    assert!(
        msg.contains("err_key=pc_dist_strict_mode_rejected")
            && msg.contains("pc_dist_local_apply=strict")
            && msg.contains("pc_global=BlockJacobi")
            && msg.contains("detail_key=unsupported_local_pc"),
        "unexpected strict-mode error: {msg}"
    );
}
