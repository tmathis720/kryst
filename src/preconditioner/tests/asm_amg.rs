use crate::algebra::prelude::*;
use crate::assert_vec_close;
use crate::matrix::op::CsrOp;
use crate::matrix::sparse::CsrMatrix;
use crate::preconditioner::Preconditioner;
use crate::preconditioner::{
    Asm, AsmAmg, AsmCombine, AsmConfig, AsmLocalSolver, TwoLevelConfig, TwoLevelMode,
};
use std::sync::Arc;

fn identity(n: usize) -> CsrMatrix<R> {
    CsrMatrix::identity(n)
}

fn poisson_1d(n: usize) -> CsrMatrix<R> {
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::new();
    let mut values = Vec::new();
    row_ptr.push(0);
    for i in 0..n {
        if i > 0 {
            col_idx.push(i - 1);
            values.push(R::from(-1.0));
        }
        col_idx.push(i);
        values.push(R::from(2.0));
        if i + 1 < n {
            col_idx.push(i + 1);
            values.push(R::from(-1.0));
        }
        row_ptr.push(col_idx.len());
    }
    CsrMatrix::from_csr(n, n, row_ptr, col_idx, values)
}

#[test]
fn asm_identity_matches_input() {
    let a = Arc::new(identity(4));
    let op = CsrOp::new(a.clone());
    let mut cfg = AsmConfig::default();
    cfg.overlap = 1;
    cfg.combine = AsmCombine::Additive;
    cfg.local_solver = AsmLocalSolver::ILU;
    cfg.deterministic = true;
    let mut asm = Asm::with_config(cfg);
    asm.setup(&op).unwrap();
    let rhs = vec![R::from(1.0), R::from(2.0), R::from(3.0), R::from(4.0)];
    let mut out: Vec<R> = vec![R::default(); rhs.len()];
    asm.apply(crate::preconditioner::PcSide::Left, &rhs, &mut out)
        .unwrap();
    assert_vec_close!("asm identity", &rhs, &out);
}

#[test]
fn asm_amg_skip_coarse_matches_asm() {
    let a = Arc::new(poisson_1d(8));
    let op = CsrOp::new(a.clone());
    let mut cfg = AsmConfig::default();
    cfg.overlap = 1;
    cfg.combine = AsmCombine::Restricted;
    cfg.local_solver = AsmLocalSolver::ILU;
    cfg.deterministic = true;
    let mut asm = Asm::with_config(cfg.clone());
    asm.setup(&op).unwrap();
    let mut two = TwoLevelConfig::default();
    two.mode = TwoLevelMode::AdditiveCoarse;
    two.coarse_every = 5;
    let mut hybrid = AsmAmg::with_configs(cfg, two);
    hybrid.setup(&op).unwrap();
    let rhs = vec![R::from(1.0); 8];
    let mut y_asm: Vec<R> = vec![R::default(); 8];
    let mut y_hybrid: Vec<R> = vec![R::default(); 8];
    asm.apply(crate::preconditioner::PcSide::Left, &rhs, &mut y_asm)
        .unwrap();
    hybrid
        .apply(crate::preconditioner::PcSide::Left, &rhs, &mut y_hybrid)
        .unwrap();
    assert_vec_close!("asm vs asm+amg restricted", &y_asm, &y_hybrid);
}

#[test]
fn asm_numeric_update_refreshes_values() {
    let a1 = Arc::new(poisson_1d(6));
    let op1 = CsrOp::new(a1.clone());
    let mut cfg = AsmConfig::default();
    cfg.overlap = 1;
    cfg.combine = AsmCombine::Additive;
    cfg.local_solver = AsmLocalSolver::ILU;
    let mut asm = Asm::with_config(cfg);
    asm.setup(&op1).unwrap();
    let mut rhs = vec![R::from(1.0); 6];
    let mut out: Vec<R> = vec![R::default(); 6];
    asm.apply(crate::preconditioner::PcSide::Left, &rhs, &mut out)
        .unwrap();
    // Update numeric values by scaling matrix
    let mut scaled = poisson_1d(6);
    for v in scaled.values_mut() {
        *v *= R::from(2.0);
    }
    let op2 = CsrOp::new(Arc::new(scaled));
    asm.update_numeric(&op2).unwrap();
    rhs.iter_mut().for_each(|v| *v = R::from(1.0));
    asm.apply(crate::preconditioner::PcSide::Left, &rhs, &mut out)
        .unwrap();
}

#[cfg(feature = "complex")]
#[test]
fn asm_apply_s_matches_real_path() {
    use crate::algebra::bridge::BridgeScratch;
    use crate::algebra::prelude::*;
    use crate::ops::kpc::KPreconditioner;
    use crate::preconditioner::PcSide;

    let a = Arc::new(poisson_1d(6));
    let op = CsrOp::new(a.clone());
    let mut cfg = AsmConfig::default();
    cfg.overlap = 1;
    cfg.combine = AsmCombine::Additive;
    cfg.local_solver = AsmLocalSolver::ILU;
    cfg.deterministic = true;
    let mut asm = Asm::with_config(cfg);
    asm.setup(&op).expect("asm setup");

    let rhs_real: Vec<R> = (0..6).map(|i| R::from((i + 1) as f64)).collect();
    let mut out_real: Vec<R> = vec![R::default(); rhs_real.len()];
    asm.apply(PcSide::Left, &rhs_real, &mut out_real)
        .expect("asm apply real");

    let rhs_s: Vec<S> = rhs_real.iter().copied().map(S::from_real).collect();
    let mut out_s = vec![S::zero(); rhs_real.len()];
    let mut scratch = BridgeScratch::default();
    asm.apply_s(PcSide::Left, &rhs_s, &mut out_s, &mut scratch)
        .expect("asm apply_s");

    for (ys, yr) in out_s.iter().zip(out_real.iter()) {
        assert!(
            (ys.real() - yr).abs() < R::from(1e-10),
            "expected real match, got {} vs {}",
            ys,
            yr
        );
    }
}

#[cfg(feature = "complex")]
#[test]
fn asm_amg_apply_s_matches_real_path() {
    use crate::algebra::bridge::BridgeScratch;
    use crate::algebra::prelude::*;
    use crate::ops::kpc::KPreconditioner;
    use crate::preconditioner::PcSide;

    let a = Arc::new(poisson_1d(8));
    let op = CsrOp::new(a.clone());
    let mut asm_cfg = AsmConfig::default();
    asm_cfg.overlap = 1;
    asm_cfg.combine = AsmCombine::Restricted;
    asm_cfg.local_solver = AsmLocalSolver::ILU;
    asm_cfg.deterministic = true;
    let mut two_cfg = TwoLevelConfig::default();
    two_cfg.mode = TwoLevelMode::AdditiveCoarse;
    two_cfg.coarse_every = 1;
    let mut pc = AsmAmg::with_configs(asm_cfg, two_cfg);
    pc.setup(&op).expect("asm_amg setup");

    let rhs_real: Vec<R> = (0..8).map(|i| R::from((i + 1) as f64)).collect();
    let mut out_real: Vec<R> = vec![R::default(); rhs_real.len()];
    pc.apply(PcSide::Left, &rhs_real, &mut out_real)
        .expect("asm_amg apply real");

    let rhs_s: Vec<S> = rhs_real.iter().copied().map(S::from_real).collect();
    let mut out_s = vec![S::zero(); rhs_real.len()];
    let mut scratch = BridgeScratch::default();
    pc.apply_s(PcSide::Left, &rhs_s, &mut out_s, &mut scratch)
        .expect("asm_amg apply_s");

    for (ys, yr) in out_s.iter().zip(out_real.iter()) {
        assert!(
            (ys.real() - yr).abs() < R::from(1e-10),
            "expected real match, got {} vs {}",
            ys,
            yr
        );
    }
}
