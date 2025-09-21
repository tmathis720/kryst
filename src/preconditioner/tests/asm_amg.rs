use crate::matrix::op::CsrOp;
use crate::matrix::sparse::CsrMatrix;
use crate::preconditioner::Preconditioner;
use crate::preconditioner::{
    Asm, AsmAmg, AsmCombine, AsmConfig, AsmLocalSolver, TwoLevelConfig, TwoLevelMode,
};
use std::sync::Arc;

fn identity(n: usize) -> CsrMatrix<f64> {
    CsrMatrix::identity(n)
}

fn poisson_1d(n: usize) -> CsrMatrix<f64> {
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::new();
    let mut values = Vec::new();
    row_ptr.push(0);
    for i in 0..n {
        if i > 0 {
            col_idx.push(i - 1);
            values.push(-1.0);
        }
        col_idx.push(i);
        values.push(2.0);
        if i + 1 < n {
            col_idx.push(i + 1);
            values.push(-1.0);
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
    let rhs = vec![1.0, 2.0, 3.0, 4.0];
    let mut out = vec![0.0; rhs.len()];
    asm.apply(crate::preconditioner::PcSide::Left, &rhs, &mut out)
        .unwrap();
    assert_eq!(rhs, out);
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
    let rhs = vec![1.0; 8];
    let mut y_asm = vec![0.0; 8];
    let mut y_hybrid = vec![0.0; 8];
    asm.apply(crate::preconditioner::PcSide::Left, &rhs, &mut y_asm)
        .unwrap();
    hybrid
        .apply(crate::preconditioner::PcSide::Left, &rhs, &mut y_hybrid)
        .unwrap();
    for i in 0..rhs.len() {
        assert!((y_asm[i] - y_hybrid[i]).abs() < 1e-12);
    }
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
    let mut rhs = vec![1.0; 6];
    let mut out = vec![0.0; 6];
    asm.apply(crate::preconditioner::PcSide::Left, &rhs, &mut out)
        .unwrap();
    // Update numeric values by scaling matrix
    let mut scaled = poisson_1d(6);
    for v in scaled.values_mut() {
        *v *= 2.0;
    }
    let op2 = CsrOp::new(Arc::new(scaled));
    asm.update_numeric(&op2).unwrap();
    rhs.iter_mut().for_each(|v| *v = 1.0);
    asm.apply(crate::preconditioner::PcSide::Left, &rhs, &mut out)
        .unwrap();
}
