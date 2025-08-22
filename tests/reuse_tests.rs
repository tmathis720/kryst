use std::sync::Arc;
use kryst::matrix::{CsrOp};
use kryst::matrix::sparse::CsrMatrix;
use kryst::context::ksp_context::{KspContext, SolverType};
use kryst::preconditioner::{PcReusePolicy};
use kryst::context::pc_context::PcType;

#[test]
fn pc_rebuilds_on_structure_change() {
    let a1 = Arc::new(CsrMatrix::from_csr(2,2, vec![0,1,2], vec![0,1], vec![1.0,2.0]));
    let op1 = Arc::new(CsrOp::new(a1));
    let a2 = Arc::new(CsrMatrix::from_csr(2,2, vec![0,1,2], vec![0,1], vec![3.0,4.0]));
    let op2 = Arc::new(CsrOp::new(a2));

    let mut ksp = KspContext::new();
    ksp.set_type(SolverType::Gmres).unwrap();
    ksp.set_pc_type(PcType::Jacobi, None).unwrap();
    ksp.set_operators(op1.clone(), None);

    ksp.setup().unwrap();
    let sid1 = ksp.last_pc_sid();

    ksp.set_operators(op2.clone(), None);
    op2.mark_structure_changed();
    ksp.setup().unwrap();
    assert_ne!(sid1, ksp.last_pc_sid());
}

#[test]
fn jacobi_numeric_update_without_rebuild() {
    let a = Arc::new(CsrMatrix::from_csr(2,2, vec![0,1,2], vec![0,1], vec![1.0,2.0]));
    let op = Arc::new(CsrOp::new(a));

    let mut ksp = KspContext::new();
    ksp.set_type(SolverType::Gmres).unwrap();
    ksp.set_pc_type(PcType::Jacobi, None).unwrap();
    ksp.set_pc_reuse_policy(PcReusePolicy::ReuseNumeric);
    ksp.set_operators(op.clone(), None);

    ksp.setup().unwrap();
    let sid0 = ksp.last_pc_sid();
    let vid0 = ksp.last_pc_vid();

    op.mark_values_changed();
    ksp.setup().unwrap();
    assert_eq!(sid0, ksp.last_pc_sid());
    assert_ne!(vid0, ksp.last_pc_vid());
}
