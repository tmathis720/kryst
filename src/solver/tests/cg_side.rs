#[cfg(test)]
mod tests_cg_side {
    use crate::context::ksp_context::{KspContext, SolverType};
    use crate::context::pc_context::PcType;
    use crate::error::KError;
    use crate::matrix::op::LinOp;
    use crate::preconditioner::PcSide;
    use faer::Mat;
    use std::sync::Arc;

    #[test]
    fn cg_rejects_right_side() {
        // SPD 2x2
        let a = Mat::from_fn(2, 2, |i, j| if i == j { 2.0 } else { 1.0 });
        let b = [1.0, 0.0];
        let amat: Arc<dyn LinOp<S = f64>> = Arc::new(a);

        let mut ksp = KspContext::new();
        ksp.set_type(SolverType::Cg).unwrap();
        ksp.set_pc_type(PcType::Jacobi, None).unwrap();
        ksp.set_operators(amat.clone(), None);
        ksp.pc_side = PcSide::Right; // invalid for CG

        let mut x = [0.0, 0.0];
        let err = ksp.solve(&b, &mut x).unwrap_err();

        match err {
            KError::SolveError(msg) => assert!(msg.to_lowercase().contains("left")),
            _ => panic!("expected SolveError, got {:?}", err),
        }
    }
}
