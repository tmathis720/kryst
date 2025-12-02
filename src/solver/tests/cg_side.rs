#[cfg(test)]
mod tests_cg_side {
    use crate::algebra::prelude::*;
    use crate::context::ksp_context::{KspContext, SolverType};
    use crate::context::pc_context::PcType;
    use crate::error::KError;
    use crate::matrix::op::LinOp;
    use crate::preconditioner::PcSide;
    use faer::Mat;
    use std::sync::Arc;

    #[test]
    #[cfg(not(feature = "complex"))]
    fn cg_rejects_right_side() {
        // SPD 2x2
        let a = Mat::<R>::from_fn(
            2,
            2,
            |i, j| if i == j { R::from(2.0) } else { R::from(1.0) },
        );
        let b = [R::from(1.0), R::default()];
        let amat: Arc<dyn LinOp<S = f64>> = Arc::new(a);

        let mut ksp = KspContext::new();
        ksp.set_type(SolverType::Cg).unwrap();
        ksp.set_pc_type(PcType::Jacobi, None).unwrap();
        ksp.set_operators(amat.clone(), None);
        ksp.pc_side = PcSide::Right; // invalid for CG

        let mut x = [R::default(); 2];
        let err = ksp.solve(&b, &mut x).unwrap_err();

        match err {
            KError::SolveError(msg) | KError::InvalidInput(msg) => {
                assert!(msg.to_lowercase().contains("left"))
            }
            _ => panic!("expected SolveError, got {:?}", err),
        }
    }
}
