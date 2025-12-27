use std::sync::Arc;

use kryst::algebra::prelude::*;
use kryst::context::ksp_context::{KspContext, SolverType};
use kryst::context::pc_context::PcType;
use kryst::matrix::LinOp;

#[test]
fn no_backend_gmres_pcnone_solves_identity() {
    struct I(usize);
    impl LinOp for I {
        type S = S;
        fn dims(&self) -> (usize, usize) {
            (self.0, self.0)
        }
        fn matvec(&self, x: &[S], y: &mut [S]) {
            y.copy_from_slice(x);
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    let op: Arc<dyn LinOp<S = S>> = Arc::new(I(8));
    let b = vec![S::from_real(1.0); 8];
    let mut x = vec![S::zero(); 8];

    let mut ksp = KspContext::new();
    ksp.set_type(SolverType::Gmres).unwrap();
    ksp.set_pc_type(PcType::None, None).unwrap();
    ksp.set_operators(op, None);

    let stats = ksp.solve(&b, &mut x).unwrap();
    assert!(stats.final_residual.is_finite());
}

#[cfg(feature = "backend-nalgebra")]
#[test]
fn nalgebra_preonly_lu_solves() {
    let a = nalgebra::DMatrix::<S>::from_row_slice(2, 2, &[
        S::from_real(2.0),
        S::from_real(1.0),
        S::from_real(1.0),
        S::from_real(2.0),
    ]);
    let op = Arc::new(kryst::matrix::op_nalgebra::NalgebraDenseOp::new(Arc::new(a)));

    let b = vec![S::from_real(3.0), S::from_real(3.0)];
    let mut x = vec![S::zero(); 2];

    let mut ksp = KspContext::new();
    ksp.set_preonly_with_pc(PcType::Lu, None).unwrap();
    ksp.set_operators(op, None);

    let stats = ksp.solve(&b, &mut x).unwrap();
    assert!(stats.final_residual.is_finite());
}
