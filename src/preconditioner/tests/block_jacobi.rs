use crate::algebra::prelude::*;
use crate::config::options::PcOptions;
use crate::context::pc_context::{PcFactory, PcType};
use crate::matrix::op::DenseOp;
use crate::preconditioner::{PcSide, Preconditioner};
use std::sync::Arc;

#[test]
#[cfg(not(feature = "complex"))]
fn block_jacobi_apply_matches_block_solve() {
    let mat = faer::Mat::<f64>::from_fn(4, 4, |i, j| match (i, j) {
        (0, 0) => 4.0,
        (0, 1) => 1.0,
        (1, 0) => 1.0,
        (1, 1) => 3.0,
        (2, 2) => 2.0,
        (3, 3) => 5.0,
        _ => 0.0,
    });
    let op = DenseOp::new(Arc::new(mat));

    let mut opts = PcOptions::default();
    opts.jacobi_block_size = Some(2);

    let mut pc = PcFactory::create_preconditioner(PcType::BlockJacobi, Some(&opts))
        .expect("block jacobi preconditioner");
    pc.setup(&op).expect("block jacobi setup");

    let rhs = vec![S::from_real(1.0), S::from_real(2.0), S::from_real(3.0), S::from_real(4.0)];
    let mut out = vec![S::zero(); rhs.len()];
    pc.apply(PcSide::Left, &rhs, &mut out)
        .expect("block jacobi apply");

    let expected = [1.0 / 11.0, 7.0 / 11.0, 1.5, 0.8];
    for (y, exp) in out.iter().zip(expected.iter()) {
        assert!((y.real() - exp).abs() < 1e-10);
    }
}
