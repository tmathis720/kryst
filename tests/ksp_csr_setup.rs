use std::sync::Arc;

use kryst::context::ksp_context::{KspContext, SolverType};
use kryst::context::pc_context::PcType;
use kryst::matrix::op::LinOp;
use kryst::matrix::sparse::CsrMatrix;

#[test]
fn ksp_setup_accepts_csr_without_dense_downcast() {
    // Build a simple 3x3 tridiagonal matrix in CSR format
    let csr = CsrMatrix::from_csr(
        3,
        3,
        vec![0, 2, 4, 5],
        vec![0, 1, 0, 2, 1],
        vec![2.0, -1.0, -1.0, -1.0, 2.0],
    );
    let a: Arc<dyn LinOp<S = f64>> = Arc::new(csr);

    // Configure KSP with a Jacobi preconditioner and GMRES solver
    let mut ksp = KspContext::new();
    ksp.set_type(SolverType::Gmres)
        .unwrap()
        .set_pc_type(PcType::Jacobi, None)
        .unwrap()
        .set_operators(a.clone(), None);

    // setup should succeed without requiring a dense matrix downcast
    ksp.setup().unwrap();
}
