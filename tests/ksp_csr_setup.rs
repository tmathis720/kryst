#![cfg(all(feature = "backend-faer", not(feature = "complex")))]
use std::sync::Arc;

use kryst::config::options::{CgVariant, KspOptions};
use kryst::context::ksp_context::{KspContext, SolverType};
use kryst::context::pc_context::PcType;
use kryst::matrix::csr::CsrMatrix as ScalarCsrMatrix;
use kryst::matrix::op::{GenericCsrOp, LinOp};
use kryst::matrix::spmv::plan::SpmvTuning;

#[test]
fn ksp_setup_accepts_csr_without_dense_downcast() {
    // Build a simple 3x3 tridiagonal matrix in CSR format
    let csr = Arc::new(ScalarCsrMatrix::<f64>::new(
        3,
        3,
        vec![0, 2, 4, 5],
        vec![0, 1, 0, 2, 1],
        vec![2.0, -1.0, -1.0, -1.0, 2.0],
    ));
    let a: Arc<dyn LinOp<S = f64>> = Arc::new(GenericCsrOp::new(csr, &SpmvTuning::default()));

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

#[test]
fn cg_and_pcg_solve_generic_csr_without_dense_materialization() {
    let csr = Arc::new(ScalarCsrMatrix::<f64>::new(
        4,
        4,
        vec![0, 2, 5, 8, 10],
        vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3],
        vec![4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0],
    ));
    let a: Arc<dyn LinOp<S = f64>> = Arc::new(GenericCsrOp::new(csr, &SpmvTuning::default()));
    let x_true = [1.0, -2.0, 0.5, 3.0];
    let mut b = vec![0.0; x_true.len()];
    a.matvec(&x_true, &mut b);

    for solver_type in [SolverType::Cg, SolverType::Pcg] {
        let mut ksp = KspContext::new();
        ksp.set_type(solver_type).expect("set CG-family solver");
        ksp.set_from_options(&KspOptions {
            cg_variant: Some(CgVariant::Pipelined),
            cg_replace_every: Some(0),
            ..KspOptions::default()
        })
        .expect("set pipelined CG options");
        ksp.set_pc_type(PcType::Jacobi, None)
            .expect("set Jacobi preconditioner");
        ksp.set_tolerances(1e-12, 1e-14, 1e8, 64);
        ksp.set_operators(a.clone(), None);
        ksp.setup().expect("generic CSR setup");

        let view = ksp.view();
        assert_eq!(
            view.solver_config
                .get("operator_route")
                .and_then(|value| value.as_str()),
            Some("generic-csr")
        );

        let mut x = vec![0.0; x_true.len()];
        let stats = ksp.solve(&b, &mut x).expect("generic CSR solve");
        assert!(
            stats.reason.is_converged(),
            "{solver_type:?} did not converge: {stats:?}"
        );
        assert!(
            stats.final_residual < 1e-10,
            "unexpected residual: {stats:?}"
        );
        assert_eq!(stats.final_true_residual, Some(stats.final_residual));
        for (actual, expected) in x.iter().zip(x_true) {
            assert!((actual - expected).abs() < 1e-10);
        }
    }
}
