#![cfg(all(feature = "backend-faer", feature = "complex"))]

use std::sync::Arc;

use faer::Mat;
use kryst::algebra::prelude::S;
use kryst::config::options::{KspOptions, PcOptions};
use kryst::context::ksp_context::{KspContext, SolverType};
use kryst::matrix::op::DenseOp;

#[test]
fn nested_ksp_pc_solves_small_complex_system() {
    let a = Mat::<S>::from_fn(3, 3, |i, j| {
        if i == j {
            S::from_real(4.0)
        } else if (i as isize - j as isize).abs() == 1 {
            S::from_real(-1.0)
        } else {
            S::zero()
        }
    });
    let op = Arc::new(DenseOp::new(Arc::new(a)));

    let mut ksp = KspContext::new();
    ksp.set_type(SolverType::Gmres).expect("outer type");

    let ksp_opts = KspOptions {
        maxits: Some(30),
        rtol: Some(1e-10),
        ..Default::default()
    };
    let pc_opts = PcOptions {
        pc_type: Some("ksp".into()),
        pc_ksp_ksp_type: Some("richardson".into()),
        pc_ksp_maxits: Some(2),
        pc_ksp_rtol: Some(1e-2),
        pc_ksp_pc_type: Some("jacobi".into()),
        ..Default::default()
    };

    ksp.set_from_all_options(&ksp_opts, &pc_opts).expect("opts");
    ksp.set_operators(op, None);

    let b = vec![S::from_real(1.0); 3];
    let mut x = vec![S::zero(); 3];
    let stats = ksp.solve(&b, &mut x).expect("solve");

    assert!(
        stats.reason.is_converged()
            || matches!(
                stats.reason,
                kryst::utils::convergence::ConvergedReason::DivergedMaxIts
            )
    );
    assert!(stats.nested_pc_failure.is_none());
    assert!(stats.final_residual.is_finite());
}

#[test]
fn nested_ksp_pc_complex_fieldsplit_composition_reports_consistent_state() {
    let a = Mat::<S>::from_fn(4, 4, |i, j| {
        if i == j {
            S::from_real(5.0)
        } else if (i as isize - j as isize).abs() == 1 {
            S::from_real(-1.0)
        } else {
            S::zero()
        }
    });
    let op = Arc::new(DenseOp::new(Arc::new(a)));

    let mut ksp = KspContext::new();
    ksp.set_type(SolverType::Gmres).expect("outer type");

    let ksp_opts = KspOptions {
        maxits: Some(40),
        rtol: Some(1e-10),
        ..Default::default()
    };
    let pc_opts = PcOptions {
        pc_type: Some("fieldsplit".into()),
        pc_fieldsplit_block_sizes: Some(vec![2, 2]),
        pc_fieldsplit_type: Some("schur".into()),
        pc_fieldsplit_schur_fact_type: Some("full".into()),
        pc_fieldsplit_schur_precondition: Some("user".into()),
        pc_fieldsplit_prefixes: Some(vec!["pc_fieldsplit_0_".into(), "pc_fieldsplit_1_".into()]),
        scoped_children: vec![
            (
                "pc_fieldsplit_0_".into(),
                Box::new(PcOptions {
                    pc_type: Some("ksp".into()),
                    pc_ksp_ksp_type: Some("richardson".into()),
                    pc_ksp_maxits: Some(2),
                    pc_ksp_rtol: Some(1e-2),
                    pc_ksp_pc_type: Some("jacobi".into()),
                    ..Default::default()
                }),
            ),
            (
                "pc_fieldsplit_1_".into(),
                Box::new(PcOptions {
                    pc_type: Some("jacobi".into()),
                    ..Default::default()
                }),
            ),
        ],
        ..Default::default()
    };

    ksp.set_from_all_options(&ksp_opts, &pc_opts).expect("opts");
    ksp.set_operators(op, None);

    let b = vec![S::from_real(1.0); 4];
    let mut x = vec![S::zero(); 4];
    let stats = ksp.solve(&b, &mut x).expect("solve");

    assert!(stats.nested_pc_failure.is_none());
    assert!(
        stats.reason.is_converged()
            || matches!(
                stats.reason,
                kryst::utils::convergence::ConvergedReason::DivergedMaxIts
            )
    );
}
