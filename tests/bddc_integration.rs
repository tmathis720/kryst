#![cfg(all(feature = "backend-faer", not(feature = "complex")))]

use kryst::algebra::prelude::*;
use kryst::context::ksp_context::Workspace;
use kryst::error::KError;
use kryst::matrix::op::{DistLayout, LinOp};
use kryst::parallel::{NoComm, UniverseComm};
use kryst::preconditioner::bddc::{BddcConfig, BddcPc};
use kryst::preconditioner::{PcSide, Preconditioner};
use kryst::solver::{GmresSolver, LinearSolver};

fn decomposed_spd(n: usize) -> (faer::Mat<R>, Vec<R>) {
    let mut a = faer::Mat::<R>::zeros(n, n);
    for i in 0..n {
        a[(i, i)] = 4.0;
        if i > 0 {
            // weaker interface-style coupling improves BDDC separation effect
            let off = if i == n / 2 { -0.2 } else { -1.0 };
            a[(i, i - 1)] = off;
            a[(i - 1, i)] = off;
        }
    }
    let b = vec![1.0; n];
    (a, b)
}

#[test]
fn bddc_reduces_iterations_vs_none_on_domain_decomposed_spd() {
    let (a, b) = decomposed_spd(80);
    let comm = UniverseComm::NoComm(NoComm);

    let mut x_none = vec![0.0; b.len()];
    let mut ws_none = Workspace::new(b.len());
    let mut gmres_none = GmresSolver::new(25, 1e-10, 400);
    gmres_none.setup_workspace(&mut ws_none);
    let stats_none = gmres_none
        .solve_f64(
            &a,
            None,
            &b,
            &mut x_none,
            PcSide::Left,
            &comm,
            None,
            Some(&mut ws_none),
        )
        .unwrap();

    let mut bddc = BddcPc::new(BddcConfig {
        coarse_ksp_type: Some("preonly".into()),
        coarse_pc_type: Some("lu".into()),
        local_ksp_type: Some("cg".into()),
        local_pc_type: Some("jacobi".into()),
        use_vertices: true,
        constraint_selection: None,
        scaling: None,
    });
    bddc.setup(&a).unwrap();

    let mut x_bddc = vec![0.0; b.len()];
    let mut ws_bddc = Workspace::new(b.len());
    let mut gmres_bddc = GmresSolver::new(25, 1e-10, 400);
    gmres_bddc.setup_workspace(&mut ws_bddc);
    let stats_bddc = gmres_bddc
        .solve_f64(
            &a,
            Some(&mut bddc),
            &b,
            &mut x_bddc,
            PcSide::Left,
            &comm,
            None,
            Some(&mut ws_bddc),
        )
        .unwrap();

    assert!(stats_bddc.iterations < stats_none.iterations);
}

#[test]
fn bddc_rejects_non_square_operator() {
    struct Rect;
    impl LinOp for Rect {
        type S = S;
        fn dims(&self) -> (usize, usize) {
            (3, 4)
        }
        fn matvec(&self, x: &[S], y: &mut [S]) {
            y.fill(0.0);
            for i in 0..3 {
                y[i] = x[i];
            }
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    let mut bddc = BddcPc::new(BddcConfig {
        coarse_ksp_type: None,
        coarse_pc_type: None,
        local_ksp_type: None,
        local_pc_type: None,
        use_vertices: true,
        constraint_selection: None,
        scaling: None,
    });
    let err = bddc.setup(&Rect).unwrap_err();
    assert!(matches!(err, KError::InvalidInput(_)));
}

#[test]
fn bddc_validates_distributed_layout_consistency() {
    struct LayoutMismatch;
    impl LinOp for LayoutMismatch {
        type S = S;
        fn dims(&self) -> (usize, usize) {
            (4, 4)
        }
        fn matvec(&self, x: &[S], y: &mut [S]) {
            y.copy_from_slice(x);
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
        fn dist_layout(&self) -> Option<&DistLayout> {
            static L: std::sync::OnceLock<DistLayout> = std::sync::OnceLock::new();
            Some(L.get_or_init(|| DistLayout {
                global_rows: 8,
                global_cols: 8,
                row_start: 0,
                row_end: 2,
                col_start: 0,
                col_end: 2,
            }))
        }
    }

    let mut bddc = BddcPc::new(BddcConfig {
        coarse_ksp_type: None,
        coarse_pc_type: None,
        local_ksp_type: None,
        local_pc_type: None,
        use_vertices: false,
        constraint_selection: None,
        scaling: None,
    });
    let err = bddc.setup(&LayoutMismatch).unwrap_err();
    assert!(matches!(err, KError::InvalidInput(_)));
}

#[test]
fn bddc_reports_sparse_diagnostics_and_constraint_modes() {
    let (a, _) = decomposed_spd(24);
    let mut bddc_vertices = BddcPc::new(BddcConfig {
        coarse_ksp_type: Some("preonly".into()),
        coarse_pc_type: Some("jacobi".into()),
        local_ksp_type: Some("cg".into()),
        local_pc_type: Some("jacobi".into()),
        use_vertices: true,
        constraint_selection: Some(kryst::preconditioner::bddc::BddcConstraintSelection::Vertices),
        scaling: Some(kryst::preconditioner::bddc::BddcScaling::Uniform),
    });
    bddc_vertices.setup(&a).unwrap();
    let d = bddc_vertices.diagnostics().expect("diagnostics");
    assert!(d.local_nnz > 0);
    assert!(d.coarse_nnz > 0);
    assert!(d.local_solve_route.contains("cg"));
    assert!(d.coarse_solve_route.contains("jacobi"));
    assert!(d.comm_interface_dofs > 0);

    let mut bddc_interface = BddcPc::new(BddcConfig {
        coarse_ksp_type: Some("preonly".into()),
        coarse_pc_type: Some("jacobi".into()),
        local_ksp_type: Some("preonly".into()),
        local_pc_type: Some("jacobi".into()),
        use_vertices: false,
        constraint_selection: Some(kryst::preconditioner::bddc::BddcConstraintSelection::Interface),
        scaling: Some(kryst::preconditioner::bddc::BddcScaling::DeluxeLike),
    });
    bddc_interface.setup(&a).unwrap();
    let x = vec![1.0; 24];
    let mut y_v = vec![0.0; 24];
    let mut y_i = vec![0.0; 24];
    bddc_vertices.apply(PcSide::Left, &x, &mut y_v).unwrap();
    bddc_interface.apply(PcSide::Left, &x, &mut y_i).unwrap();
    let diff: f64 = y_v.iter().zip(y_i.iter()).map(|(a, b)| (a - b).abs()).sum();
    assert!(
        diff > 1e-9,
        "different constraint/scaling modes should alter response"
    );
}

#[test]
fn bddc_supports_coarse_backend_combinations_and_reports_runtime_diagnostics() {
    let (a, _) = decomposed_spd(32);
    let combos = [("preonly", "lu"), ("cg", "jacobi"), ("gmres", "ilu")];

    for (ksp, pc) in combos {
        let mut bddc = BddcPc::new(BddcConfig {
            coarse_ksp_type: Some(ksp.into()),
            coarse_pc_type: Some(pc.into()),
            local_ksp_type: Some("cg".into()),
            local_pc_type: Some("jacobi".into()),
            use_vertices: true,
            constraint_selection: None,
            scaling: None,
        });
        bddc.setup(&a).expect("setup");
        let x = vec![1.0; 32];
        let mut y = vec![0.0; 32];
        bddc.apply(PcSide::Left, &x, &mut y).expect("apply");
        let d = bddc.diagnostics().expect("diagnostics");
        assert!(!d.coarse_solve_route.is_empty());
        assert!(d.coarse_solve_route.contains(ksp) || d.coarse_solve_route.contains("preonly"));
        assert!(d.coarse_final_residual.is_finite());
        assert!(d.coarse_iterations < 2000);
    }
}
