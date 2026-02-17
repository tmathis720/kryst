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
        use_vertices: true,
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
        use_vertices: true,
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
        use_vertices: false,
    });
    let err = bddc.setup(&LayoutMismatch).unwrap_err();
    assert!(matches!(err, KError::InvalidInput(_)));
}
