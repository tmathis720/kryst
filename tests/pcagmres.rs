#![cfg(all(feature = "backend-faer", not(feature = "complex")))]
use std::sync::Arc;

use faer::Mat;
use kryst::algebra::prelude::*;
use kryst::assert_vec_close;
use kryst::error::KError;
use kryst::matrix::op::LinOp;
use kryst::parallel::UniverseComm;
use kryst::preconditioner::{PcSide, Preconditioner};
use kryst::solver::{PcaGmresSolver, PcaPcMode};

#[test]
fn pcagmres_solves_dd_nonsym_right_pc() {
    // Build simple non-symmetric, diagonally dominant 5x5 matrix
    let data: [[R; 5]; 5] = [
        [
            R::from(10.0),
            R::from(2.0),
            R::default(),
            R::default(),
            R::default(),
        ],
        [
            R::from(3.0),
            R::from(15.0),
            R::from(4.0),
            R::default(),
            R::default(),
        ],
        [
            R::default(),
            R::from(-2.0),
            R::from(8.0),
            R::from(1.0),
            R::default(),
        ],
        [
            R::default(),
            R::default(),
            R::from(1.0),
            R::from(7.0),
            R::from(3.0),
        ],
        [
            R::default(),
            R::default(),
            R::default(),
            R::from(2.0),
            R::from(12.0),
        ],
    ];
    let mut a = Mat::<R>::zeros(5, 5);
    for i in 0..5 {
        for j in 0..5 {
            a[(i, j)] = data[i][j];
        }
    }
    let aop: Arc<dyn LinOp<S = f64>> = Arc::new(a.clone());

    // Right Jacobi preconditioner
    struct RJ {
        d: [R; 5],
    }
    impl Preconditioner for RJ {
        fn setup(&mut self, _pmat: &dyn LinOp<S = f64>) -> Result<(), KError> {
            Ok(())
        }
        fn apply(&self, side: PcSide, r: &[f64], z: &mut [f64]) -> Result<(), KError> {
            match side {
                PcSide::Right => {
                    for i in 0..5 {
                        z[i] = r[i] / self.d[i];
                    }
                }
                _ => z.copy_from_slice(r),
            }
            Ok(())
        }
    }
    let mut pc = RJ {
        d: [
            R::from(10.0),
            R::from(15.0),
            R::from(8.0),
            R::from(7.0),
            R::from(12.0),
        ],
    };

    // x_true = [1,2,3,4,5], b = A * x_true
    let x_true: Vec<f64> = (1..=5).map(|i| i as f64).collect();
    let mut b = vec![0.0f64; 5];
    aop.matvec(&x_true, &mut b);

    let mut x = vec![0.0f64; 5];
    let mut solver = PcaGmresSolver::new(20, 1, 1, 1e-10, 200);
    solver.pc_mode = PcaPcMode::Right;
    let stats = solver
        .solve_f64(
            aop.as_ref(),
            Some(&mut pc),
            &b,
            &mut x,
            PcSide::Right,
            &UniverseComm::NoComm(kryst::parallel::NoComm),
            None,
            None,
        )
        .unwrap();
    let tol = 1e-6;
    assert!(stats.final_residual <= tol, "res={}", stats.final_residual);
    let x_s: Vec<S> = x.iter().copied().map(S::from_real).collect();
    let x_true_s: Vec<S> = x_true.iter().copied().map(S::from_real).collect();
    assert_vec_close!("pcagmres solution", &x_s, &x_true_s);
}
