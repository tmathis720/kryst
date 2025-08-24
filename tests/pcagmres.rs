use std::sync::Arc;

use faer::Mat;
use kryst::error::KError;
use kryst::matrix::op::LinOp;
use kryst::parallel::UniverseComm;
use kryst::preconditioner::{PcSide, Preconditioner};
use kryst::solver::{LinearSolver, PcaGmresSolver, PcaPcMode};

#[test]
fn pcagmres_solves_dd_nonsym_right_pc() {
    // Build simple non-symmetric, diagonally dominant 5x5 matrix
    let data = [
        [10.0, 2.0, 0.0, 0.0, 0.0],
        [3.0, 15.0, 4.0, 0.0, 0.0],
        [0.0, -2.0, 8.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 7.0, 3.0],
        [0.0, 0.0, 0.0, 2.0, 12.0],
    ];
    let mut a = Mat::<f64>::zeros(5, 5);
    for i in 0..5 {
        for j in 0..5 {
            a[(i, j)] = data[i][j];
        }
    }
    let aop: Arc<dyn LinOp<S = f64>> = Arc::new(a.clone());

    // Right Jacobi preconditioner
    struct RJ {
        d: [f64; 5],
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
    let pc = RJ {
        d: [10.0, 15.0, 8.0, 7.0, 12.0],
    };

    // x_true = [1,2,3,4,5], b = A * x_true
    let x_true = [1.0, 2.0, 3.0, 4.0, 5.0];
    let mut b = [0.0; 5];
    aop.matvec(&x_true, &mut b);

    let mut x = [0.0; 5];
    let mut solver = PcaGmresSolver::new(20, 1, 1, 1e-10, 200);
    solver.pc_mode = PcaPcMode::Right;
    let stats = solver
        .solve(
            aop.as_ref(),
            Some(&pc),
            &b,
            &mut x,
            PcSide::Right,
            &UniverseComm::NoComm(kryst::parallel::NoComm),
            None,
            None,
        )
        .unwrap();
    assert!(stats.final_residual <= 1e-6, "res={}", stats.final_residual);
    for (xi, &xt) in x.iter().zip(x_true.iter()) {
        assert!((xi - xt).abs() <= 1e-6, "xi = {}, expected = {}", xi, xt);
    }
}

