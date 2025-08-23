use std::any::Any;
use std::sync::Arc;

use kryst::context::ksp_context::{KspContext, SolverType};
use kryst::matrix::op::LinOp;

struct DenseOp {
    a: Vec<Vec<f64>>,
}

impl LinOp for DenseOp {
    type S = f64;

    fn dims(&self) -> (usize, usize) { (self.a.len(), self.a[0].len()) }

    fn matvec(&self, x: &[f64], y: &mut [f64]) {
        let (m, n) = self.dims();
        assert_eq!(x.len(), n);
        assert_eq!(y.len(), m);
        for i in 0..m {
            let mut acc = 0.0;
            for j in 0..n {
                acc += self.a[i][j] * x[j];
            }
            y[i] = acc;
        }
    }

    fn as_any(&self) -> &dyn Any { self }
}

#[test]
fn cgs_solves_dd_nonsymmetric() {
    // A 5x5 diagonally dominant non-symmetric matrix (well-conditioned)
    let a = DenseOp {
        a: vec![
            vec![10.0, 2.0, 0.0, 0.0, 0.0],
            vec![3.0, 15.0, 4.0, 0.0, 0.0],
            vec![0.0, -2.0, 8.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0, 7.0, 3.0],
            vec![0.0, 0.0, 0.0, 2.0, 12.0],
        ],
    };
    let (m, n) = a.dims();
    assert_eq!(m, n);
    let x_true = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let mut b = vec![0.0; n];
    a.matvec(&x_true, &mut b);

    let amat: Arc<dyn LinOp<S = f64>> = Arc::new(a);
    let mut ksp = KspContext::new();
    ksp.set_type(SolverType::Cgs).unwrap();
    ksp.set_operators(amat.clone(), Some(amat));

    let mut x = vec![0.0; n];
    let stats = ksp.solve(&b, &mut x).unwrap();

    let tol = 1e-6;
    for (xi, ei) in x.iter().zip(x_true.iter()) {
        assert!((xi - ei).abs() <= tol, "xi = {:.6}, expected = {:.6}", xi, ei);
    }
    assert!(matches!(
        stats.reason,
        kryst::utils::convergence::ConvergedReason::ConvergedRtol
            | kryst::utils::convergence::ConvergedReason::ConvergedAtol
    ));
}

