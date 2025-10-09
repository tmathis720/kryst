use std::any::Any;
use std::sync::Arc;

use kryst::algebra::prelude::*;
use kryst::assert_vec_close;
use kryst::context::ksp_context::{KspContext, SolverType};
use kryst::matrix::op::LinOp;

struct DenseOp {
    a: Vec<Vec<R>>,
}

impl LinOp for DenseOp {
    type S = f64;

    fn dims(&self) -> (usize, usize) {
        (self.a.len(), self.a[0].len())
    }

    fn matvec(&self, x: &[f64], y: &mut [f64]) {
        let (m, n) = self.dims();
        assert_eq!(x.len(), n);
        assert_eq!(y.len(), m);
        for i in 0..m {
            let mut acc = R::default();
            for j in 0..n {
                acc += self.a[i][j] * x[j];
            }
            y[i] = acc;
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[test]
fn cgs_solves_dd_nonsymmetric() {
    // A 5x5 diagonally dominant non-symmetric matrix (well-conditioned)
    let a = DenseOp {
        a: vec![
            vec![
                S::from_real(10.0).real(),
                S::from_real(2.0).real(),
                R::default(),
                R::default(),
                R::default(),
            ],
            vec![
                S::from_real(3.0).real(),
                S::from_real(15.0).real(),
                S::from_real(4.0).real(),
                R::default(),
                R::default(),
            ],
            vec![
                R::default(),
                S::from_real(-2.0).real(),
                S::from_real(8.0).real(),
                S::from_real(1.0).real(),
                R::default(),
            ],
            vec![
                R::default(),
                R::default(),
                S::from_real(1.0).real(),
                S::from_real(7.0).real(),
                S::from_real(3.0).real(),
            ],
            vec![
                R::default(),
                R::default(),
                R::default(),
                S::from_real(2.0).real(),
                S::from_real(12.0).real(),
            ],
        ],
    };
    let (m, n) = a.dims();
    assert_eq!(m, n);
    let x_true = vec![
        S::from_real(1.0).real(),
        S::from_real(2.0).real(),
        S::from_real(3.0).real(),
        S::from_real(4.0).real(),
        S::from_real(5.0).real(),
    ];

    let mut b = vec![R::default(); n];
    a.matvec(&x_true, &mut b);

    let amat: Arc<dyn LinOp<S = f64>> = Arc::new(a);
    let mut ksp = KspContext::new();
    ksp.set_type(SolverType::Cgs).unwrap();
    ksp.set_operators(amat.clone(), Some(amat));

    let mut x = vec![R::default(); n];
    let stats = ksp.solve(&b, &mut x).unwrap();

    let x_s: Vec<S> = x.iter().map(|&v| S::from_real(v)).collect();
    let x_true_s: Vec<S> = x_true.iter().map(|&v| S::from_real(v)).collect();
    assert_vec_close!("cgs solution", &x_s, &x_true_s);
    assert!(matches!(
        stats.reason,
        kryst::utils::convergence::ConvergedReason::ConvergedRtol
            | kryst::utils::convergence::ConvergedReason::ConvergedAtol
    ));
}
