use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use kryst::error::KError;
use kryst::matrix::op::LinOp;
use kryst::parallel::UniverseComm;
use kryst::preconditioner::{PcSide, Preconditioner};
use kryst::solver::FgmresSolver;

struct MutableCountPc {
    hits: Arc<AtomicUsize>,
}

impl Preconditioner for MutableCountPc {
    fn setup(&mut self, _a: &dyn LinOp<S = f64>) -> Result<(), KError> {
        Ok(())
    }
    fn apply(&self, _side: PcSide, _x: &[f64], _y: &mut [f64]) -> Result<(), KError> {
        panic!("FGMRES should not call immutable apply");
    }
    fn apply_mut(&mut self, _side: PcSide, x: &[f64], y: &mut [f64]) -> Result<(), KError> {
        self.hits.fetch_add(1, Ordering::Relaxed);
        y.copy_from_slice(x);
        Ok(())
    }
}

#[test]
fn fgmres_uses_apply_mut() {
    let hits = Arc::new(AtomicUsize::new(0));
    let mut pc = MutableCountPc { hits: hits.clone() };

    let a = faer::Mat::<f64>::from_fn(2, 2, |i, j| if i == j { 1.0 } else { 0.0 });
    let amat = &a as &dyn LinOp<S = f64>;
    let mut solver = FgmresSolver::new(1e-6, 10, 2);
    let b = [1.0, 2.0];
    let mut x = [0.0, 0.0];
    solver
        .solve_flexible(
            amat,
            Some(&mut pc),
            &b,
            &mut x,
            PcSide::Right,
            &UniverseComm::NoComm(kryst::parallel::NoComm),
            None,
            None,
        )
        .unwrap();

    assert!(
        hits.load(Ordering::Relaxed) > 0,
        "apply_mut was not invoked"
    );
}
