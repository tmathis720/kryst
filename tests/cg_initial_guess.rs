use kryst::algebra::prelude::*;
use kryst::context::ksp_context::Workspace;
use kryst::matrix::op::{LinOp, LinOpF64};
use kryst::parallel::{NoComm, UniverseComm};
use kryst::preconditioner::PcSide;
use kryst::solver::{CgSolver, LinearSolver};
use std::any::Any;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

struct CountingOp {
    count: Arc<AtomicUsize>,
}
impl CountingOp {
    fn new(count: Arc<AtomicUsize>) -> Self {
        Self { count }
    }
}
impl LinOp for CountingOp {
    type S = f64;
    fn dims(&self) -> (usize, usize) {
        (2, 2)
    }
    fn matvec(&self, x: &[f64], y: &mut [f64]) {
        self.count.fetch_add(1, Ordering::SeqCst);
        y[0] = R::from(2.0) * x[0] + x[1];
        y[1] = x[0] + R::from(2.0) * x[1];
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl LinOpF64 for CountingOp {
    #[inline]
    fn dims(&self) -> (usize, usize) {
        <Self as LinOp>::dims(self)
    }

    #[inline]
    fn matvec(&self, x: &[f64], y: &mut [f64]) {
        <Self as LinOp>::matvec(self, x, y)
    }
}

#[test]
fn cg_respects_nonzero_guess_flag() {
    let b = vec![R::from(1.0), R::from(2.0)];
    let comm = UniverseComm::NoComm(NoComm);

    // Default: zero initial guess skips first matvec
    let counter = Arc::new(AtomicUsize::new(0));
    let op = CountingOp::new(counter.clone());
    let mut x = vec![R::default(); 2];
    let mut solver = CgSolver::new(1e-12, 1);
    let mut wk = Workspace::default();
    solver.setup_workspace(&mut wk);
    solver
        .solve_with_comm(
            &op,
            None,
            &b,
            &mut x,
            PcSide::Left,
            &comm,
            None,
            Some(&mut wk),
        )
        .unwrap();
    let calls_default = counter.load(Ordering::SeqCst);

    // Force nonzero guess: extra matvec to compute initial residual
    let counter2 = Arc::new(AtomicUsize::new(0));
    let op2 = CountingOp::new(counter2.clone());
    let mut x2 = vec![R::default(); 2];
    let mut solver2 = CgSolver::new(1e-12, 1).with_nonzero_guess(true);
    let mut wk2 = Workspace::default();
    solver2.setup_workspace(&mut wk2);
    solver2
        .solve_with_comm(
            &op2,
            None,
            &b,
            &mut x2,
            PcSide::Left,
            &comm,
            None,
            Some(&mut wk2),
        )
        .unwrap();
    let calls_nonzero = counter2.load(Ordering::SeqCst);
    assert_eq!(calls_nonzero, calls_default + 1);
}
