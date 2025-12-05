#![cfg(feature = "backend-faer")]
use faer::Mat;
use kryst::algebra::prelude::*;
use kryst::context::ksp_context::Workspace;
use kryst::parallel::{NoComm, UniverseComm};
use kryst::preconditioner::PcSide;
use kryst::solver::{CgSolver, LinearSolver};
use std::sync::{Arc, Mutex};

#[test]
fn cg_reports_true_residual() {
    let comm = UniverseComm::NoComm(NoComm);
    let two = 2.0;
    let one = 1.0;
    let a = Mat::<f64>::from_fn(2, 2, |i, j| if i == j { two } else { one });
    let b = vec![one, two];
    let mut x = vec![0.0f64, 0.0];
    let mut solver = CgSolver::new(1e-12, 1);
    let mut wk = Workspace::default();
    solver.setup_workspace(&mut wk);

    let log: Arc<Mutex<Vec<(usize, R)>>> = Arc::new(Mutex::new(Vec::new()));
    let log_clone = log.clone();
    solver.set_true_residual_monitor(Some(Box::new(move |i, r| {
        log_clone.lock().unwrap().push((i, r));
    })));

    solver
        .solve_f64(
            &a,
            None,
            &b,
            &mut x,
            PcSide::Left,
            &comm,
            None,
            Some(&mut wk),
        )
        .unwrap();

    let l = log.lock().unwrap();
    assert!(l.len() >= 2);
    assert_eq!(l[0].0, 0);
    assert!(l[1].1 < l[0].1);
}
