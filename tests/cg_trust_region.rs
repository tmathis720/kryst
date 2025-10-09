use faer::Mat;
use kryst::algebra::prelude::*;
use kryst::context::ksp_context::Workspace;
use kryst::parallel::UniverseComm;
use kryst::preconditioner::PcSide;
use kryst::solver::{CgSolver, LinearSolver};
use kryst::utils::convergence::ConvergedReason;

#[test]
fn cg_hits_trust_region() {
    let comm = UniverseComm::NoComm(kryst::parallel::NoComm);
    let two = S::from_real(2.0).real();
    let zero = S::zero().real();
    let a = Mat::<f64>::from_fn(1, 1, |_i, _j| two);
    let b = vec![two];
    let mut x = vec![zero];
    let mut solver =
        CgSolver::new(S::from_real(1e-8).real(), 5).with_trust_region(S::from_real(0.5).real());
    let mut ws = Workspace::new(1);
    solver.setup_workspace(&mut ws);
    let stats = solver
        .solve_f64(
            &a,
            None,
            &b,
            &mut x,
            PcSide::Left,
            &comm,
            None,
            Some(&mut ws),
        )
        .unwrap();
    assert_eq!(stats.reason, ConvergedReason::ConvergedTrustRegion);
}
