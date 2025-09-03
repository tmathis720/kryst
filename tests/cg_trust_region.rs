use faer::Mat;
use kryst::context::ksp_context::Workspace;
use kryst::parallel::UniverseComm;
use kryst::preconditioner::PcSide;
use kryst::solver::{CgSolver, LinearSolver};
use kryst::utils::convergence::ConvergedReason;

#[test]
fn cg_hits_trust_region() {
    let comm = UniverseComm::NoComm(kryst::parallel::NoComm);
    let a = Mat::<f64>::from_fn(1, 1, |_i, _j| 2.0);
    let b = vec![2.0];
    let mut x = vec![0.0];
    let mut solver = CgSolver::new(1e-8, 5).with_trust_region(0.5);
    let mut ws = Workspace::new(1);
    solver.setup_workspace(&mut ws);
    let stats = solver
        .solve(
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
