use kryst::context::ksp_context::{GmresSpec, Workspace};
use kryst::matrix::op::LinOp;
use kryst::parallel::UniverseComm;
use kryst::preconditioner::PcSide;
use kryst::solver::{GmresSolver, LinearSolver};

#[test]
fn gmres_workspace_allocation_stable_and_sized() {
    let n = 4;
    let restart = 3;
    // allocate workspace manually
    let mut ws = Workspace::default();
    ws.acquire_gmres(GmresSpec {
        n,
        m: restart,
        need_z: true,
        block_s: 0,
    });

    let v_ptr = ws.v_mem.as_ptr();
    let z_ptr = ws.z_mem.as_ptr();
    let h_ptr = ws.h_mem.as_ptr();
    let v_cap = ws.v_mem.capacity();
    let z_cap = ws.z_mem.capacity();
    let h_cap = ws.h_mem.capacity();

    // Setup small identity system
    let a = faer::Mat::<f64>::from_fn(n, n, |i, j| if i == j { 1.0 } else { 0.0 });
    let amat = &a as &dyn LinOp<S = f64>;
    let mut solver = GmresSolver::new(restart, 1e-6, 10);
    let b: Vec<f64> = (0..n).map(|i| i as f64 + 1.0).collect();
    let mut x = vec![0.0; n];

    solver
        .solve(
            amat,
            None,
            &b,
            &mut x,
            PcSide::Right,
            &UniverseComm::NoComm(kryst::parallel::NoComm),
            None,
            Some(&mut ws),
        )
        .unwrap();

    // Pointers and capacities should be unchanged
    assert_eq!(v_ptr, ws.v_mem.as_ptr());
    assert_eq!(z_ptr, ws.z_mem.as_ptr());
    assert_eq!(h_ptr, ws.h_mem.as_ptr());
    assert!(ws.v_mem.capacity() >= v_cap);
    assert!(ws.z_mem.capacity() >= z_cap);
    assert!(ws.h_mem.capacity() >= h_cap);

    // Lengths should match exact expectations
    assert_eq!(ws.v_mem.len(), (restart + 1) * n);
    assert_eq!(ws.z_mem.len(), restart * n);
    assert_eq!(ws.h_mem.len(), (restart + 1) * restart);
    assert_eq!(ws.g.len(), restart + 1);
    assert_eq!(ws.cs.len(), restart);
    assert_eq!(ws.sn.len(), restart);
}
