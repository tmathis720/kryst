#![cfg(all(feature = "backend-faer", not(feature = "complex")))]
use kryst::algebra::prelude::*;
use kryst::context::ksp_context::{GmresSpec, Workspace};
use kryst::parallel::UniverseComm;
use kryst::preconditioner::PcSide;
use kryst::solver::GmresSolver;

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
    let a = faer::Mat::<f64>::from_fn(
        n,
        n,
        |i, j| {
            if i == j { R::from(1.0) } else { R::default() }
        },
    );
    let mut solver = GmresSolver::new(restart, 1e-6, 10);
    let b: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
    let mut x = vec![0.0f64; n];

    solver
        .solve_f64(
            &a,
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

#[test]
fn ensure_block_reuses_buffer() {
    let mut ws = Workspace::default();
    ws.ensure_block(6, 4);
    let buf = ws.block_buf.as_ref().expect("block vec allocated");
    assert_eq!(buf.nrows(), 6);
    assert!(buf.ncols() >= 4);
    let base_ptr = buf.as_slice().as_ptr();

    // Request fewer columns: should reuse existing allocation.
    ws.ensure_block(6, 2);
    let buf_small = ws.block_buf.as_ref().unwrap();
    assert_eq!(buf_small.as_slice().as_ptr(), base_ptr);

    // Request more columns: should grow allocation.
    ws.ensure_block(6, 8);
    let buf_large = ws.block_buf.as_ref().unwrap();
    assert!(buf_large.ncols() >= 8);
    assert!(buf_large.as_slice().as_ptr() != base_ptr);
}

#[test]
fn ensure_tsqr_grows_monotonically() {
    let mut ws = Workspace::default();
    ws.ensure_tsqr(3);
    let tsqr = ws.tsqr.as_ref().expect("tsqr allocated");
    assert_eq!(tsqr.w_max, 3);
    assert_eq!(tsqr.taus.len(), 3);
    assert_eq!(tsqr.rmat.len(), 9);

    // Smaller request should keep existing allocation.
    ws.ensure_tsqr(2);
    let tsqr_small = ws.tsqr.as_ref().unwrap();
    assert_eq!(tsqr_small.w_max, 3);

    // Larger request should grow buffers.
    ws.ensure_tsqr(5);
    let tsqr_large = ws.tsqr.as_ref().unwrap();
    assert_eq!(tsqr_large.w_max, 5);
    assert_eq!(tsqr_large.taus.len(), 5);
    assert_eq!(tsqr_large.rmat.len(), 25);
}
