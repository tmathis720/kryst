#![cfg(all(feature = "complex", feature = "mpi"))]

use kryst::Comm;
use kryst::algebra::prelude::*;
use kryst::parallel::{MpiComm, UniverseComm, global_dot_conj};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

fn mpi_test_guard() -> MutexGuard<'static, ()> {
    static GUARD: OnceLock<Mutex<()>> = OnceLock::new();
    GUARD
        .get_or_init(|| Mutex::new(()))
        .lock()
        .expect("mpi_test_guard poisoned")
}

#[test]
fn mpi_complex_dot_conj_matches_expected() {
    let _guard = mpi_test_guard();
    let comm = UniverseComm::Mpi(Arc::new(MpiComm::new()));
    if comm.size() < 2 {
        return;
    }

    let rank = comm.rank();
    let x_local = [S::from_parts(rank as f64 + 1.0, 0.25 * rank as f64)];
    let y_local = [S::from_parts(0.5, -1.0)];

    let dot = global_dot_conj(&comm, &x_local, &y_local);

    let mut expected = S::zero();
    for r in 0..comm.size() {
        let xr = S::from_parts(r as f64 + 1.0, 0.25 * r as f64);
        expected = expected + xr.conj() * y_local[0];
    }

    let tol = 1e-12 * (comm.size() as f64);
    assert!(
        (dot - expected).abs() < tol,
        "dot={:?} expected={:?}",
        dot,
        expected
    );
}
