#![cfg(all(feature = "complex", feature = "mpi"))]

use kryst::Comm;
use kryst::algebra::prelude::*;
use kryst::parallel::{MpiComm, UniverseComm, global_dot_conj, global_dot_conj_repro};
use mpi::collective::SystemOperation;
use mpi::traits::CommunicatorCollectives;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::time::Instant;

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

#[test]
fn mpi_complex_dot_many_matches_pairwise() {
    let _guard = mpi_test_guard();
    let comm = UniverseComm::Mpi(Arc::new(MpiComm::new()));
    if comm.size() < 2 {
        return;
    }

    let rank = comm.rank() as f64;
    let x0 = [S::from_parts(rank + 1.0, 0.1 * rank)];
    let y0 = [S::from_parts(0.25, -0.5)];
    let x1 = [S::from_parts(2.0 - rank, -0.2 * rank)];
    let y1 = [S::from_parts(-1.5, 0.75)];

    let many = kryst::parallel::global_dot_conj_many(&comm, &[(&x0, &y0), (&x1, &y1)]);
    let d0 = global_dot_conj(&comm, &x0, &y0);
    let d1 = global_dot_conj(&comm, &x1, &y1);

    let tol = 1e-12 * (comm.size() as f64);
    assert!(
        (many[0] - d0).abs() < tol,
        "many0={:?} d0={:?}",
        many[0],
        d0
    );
    assert!(
        (many[1] - d1).abs() < tol,
        "many1={:?} d1={:?}",
        many[1],
        d1
    );
}

#[test]
fn mpi_complex_scalar_slice_reduction_matches_closed_form() {
    let _guard = mpi_test_guard();
    let comm = Arc::new(MpiComm::new());
    if comm.size() < 2 {
        return;
    }

    let rank = comm.rank() as f64;
    let mut values = vec![
        S::from_parts(rank + 1.0, 0.25 * rank),
        S::from_parts(-0.5 * rank, 1.0 + rank),
        S::from_parts(2.0, -0.75 * rank),
    ];
    comm.allreduce_sum_scalars(values.as_mut_slice());

    let mut expected = vec![S::zero(); values.len()];
    for r in 0..comm.size() {
        let rr = r as f64;
        expected[0] += S::from_parts(rr + 1.0, 0.25 * rr);
        expected[1] += S::from_parts(-0.5 * rr, 1.0 + rr);
        expected[2] += S::from_parts(2.0, -0.75 * rr);
    }

    let tol = 1e-12 * (comm.size() as f64);
    for (got, want) in values.iter().zip(expected.iter()) {
        assert!((*got - *want).abs() < tol, "got={got:?} want={want:?}");
    }
}

#[test]
fn mpi_complex_deterministic_mode_matches_fast_mode() {
    let _guard = mpi_test_guard();
    let comm = Arc::new(MpiComm::new());
    if comm.size() < 2 {
        return;
    }

    let world = UniverseComm::Mpi(comm.clone());
    let rank = world.rank() as f64;
    let x = [S::from_parts(rank + 1.0, -0.2 * rank)];
    let y = [S::from_parts(0.75, 1.25)];

    let fast = global_dot_conj(&world, &x, &y);
    world.set_reproducible(true);
    let repro = global_dot_conj_repro(&world, &x, &y);
    world.set_reproducible(false);

    let tol = 1e-12 * (world.size() as f64);
    assert!((fast - repro).abs() < tol, "fast={fast:?} repro={repro:?}");
}

#[test]
fn mpi_complex_vector_reduction_latency_regression_guard() {
    let _guard = mpi_test_guard();
    let comm = MpiComm::new();
    if comm.size() < 2 {
        return;
    }

    let n = 1024usize;
    let iters = 16usize;
    let rank = comm.rank() as f64;
    let seed: Vec<S> = (0..n)
        .map(|i| S::from_parts(rank + i as f64 * 1e-3, rank * 1e-3 - i as f64 * 1e-4))
        .collect();

    let t_packed = {
        let start = Instant::now();
        for _ in 0..iters {
            let mut tmp = seed.clone();
            comm.allreduce_sum_scalars(tmp.as_mut_slice());
        }
        start.elapsed().as_secs_f64()
    };

    let t_split = {
        let start = Instant::now();
        for _ in 0..iters {
            let mut re: Vec<R> = seed.iter().map(|z| z.real()).collect();
            let mut im: Vec<R> = seed.iter().map(|z| z.imag()).collect();
            let mut re_out = vec![0.0; n];
            let mut im_out = vec![0.0; n];
            comm.world.all_reduce_into(
                re.as_slice(),
                re_out.as_mut_slice(),
                SystemOperation::sum(),
            );
            comm.world.all_reduce_into(
                im.as_slice(),
                im_out.as_mut_slice(),
                SystemOperation::sum(),
            );
            re.clear();
            im.clear();
        }
        start.elapsed().as_secs_f64()
    };

    // Regression guard: single-packed reduction should not be materially slower
    // than the historical split real/imag two-collective baseline.
    assert!(
        t_packed <= t_split * 1.75,
        "packed={t_packed:.6}s split={t_split:.6}s"
    );
}
