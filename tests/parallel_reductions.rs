#![cfg(not(feature = "complex"))]
#![cfg(feature = "mpi")]

use kryst::algebra::blas::dot_conj;
use kryst::algebra::prelude::*;
use kryst::matrix::dist_csr::DistCsrOp;
use kryst::matrix::op::LinOp;
use kryst::matrix::sparse::CsrMatrix;
use kryst::parallel::{
    Comm, MpiComm, UniverseComm, allreduce_sum_scalar_mpi_sys, allreduce_sum_scalar_slice_in_place,
    allreduce_sum_scalar_slice_owned, global_dot_conj, global_dot_conj_accurate,
    global_dot_conj_many, global_dot_conj_many_accurate, global_dot_conj_many_into,
    global_dot_conj_many_into_accurate, global_dot_conj_many_into_repro,
    global_dot_conj_many_repro, global_dot_conj_repro, global_nrm2, global_nrm2_accurate,
    global_nrm2_many, global_nrm2_many_accurate, global_nrm2_many_into,
    global_nrm2_many_into_accurate, global_nrm2_many_into_repro, global_nrm2_many_repro,
    global_nrm2_repro,
};
use kryst::reduction::ReproMode;
use kryst::utils::reduction::{AllreduceOps, ReductOptions};
use kryst::{assert_s_close, assert_vec_close, testkit};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

fn make_world() -> UniverseComm {
    UniverseComm::Mpi(Arc::new(MpiComm::new()))
}

fn mpi_test_guard() -> MutexGuard<'static, ()> {
    static GUARD: OnceLock<Mutex<()>> = OnceLock::new();
    GUARD
        .get_or_init(|| Mutex::new(()))
        .lock()
        .expect("mpi_test_guard poisoned")
}

fn scaled_tol(base: f64, factor: usize) -> R {
    S::from_real(base).real() * (factor as R)
}

fn local_scalar(rank: usize) -> S {
    let re = rank as f64 + 1.0;
    let im = 0.5 * rank as f64;
    S::from_parts(re, im)
}

fn local_vectors(rank: usize) -> ([S; 2], [S; 2]) {
    let x0 = S::from_parts(rank as f64 + 0.25, 0.5 * rank as f64);
    let x1 = S::from_parts(-0.75 + 0.1 * rank as f64, -0.25 * rank as f64);
    let y0 = S::from_parts(1.25, -0.75);
    let y1 = S::from_parts(-0.5, 0.5);
    ([x0, x1], [y0, y1])
}

fn local_slice(rank: usize) -> Vec<S> {
    vec![
        S::from_parts(rank as f64 + 1.0, 0.25 * rank as f64),
        S::from_parts(rank as f64 + 2.0, -0.4 * rank as f64),
        S::from_parts(0.5 * rank as f64, 0.1 * (rank + 1) as f64),
    ]
}

#[cfg(feature = "rayon")]
#[test]
fn dist_matvec_mpi_rayon_stress() {
    use rayon::prelude::*;

    let _guard = mpi_test_guard();
    let comm = make_world();
    let rank = comm.rank();
    let size = comm.size();
    let local_n = 2usize;
    let n_global = local_n * size;
    let row_start = rank * local_n;

    let next = if rank + 1 < size { rank + 1 } else { 0 };
    let remote_col = next * local_n;
    let mut row0 = [
        (row_start, S::from_real(2.0)),
        (remote_col, S::from_real(-0.25)),
    ];
    row0.sort_by_key(|(col, _)| *col);
    let mut row1 = [
        (row_start + 1, S::from_real(3.0)),
        (remote_col, S::from_real(0.5)),
    ];
    row1.sort_by_key(|(col, _)| *col);

    let local = CsrMatrix::<S>::from_csr(
        local_n,
        n_global,
        vec![0, 2, 4],
        vec![row0[0].0, row0[1].0, row1[0].0, row1[1].0],
        vec![row0[0].1, row0[1].1, row1[0].1, row1[1].1],
    );
    let part: Vec<usize> = (0..=size).map(|r| r * local_n).collect();
    let op = DistCsrOp::from_local_rows(n_global, row_start, &local, &part, comm.clone()).unwrap();

    let rhs: Vec<Vec<S>> = (0..32)
        .map(|k| {
            vec![
                S::from_real(rank as f64 + 1.0 + k as f64 * 0.01),
                S::from_real(0.5 * rank as f64 - k as f64 * 0.02),
            ]
        })
        .collect();

    rhs.par_iter().for_each(|x| {
        let mut y = vec![S::zero(); local_n];
        op.try_matvec(x, &mut y).unwrap();
    });
}

#[test]
fn allreduce_scalar_matches_closed_form() {
    let _guard = mpi_test_guard();
    let comm = make_world();
    let rank = comm.rank();
    let size = comm.size();

    let local = local_scalar(rank);
    let reduced = comm.allreduce_sum_scalar(local);

    let mut expected_re = R::default();
    let mut expected_im = R::default();
    for r in 0..size {
        let value = local_scalar(r);
        expected_re += value.real();
        expected_im += value.imag();
    }

    let tol = scaled_tol(1e-12, size);
    testkit::assert_s_close(
        "allreduce scalar matches closed form",
        reduced,
        S::from_parts(expected_re, expected_im),
        tol,
        testkit::RTOL,
    );
}

#[test]
fn mpi_sys_scalar_matches_safe_path() {
    let _guard = mpi_test_guard();
    let comm = make_world();

    if comm.size() <= 1 {
        // No collective exchange occurs in serial mode, so both helpers are identity maps.
        return;
    }

    let rank = comm.rank();
    let local = local_scalar(rank);

    let safe = comm.allreduce_sum_scalar(local);
    let raw = allreduce_sum_scalar_mpi_sys(&comm, local);

    let tol = scaled_tol(1e-12, comm.size());
    testkit::assert_s_close("mpi sys scalar", safe, raw, tol, testkit::RTOL);
}

#[test]
fn allreduce_scalar_accurate_matches_safe_path() {
    let _guard = mpi_test_guard();
    let comm = make_world();
    let rank = comm.rank();

    let local = local_scalar(rank);
    let fast = comm.allreduce_sum_scalar(local);
    let accurate = comm.allreduce_sum_scalar_accurate(local);

    assert_s_close!(
        "allreduce scalar accurate matches safe path",
        fast,
        accurate
    );
}

#[test]
fn global_dot_conj_matches_manual_sum() {
    let _guard = mpi_test_guard();
    let comm = make_world();
    let rank = comm.rank();
    let size = comm.size();

    let (x_local, y_local) = local_vectors(rank);
    let dot = global_dot_conj(&comm, &x_local, &y_local);

    let mut expected = S::zero();
    for r in 0..size {
        let (x, y) = local_vectors(r);
        expected = expected + dot_conj(&x, &y);
    }

    let tol = scaled_tol(1e-12, size);
    testkit::assert_s_close("global dot conj manual", dot, expected, tol, testkit::RTOL);
}

#[test]
fn global_dot_conj_accurate_matches_manual_sum() {
    let _guard = mpi_test_guard();
    let comm = make_world();
    let rank = comm.rank();
    let size = comm.size();

    let (x_local, y_local) = local_vectors(rank);
    let dot = global_dot_conj_accurate(&comm, &x_local, &y_local);

    let mut expected = S::zero();
    for r in 0..size {
        let (x, y) = local_vectors(r);
        expected = expected + dot_conj(&x, &y);
    }

    let tol = scaled_tol(1e-12, size);
    testkit::assert_s_close(
        "global dot conj accurate manual",
        dot,
        expected,
        tol,
        testkit::RTOL,
    );
}

#[test]
fn global_dot_conj_repro_matches_fast() {
    let _guard = mpi_test_guard();
    let comm = make_world();
    let rank = comm.rank();
    let (x_local, y_local) = local_vectors(rank);

    let fast = global_dot_conj(&comm, &x_local, &y_local);
    let repro = global_dot_conj_repro(&comm, &x_local, &y_local);

    assert_s_close!("global dot conj repro", fast, repro);
}

#[test]
fn global_dot_conj_many_matches_individual_calls() {
    let _guard = mpi_test_guard();
    let comm = make_world();
    let rank = comm.rank();
    let size = comm.size();

    let (x_local, y_local) = local_vectors(rank);
    let slice = local_slice(rank);
    let pairs = vec![(&x_local[..], &y_local[..]), (&slice[..1], &slice[..1])];

    let bundled = global_dot_conj_many(&comm, &pairs);
    let repro = global_dot_conj_many_repro(&comm, &pairs);

    assert_eq!(bundled.len(), pairs.len());
    assert_vec_close!("global dot conj many repro", &bundled, &repro);

    let mut expected = Vec::with_capacity(2);
    let mut accum0 = S::zero();
    let mut accum1 = S::zero();
    for r in 0..size {
        let (vx, vy) = local_vectors(r);
        accum0 = accum0 + dot_conj(&vx, &vy);

        let sl = local_slice(r);
        accum1 = accum1 + dot_conj(&sl[..1], &sl[..1]);
    }
    expected.push(accum0);
    expected.push(accum1);

    for (idx, (g, e)) in bundled.iter().zip(expected.iter()).enumerate() {
        let tol = scaled_tol(1e-12, size);
        let label = format!("global dot conj many pair {idx}");
        testkit::assert_s_close(&label, *g, *e, tol, testkit::RTOL);
    }
}

#[test]
fn global_dot_conj_many_accurate_matches_individual_calls() {
    let _guard = mpi_test_guard();
    let comm = make_world();
    let rank = comm.rank();
    let size = comm.size();

    let (x_local, y_local) = local_vectors(rank);
    let slice = local_slice(rank);
    let pairs = vec![(&x_local[..], &y_local[..]), (&slice[..2], &slice[..2])];

    let bundled = global_dot_conj_many_accurate(&comm, &pairs);
    assert_eq!(bundled.len(), pairs.len());

    let mut expected = Vec::with_capacity(pairs.len());
    let mut accum0 = S::zero();
    let mut accum1 = S::zero();
    for r in 0..size {
        let (vx, vy) = local_vectors(r);
        accum0 = accum0 + dot_conj(&vx, &vy);

        let sl = local_slice(r);
        accum1 = accum1 + dot_conj(&sl[..2], &sl[..2]);
    }
    expected.push(accum0);
    expected.push(accum1);

    for (idx, (g, e)) in bundled.iter().zip(expected.iter()).enumerate() {
        let tol = scaled_tol(1e-12, size);
        let label = format!("global dot conj many accurate pair {idx}");
        testkit::assert_s_close(&label, *g, *e, tol, testkit::RTOL);
    }
}

#[test]
fn global_dot_conj_many_into_matches_owned_helpers() {
    let _guard = mpi_test_guard();
    let comm = make_world();
    let rank = comm.rank();
    let size = comm.size();

    let (x_local, y_local) = local_vectors(rank);
    let slice = local_slice(rank);
    let pairs = vec![(&x_local[..], &y_local[..]), (&slice[..2], &slice[..2])];

    let mut into = vec![S::zero(); pairs.len()];
    global_dot_conj_many_into(&comm, &pairs, &mut into);

    let owned = global_dot_conj_many(&comm, &pairs);
    assert_vec_close!("global dot conj many into vs owned", &into, &owned);

    let mut accurate = vec![S::zero(); pairs.len()];
    global_dot_conj_many_into_accurate(&comm, &pairs, &mut accurate);
    assert_vec_close!("global dot conj many into vs accurate", &into, &accurate);

    let mut repro = vec![S::zero(); pairs.len()];
    global_dot_conj_many_into_repro(&comm, &pairs, &mut repro);
    assert_vec_close!("global dot conj many into vs repro", &into, &repro);

    let mut manual = vec![S::zero(); pairs.len()];
    for r in 0..size {
        let (vx, vy) = local_vectors(r);
        manual[0] = manual[0] + dot_conj(&vx, &vy);

        let sl = local_slice(r);
        manual[1] = manual[1] + dot_conj(&sl[..2], &sl[..2]);
    }

    let tol = scaled_tol(1e-12, size);
    for (idx, (result, expected)) in into.iter().zip(manual.iter()).enumerate() {
        let label = format!("global dot conj many into manual {idx}");
        testkit::assert_s_close(&label, *result, *expected, tol, testkit::RTOL);
    }
}

#[test]
fn global_nrm2_many_matches_individual_calls() {
    let _guard = mpi_test_guard();
    let comm = make_world();
    let rank = comm.rank();

    let (x_local, _) = local_vectors(rank);
    let slice = local_slice(rank);
    let local_refs = vec![&x_local[..], &slice[..2]];

    let bundled = global_nrm2_many(&comm, &local_refs);
    assert_eq!(bundled.len(), local_refs.len());

    let single0 = global_nrm2(&comm, &x_local);
    let single1 = global_nrm2(&comm, &slice[..2]);

    let tol = S::from_real(1e-13).real();
    testkit::assert_s_close(
        "global nrm2 many entry 0",
        S::from_real(bundled[0]),
        S::from_real(single0),
        tol,
        testkit::RTOL,
    );
    testkit::assert_s_close(
        "global nrm2 many entry 1",
        S::from_real(bundled[1]),
        S::from_real(single1),
        tol,
        testkit::RTOL,
    );

    let repro = global_nrm2_many_repro(&comm, &local_refs);
    let accurate = global_nrm2_many_accurate(&comm, &local_refs);

    for (label, other) in [
        ("global nrm2 many repro", repro.as_slice()),
        ("global nrm2 many accurate", accurate.as_slice()),
    ] {
        for (idx, (&lhs, &rhs)) in bundled.iter().zip(other.iter()).enumerate() {
            let tol = S::from_real(1e-13).real();
            let msg = format!("{label} mismatch at {idx}");
            testkit::assert_s_close(
                &msg,
                S::from_real(lhs),
                S::from_real(rhs),
                tol,
                testkit::RTOL,
            );
        }
    }
}

#[test]
fn global_nrm2_many_into_matches_owned_helpers() {
    let _guard = mpi_test_guard();
    let comm = make_world();
    let rank = comm.rank();

    let (x_local, _) = local_vectors(rank);
    let slice = local_slice(rank);
    let local_refs = vec![&x_local[..], &slice[..2]];

    let mut into = vec![R::default(); local_refs.len()];
    global_nrm2_many_into(&comm, &local_refs, &mut into);

    let owned = global_nrm2_many(&comm, &local_refs);
    for (idx, (&lhs, &rhs)) in into.iter().zip(owned.iter()).enumerate() {
        let tol = S::from_real(1e-13).real();
        let msg = format!("global nrm2 many into vs owned mismatch at {idx}");
        testkit::assert_s_close(
            &msg,
            S::from_real(lhs),
            S::from_real(rhs),
            tol,
            testkit::RTOL,
        );
    }

    let mut repro = vec![R::default(); local_refs.len()];
    global_nrm2_many_into_repro(&comm, &local_refs, &mut repro);
    for (idx, (&lhs, &rhs)) in into.iter().zip(repro.iter()).enumerate() {
        let tol = S::from_real(1e-13).real();
        let msg = format!("global nrm2 many into vs repro mismatch at {idx}");
        testkit::assert_s_close(
            &msg,
            S::from_real(lhs),
            S::from_real(rhs),
            tol,
            testkit::RTOL,
        );
    }

    let mut accurate = vec![R::default(); local_refs.len()];
    global_nrm2_many_into_accurate(&comm, &local_refs, &mut accurate);
    for (idx, (&lhs, &rhs)) in into.iter().zip(accurate.iter()).enumerate() {
        let tol = S::from_real(1e-13).real();
        let msg = format!("global nrm2 many into vs accurate mismatch at {idx}");
        testkit::assert_s_close(
            &msg,
            S::from_real(lhs),
            S::from_real(rhs),
            tol,
            testkit::RTOL,
        );
    }
}

#[test]
fn global_nrm2_matches_manual_norm() {
    let _guard = mpi_test_guard();
    let comm = make_world();
    let rank = comm.rank();
    let size = comm.size();

    let values = local_slice(rank);
    let norm = global_nrm2(&comm, &values);

    let mut total_sq = R::default();
    for r in 0..size {
        for value in local_slice(r) {
            let mag = value.abs();
            total_sq += mag * mag;
        }
    }
    let expected = total_sq.max(R::default()).sqrt();

    let tol = S::from_real(1e-12).real() * (size as R).sqrt();
    testkit::assert_s_close(
        "global nrm2 manual",
        S::from_real(norm),
        S::from_real(expected),
        tol,
        testkit::RTOL,
    );
}

#[test]
fn global_nrm2_repro_matches_fast() {
    let _guard = mpi_test_guard();
    let comm = make_world();
    let rank = comm.rank();
    let values = local_slice(rank);

    let fast = global_nrm2(&comm, &values);
    let repro = global_nrm2_repro(&comm, &values);

    assert_s_close!("global nrm2 repro", S::from_real(fast), S::from_real(repro));
}

#[test]
fn global_nrm2_accurate_matches_fast() {
    let _guard = mpi_test_guard();
    let comm = make_world();
    let rank = comm.rank();
    let values = local_slice(rank);

    let fast = global_nrm2(&comm, &values);
    let accurate = global_nrm2_accurate(&comm, &values);

    assert_s_close!(
        "global nrm2 accurate",
        S::from_real(fast),
        S::from_real(accurate)
    );
}

#[test]
fn allreduce_scalar_slice_in_place_matches_component_sums() {
    let _guard = mpi_test_guard();
    let comm = make_world();
    let rank = comm.rank();
    let size = comm.size();

    let mut local = local_slice(rank);
    allreduce_sum_scalar_slice_in_place(&comm, &mut local);

    let mut expected = vec![S::zero(); local.len()];
    for r in 0..size {
        for (slot, value) in expected.iter_mut().zip(local_slice(r)) {
            *slot = *slot + value;
        }
    }

    let tol = scaled_tol(1e-12, size);
    for (idx, (result, target)) in local.iter().zip(expected.iter()).enumerate() {
        let label = format!("in-place scalar slice entry {idx}");
        testkit::assert_s_close(&label, *result, *target, tol, testkit::RTOL);
    }
}

#[test]
fn owned_slice_reduction_matches_component_sums() {
    let _guard = mpi_test_guard();
    let comm = make_world();
    let rank = comm.rank();
    let size = comm.size();

    let local = local_slice(rank);
    let reduced = allreduce_sum_scalar_slice_owned(&comm, &local);

    let mut expected = vec![S::zero(); local.len()];
    for r in 0..size {
        for (slot, value) in expected.iter_mut().zip(local_slice(r)) {
            *slot = *slot + value;
        }
    }

    assert_eq!(reduced.len(), expected.len());
    let tol = scaled_tol(1e-12, size);
    for (idx, (result, target)) in reduced.iter().zip(expected.iter()).enumerate() {
        let label = format!("owned scalar slice entry {idx}");
        testkit::assert_s_close(&label, *result, *target, tol, testkit::RTOL);
    }
}

#[test]
fn mpi_async_pair_supports_deterministic_mode() {
    let _guard = mpi_test_guard();
    let comm = make_world();
    let opt = ReductOptions {
        mode: ReproMode::Deterministic,
        ..Default::default()
    };

    let rank = comm.rank();
    let a = rank as f64 + 0.5;
    let b = -0.75 * rank as f64;
    let mut handle = comm
        .allreduce2_async(a, b, &opt)
        .expect("deterministic async pair reduction should succeed")
        .0;
    let maybe = <UniverseComm as AllreduceOps>::test_pair(&mut handle)
        .expect("deterministic pair handle should be ready");

    let mut expected_a = R::default();
    let mut expected_b = R::default();
    for r in 0..comm.size() {
        expected_a += r as f64 + 0.5;
        expected_b += -0.75 * r as f64;
    }

    let tol = scaled_tol(1e-12, comm.size());
    testkit::assert_s_close(
        "async pair deterministic real",
        S::from_real(maybe.0),
        S::from_real(expected_a),
        tol,
        testkit::RTOL,
    );
    testkit::assert_s_close(
        "async pair deterministic imag",
        S::from_real(maybe.1),
        S::from_real(expected_b),
        tol,
        testkit::RTOL,
    );
}

#[test]
fn mpi_async_vec_supports_deterministic_mode() {
    let _guard = mpi_test_guard();
    let comm = make_world();
    let opt = ReductOptions {
        mode: ReproMode::Deterministic,
        ..Default::default()
    };

    let rank = comm.rank();
    let local = local_slice(rank);
    let expected_local: Vec<R> = local.iter().map(|z| z.real()).collect();
    let real_local: Vec<R> = local.clone().into_iter().map(|z| z.real()).collect();
    let (mut handle, original) = comm
        .allreduce_n_async(real_local, &opt)
        .expect("deterministic async vector reduction should succeed");

    for (idx, (&lhs, &rhs)) in original.iter().zip(expected_local.iter()).enumerate() {
        let tol = S::from_real(1e-12).real();
        let msg = format!("deterministic async vector original mismatch at {idx}");
        testkit::assert_s_close(
            &msg,
            S::from_real(lhs),
            S::from_real(rhs),
            tol,
            testkit::RTOL,
        );
    }

    // Expect the handle to be ready immediately.
    let reduced = <UniverseComm as AllreduceOps>::test_vec(&mut handle)
        .expect("deterministic vector handle should be ready");

    let mut expected = vec![R::default(); reduced.len()];
    for r in 0..comm.size() {
        for (idx, value) in local_slice(r).iter().enumerate() {
            expected[idx] += value.real();
        }
    }

    let tol = scaled_tol(1e-12, comm.size());
    for (idx, (&got, &want)) in reduced.iter().zip(expected.iter()).enumerate() {
        let label = format!("deterministic async vector reduced entry {idx}");
        testkit::assert_s_close(
            &label,
            S::from_real(got),
            S::from_real(want),
            tol,
            testkit::RTOL,
        );
    }
}

#[test]
fn mpi_async_pair_supports_deterministic_accurate_mode() {
    let _guard = mpi_test_guard();
    let comm = make_world();
    let opt = ReductOptions {
        mode: ReproMode::DeterministicAccurate,
        ..Default::default()
    };

    let rank = comm.rank();
    let a = rank as f64 + 0.5;
    let b = -0.75 * rank as f64;
    let mut handle = comm
        .allreduce2_async(a, b, &opt)
        .expect("accurate async pair reduction should succeed")
        .0;
    let maybe = <UniverseComm as AllreduceOps>::test_pair(&mut handle)
        .expect("accurate pair handle should be ready");

    let mut expected_a = R::default();
    let mut expected_b = R::default();
    for r in 0..comm.size() {
        expected_a += r as f64 + 0.5;
        expected_b += -0.75 * r as f64;
    }

    let tol = scaled_tol(1e-12, comm.size());
    testkit::assert_s_close(
        "async pair accurate real",
        S::from_real(maybe.0),
        S::from_real(expected_a),
        tol,
        testkit::RTOL,
    );
    testkit::assert_s_close(
        "async pair accurate imag",
        S::from_real(maybe.1),
        S::from_real(expected_b),
        tol,
        testkit::RTOL,
    );
}

#[test]
fn mpi_async_vec_supports_deterministic_accurate_mode() {
    let _guard = mpi_test_guard();
    let comm = make_world();
    let opt = ReductOptions {
        mode: ReproMode::DeterministicAccurate,
        ..Default::default()
    };

    let rank = comm.rank();
    let local = local_slice(rank);
    let expected_local: Vec<R> = local.iter().map(|z| z.real()).collect();
    let real_local: Vec<R> = local.clone().into_iter().map(|z| z.real()).collect();
    let (mut handle, original) = comm
        .allreduce_n_async(real_local, &opt)
        .expect("accurate async vector reduction should succeed");

    for (idx, (&lhs, &rhs)) in original.iter().zip(expected_local.iter()).enumerate() {
        let tol = S::from_real(1e-12).real();
        let msg = format!("accurate async vector original mismatch at {idx}");
        testkit::assert_s_close(
            &msg,
            S::from_real(lhs),
            S::from_real(rhs),
            tol,
            testkit::RTOL,
        );
    }

    let reduced = <UniverseComm as AllreduceOps>::test_vec(&mut handle)
        .expect("accurate vector handle should be ready");

    let mut expected = vec![R::default(); reduced.len()];
    for r in 0..comm.size() {
        for (idx, value) in local_slice(r).iter().enumerate() {
            expected[idx] += value.real();
        }
    }

    let tol = scaled_tol(1e-12, comm.size());
    for (idx, (&observed, &target)) in reduced.iter().zip(expected.iter()).enumerate() {
        let label = format!("accurate async vector reduced entry {idx}");
        testkit::assert_s_close(
            &label,
            S::from_real(observed),
            S::from_real(target),
            tol,
            testkit::RTOL,
        );
    }
}
