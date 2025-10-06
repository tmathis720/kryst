#![cfg(feature = "mpi")]

use kryst::algebra::blas::dot_conj;
use kryst::algebra::prelude::*;
use kryst::parallel::{
    MpiComm, UniverseComm, allreduce_sum_scalar_mpi_sys, allreduce_sum_scalar_slice_in_place,
    allreduce_sum_scalar_slice_owned, global_dot_conj, global_dot_conj_accurate,
    global_dot_conj_many, global_dot_conj_many_accurate, global_dot_conj_many_into,
    global_dot_conj_many_into_accurate, global_dot_conj_many_into_repro,
    global_dot_conj_many_repro, global_dot_conj_repro, global_nrm2, global_nrm2_accurate,
    global_nrm2_many, global_nrm2_many_accurate, global_nrm2_many_into,
    global_nrm2_many_into_accurate, global_nrm2_many_into_repro, global_nrm2_many_repro,
    global_nrm2_repro,
};
use kryst::utils::reduction::{AllreduceOps, ReductMode, ReductOptions};
use std::sync::Arc;

fn make_world() -> UniverseComm {
    UniverseComm::Mpi(Arc::new(MpiComm::new()))
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

#[test]
fn allreduce_scalar_matches_closed_form() {
    let comm = make_world();
    let rank = comm.rank();
    let size = comm.size();

    let local = local_scalar(rank);
    let reduced = comm.allreduce_sum_scalar(local);

    let mut expected_re = 0.0;
    let mut expected_im = 0.0;
    for r in 0..size {
        let value = local_scalar(r);
        expected_re += value.real();
        expected_im += value.imag();
    }

    assert!((reduced.real() - expected_re).abs() < 1e-12 * size as f64);
    assert!((reduced.imag() - expected_im).abs() < 1e-12 * size as f64);
}

#[test]
fn mpi_sys_scalar_matches_safe_path() {
    let comm = make_world();

    if comm.size() <= 1 {
        // No collective exchange occurs in serial mode, so both helpers are identity maps.
        return;
    }

    let rank = comm.rank();
    let local = local_scalar(rank);

    let safe = comm.allreduce_sum_scalar(local);
    let raw = allreduce_sum_scalar_mpi_sys(&comm, local);

    assert!((safe.real() - raw.real()).abs() < 1e-12 * comm.size() as f64);
    #[cfg(feature = "complex")]
    assert!((safe.imag() - raw.imag()).abs() < 1e-12 * comm.size() as f64);
}

#[test]
fn allreduce_scalar_accurate_matches_safe_path() {
    let comm = make_world();
    let rank = comm.rank();

    let local = local_scalar(rank);
    let fast = comm.allreduce_sum_scalar(local);
    let accurate = comm.allreduce_sum_scalar_accurate(local);

    assert_eq!(fast, accurate);
}

#[test]
fn global_dot_conj_matches_manual_sum() {
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

    assert!((dot.real() - expected.real()).abs() < 1e-12 * size as f64);
    assert!((dot.imag() - expected.imag()).abs() < 1e-12 * size as f64);
}

#[test]
fn global_dot_conj_accurate_matches_manual_sum() {
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

    assert!((dot.real() - expected.real()).abs() < 1e-12 * size as f64);
    assert!((dot.imag() - expected.imag()).abs() < 1e-12 * size as f64);
}

#[test]
fn global_dot_conj_repro_matches_fast() {
    let comm = make_world();
    let rank = comm.rank();
    let (x_local, y_local) = local_vectors(rank);

    let fast = global_dot_conj(&comm, &x_local, &y_local);
    let repro = global_dot_conj_repro(&comm, &x_local, &y_local);

    assert_eq!(fast, repro);
}

#[test]
fn global_dot_conj_many_matches_individual_calls() {
    let comm = make_world();
    let rank = comm.rank();
    let size = comm.size();

    let (x_local, y_local) = local_vectors(rank);
    let slice = local_slice(rank);
    let pairs = vec![(&x_local[..], &y_local[..]), (&slice[..1], &slice[..1])];

    let bundled = global_dot_conj_many(&comm, &pairs);
    let repro = global_dot_conj_many_repro(&comm, &pairs);

    assert_eq!(bundled.len(), pairs.len());
    assert_eq!(bundled, repro);

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

    for (g, e) in bundled.iter().zip(expected.iter()) {
        assert!((g.real() - e.real()).abs() < 1e-12 * size as f64);
        assert!((g.imag() - e.imag()).abs() < 1e-12 * size as f64);
    }
}

#[test]
fn global_dot_conj_many_accurate_matches_individual_calls() {
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

    for (g, e) in bundled.iter().zip(expected.iter()) {
        assert!((g.real() - e.real()).abs() < 1e-12 * size as f64);
        assert!((g.imag() - e.imag()).abs() < 1e-12 * size as f64);
    }
}

#[test]
fn global_dot_conj_many_into_matches_owned_helpers() {
    let comm = make_world();
    let rank = comm.rank();
    let size = comm.size();

    let (x_local, y_local) = local_vectors(rank);
    let slice = local_slice(rank);
    let pairs = vec![(&x_local[..], &y_local[..]), (&slice[..2], &slice[..2])];

    let mut into = vec![S::zero(); pairs.len()];
    global_dot_conj_many_into(&comm, &pairs, &mut into);

    let owned = global_dot_conj_many(&comm, &pairs);
    assert_eq!(into, owned);

    let mut accurate = vec![S::zero(); pairs.len()];
    global_dot_conj_many_into_accurate(&comm, &pairs, &mut accurate);
    assert_eq!(into, accurate);

    let mut repro = vec![S::zero(); pairs.len()];
    global_dot_conj_many_into_repro(&comm, &pairs, &mut repro);
    assert_eq!(into, repro);

    let mut manual = vec![S::zero(); pairs.len()];
    for r in 0..size {
        let (vx, vy) = local_vectors(r);
        manual[0] = manual[0] + dot_conj(&vx, &vy);

        let sl = local_slice(r);
        manual[1] = manual[1] + dot_conj(&sl[..2], &sl[..2]);
    }

    for (result, expected) in into.iter().zip(manual.iter()) {
        assert!((result.real() - expected.real()).abs() < 1e-12 * size as f64);
        assert!((result.imag() - expected.imag()).abs() < 1e-12 * size as f64);
    }
}

#[test]
fn global_nrm2_many_matches_individual_calls() {
    let comm = make_world();
    let rank = comm.rank();

    let (x_local, _) = local_vectors(rank);
    let slice = local_slice(rank);
    let local_refs = vec![&x_local[..], &slice[..2]];

    let bundled = global_nrm2_many(&comm, &local_refs);
    assert_eq!(bundled.len(), local_refs.len());

    let single0 = global_nrm2(&comm, &x_local);
    let single1 = global_nrm2(&comm, &slice[..2]);

    assert!((bundled[0] - single0).abs() < 1e-13);
    assert!((bundled[1] - single1).abs() < 1e-13);

    let repro = global_nrm2_many_repro(&comm, &local_refs);
    assert_eq!(bundled, repro);

    let accurate = global_nrm2_many_accurate(&comm, &local_refs);
    assert_eq!(bundled, accurate);
}

#[test]
fn global_nrm2_many_into_matches_owned_helpers() {
    let comm = make_world();
    let rank = comm.rank();

    let (x_local, _) = local_vectors(rank);
    let slice = local_slice(rank);
    let local_refs = vec![&x_local[..], &slice[..2]];

    let mut into = vec![0.0; local_refs.len()];
    global_nrm2_many_into(&comm, &local_refs, &mut into);

    let owned = global_nrm2_many(&comm, &local_refs);
    assert_eq!(into, owned);

    let mut repro = vec![0.0; local_refs.len()];
    global_nrm2_many_into_repro(&comm, &local_refs, &mut repro);
    assert_eq!(into, repro);

    let mut accurate = vec![0.0; local_refs.len()];
    global_nrm2_many_into_accurate(&comm, &local_refs, &mut accurate);
    assert_eq!(into, accurate);
}

#[test]
fn global_nrm2_matches_manual_norm() {
    let comm = make_world();
    let rank = comm.rank();
    let size = comm.size();

    let values = local_slice(rank);
    let norm = global_nrm2(&comm, &values);

    let mut total_sq = 0.0;
    for r in 0..size {
        for value in local_slice(r) {
            let mag = value.abs();
            total_sq += mag * mag;
        }
    }
    let expected = total_sq.max(0.0).sqrt();

    assert!((norm - expected).abs() < 1e-12 * (size as f64).sqrt());
}

#[test]
fn global_nrm2_repro_matches_fast() {
    let comm = make_world();
    let rank = comm.rank();
    let values = local_slice(rank);

    let fast = global_nrm2(&comm, &values);
    let repro = global_nrm2_repro(&comm, &values);

    assert_eq!(fast, repro);
}

#[test]
fn global_nrm2_accurate_matches_fast() {
    let comm = make_world();
    let rank = comm.rank();
    let values = local_slice(rank);

    let fast = global_nrm2(&comm, &values);
    let accurate = global_nrm2_accurate(&comm, &values);

    assert_eq!(fast, accurate);
}

#[test]
fn allreduce_scalar_slice_in_place_matches_component_sums() {
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

    for (result, target) in local.iter().zip(expected.iter()) {
        assert!((result.real() - target.real()).abs() < 1e-12 * size as f64);
        assert!((result.imag() - target.imag()).abs() < 1e-12 * size as f64);
    }
}

#[test]
fn owned_slice_reduction_matches_component_sums() {
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
    for (result, target) in reduced.iter().zip(expected.iter()) {
        assert!((result.real() - target.real()).abs() < 1e-12 * size as f64);
        assert!((result.imag() - target.imag()).abs() < 1e-12 * size as f64);
    }
}

#[test]
fn mpi_async_pair_supports_deterministic_mode() {
    let comm = make_world();
    let opt = ReductOptions {
        mode: ReductMode::Deterministic,
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

    let mut expected_a = 0.0;
    let mut expected_b = 0.0;
    for r in 0..comm.size() {
        expected_a += r as f64 + 0.5;
        expected_b += -0.75 * r as f64;
    }

    assert!((maybe.0 - expected_a).abs() < 1e-12 * comm.size() as f64);
    assert!((maybe.1 - expected_b).abs() < 1e-12 * comm.size() as f64);
}

#[test]
fn mpi_async_vec_supports_deterministic_mode() {
    let comm = make_world();
    let opt = ReductOptions {
        mode: ReductMode::Deterministic,
        ..Default::default()
    };

    let rank = comm.rank();
    let local = local_slice(rank);
    let expected_local: Vec<f64> = local.iter().map(|z| z.real()).collect();
    let (mut handle, original) = comm
        .allreduce_n_async(local.clone().into_iter().map(|z| z.real()).collect(), &opt)
        .expect("deterministic async vector reduction should succeed");

    assert_eq!(original, expected_local);

    // Expect the handle to be ready immediately.
    let reduced = <UniverseComm as AllreduceOps>::test_vec(&mut handle)
        .expect("deterministic vector handle should be ready");

    let mut expected = vec![0.0f64; reduced.len()];
    for r in 0..comm.size() {
        for (idx, value) in local_slice(r).iter().enumerate() {
            expected[idx] += value.real();
        }
    }

    for (got, want) in reduced.iter().zip(expected.iter()) {
        assert!((*got - *want).abs() < 1e-12 * comm.size() as f64);
    }
}

#[test]
fn mpi_async_pair_supports_deterministic_accurate_mode() {
    let comm = make_world();
    let opt = ReductOptions {
        mode: ReductMode::DeterministicAccurate,
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

    let mut expected_a = 0.0;
    let mut expected_b = 0.0;
    for r in 0..comm.size() {
        expected_a += r as f64 + 0.5;
        expected_b += -0.75 * r as f64;
    }

    assert!((maybe.0 - expected_a).abs() < 1e-12 * comm.size() as f64);
    assert!((maybe.1 - expected_b).abs() < 1e-12 * comm.size() as f64);
}

#[test]
fn mpi_async_vec_supports_deterministic_accurate_mode() {
    let comm = make_world();
    let opt = ReductOptions {
        mode: ReductMode::DeterministicAccurate,
        ..Default::default()
    };

    let rank = comm.rank();
    let local = local_slice(rank);
    let expected_local: Vec<f64> = local.iter().map(|z| z.real()).collect();
    let (mut handle, original) = comm
        .allreduce_n_async(local.clone().into_iter().map(|z| z.real()).collect(), &opt)
        .expect("accurate async vector reduction should succeed");

    assert_eq!(original, expected_local);

    let reduced = <UniverseComm as AllreduceOps>::test_vec(&mut handle)
        .expect("accurate vector handle should be ready");

    let mut expected = vec![0.0f64; reduced.len()];
    for r in 0..comm.size() {
        for (idx, value) in local_slice(r).iter().enumerate() {
            expected[idx] += value.real();
        }
    }

    for (observed, target) in reduced.iter().zip(expected.iter()) {
        assert!((observed - target).abs() < 1e-12 * comm.size() as f64);
    }
}
