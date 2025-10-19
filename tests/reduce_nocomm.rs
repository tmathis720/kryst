use kryst::algebra::prelude::*;
use kryst::parallel::{NoComm, UniverseComm, global_dot_conj, global_nrm2, global_reduce_tuple2};

#[test]
fn dot_conj_real() {
    let x = vec![S::from_real(1.0), S::from_real(2.0), S::from_real(3.0)];
    let y = vec![S::from_real(4.0), S::from_real(5.0), S::from_real(6.0)];
    let comm = UniverseComm::NoComm(NoComm);
    let result = global_dot_conj(&comm, &x, &y);
    assert!((result.real() - 32.0).abs() < 1e-14);
    #[cfg(feature = "complex")]
    assert!(result.imag().abs() < 1e-14);
}

#[cfg(feature = "complex")]
#[test]
fn dot_conj_complex() {
    let x = vec![S::from_parts(1.0, 1.0), S::from_parts(1.0, -1.0)];
    let y = vec![S::from_parts(2.0, 0.0), S::from_parts(0.0, 2.0)];
    let comm = UniverseComm::NoComm(NoComm);
    let result = global_dot_conj(&comm, &x, &y);
    assert!(result.abs() < 1e-14);
}

#[test]
fn async_scalar_nocomm() {
    let comm = UniverseComm::NoComm(NoComm);
    let mut out = S::zero();
    let req = comm.iallreduce_sum_scalar(S::from_real(3.0), &mut out);
    req.wait();
    assert_eq!(out, S::from_real(3.0));
}

#[test]
fn tuple2_reduction_nocomm() {
    let comm = UniverseComm::NoComm(NoComm);
    let (a, b) = global_reduce_tuple2(&comm, S::from_real(1.5), 2.5);
    assert_eq!(a, S::from_real(1.5));
    assert_eq!(b, 2.5);
}

#[test]
fn norm_real_nocomm() {
    let x = vec![S::from_real(3.0), S::from_real(4.0)];
    let comm = UniverseComm::NoComm(NoComm);
    let norm = global_nrm2(&comm, &x);
    assert!((norm - 5.0).abs() < 1e-12);
}
