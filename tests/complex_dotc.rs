#![cfg(feature = "complex")]

use approx::assert_abs_diff_eq;
use kryst::algebra::blas::dot_conj;
use kryst::algebra::prelude::*;
use kryst::parallel::{NoComm, UniverseComm, global_dot_conj};

#[test]
fn dotc_conjugates_inputs() {
    let x = [S::from_parts(1.0, 2.0), S::from_parts(-0.5, 1.0)];
    let y = [S::from_parts(0.5, -1.0), S::from_parts(2.0, 0.25)];

    let expected = x[0].conj() * y[0] + x[1].conj() * y[1];
    let local = dot_conj(&x, &y);

    assert_abs_diff_eq!(local.real(), expected.real(), epsilon = 1e-12);
    assert_abs_diff_eq!(local.imag(), expected.imag(), epsilon = 1e-12);

    let comm = UniverseComm::NoComm(NoComm);
    let global = global_dot_conj(&comm, &x, &y);
    assert_abs_diff_eq!(global.real(), expected.real(), epsilon = 1e-12);
    assert_abs_diff_eq!(global.imag(), expected.imag(), epsilon = 1e-12);
}
