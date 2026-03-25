#![cfg(not(feature = "complex"))]
use kryst::algebra::scalar::{KrystScalar, S};
use kryst::matrix::dist_csr::DistCsrOp;
use kryst::matrix::op::LinOp;
use kryst::matrix::sparse::CsrMatrix;
use kryst::parallel::{NoComm, UniverseComm};

#[test]
#[should_panic]
fn dist_matvec_panics_on_bad_dims() {
    // Minimal 1x1 operator with serial communicator
    let local = CsrMatrix::<S>::from_csr(1, 1, vec![0, 1], vec![0], vec![S::from_real(1.0)]);
    let part = vec![0, 1]; // single-rank partition
    let comm = UniverseComm::NoComm(NoComm);
    let op = DistCsrOp::from_local_rows(1, 0, &local, &part, comm).unwrap();

    let x = vec![S::from_real(1.0), S::from_real(2.0)];
    let mut y = vec![S::zero()];
    // Should panic (previously: silently ignored the Err)
    op.matvec(&x, &mut y);
}

#[test]
fn dist_try_matvec_returns_error_on_bad_dims() {
    let local = CsrMatrix::<S>::from_csr(1, 1, vec![0, 1], vec![0], vec![S::from_real(1.0)]);
    let part = vec![0, 1];
    let comm = UniverseComm::NoComm(NoComm);
    let op = DistCsrOp::from_local_rows(1, 0, &local, &part, comm).unwrap();

    let x = vec![S::from_real(1.0), S::from_real(2.0)];
    let mut y = vec![S::zero()];

    let err = op.try_matvec(&x, &mut y).unwrap_err();
    match err {
        kryst::error::KError::InvalidInput(msg) => {
            assert!(msg.to_lowercase().contains("dimension"))
        }
        other => panic!("unexpected error: {:?}", other),
    }
}

#[cfg(feature = "rayon")]
#[test]
fn dist_try_matvec_allows_concurrent_calls() {
    use rayon::prelude::*;

    let local = CsrMatrix::<S>::from_csr(
        3,
        3,
        vec![0, 2, 4, 6],
        vec![0, 1, 0, 2, 1, 2],
        vec![
            S::from_real(2.0),
            S::from_real(-1.0),
            S::from_real(-1.0),
            S::from_real(3.0),
            S::from_real(4.0),
            S::from_real(1.5),
        ],
    );
    let part = vec![0, 3];
    let comm = UniverseComm::NoComm(NoComm);
    let op = DistCsrOp::from_local_rows(3, 0, &local, &part, comm).unwrap();

    let rhs: Vec<Vec<S>> = (0..64)
        .map(|k| {
            vec![
                S::from_real(1.0 + k as f64),
                S::from_real(-0.5 * k as f64),
                S::from_real(2.0),
            ]
        })
        .collect();

    rhs.par_iter().for_each(|x| {
        let mut y = vec![S::zero(); 3];
        op.try_matvec(x, &mut y).unwrap();
    });
}
