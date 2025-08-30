use kryst::LinOp;
use kryst::matrix::sparse::CsrMatrix;
use kryst::matrix::dist_csr::DistCsrOp;
use kryst::parallel::{NoComm, UniverseComm};

#[test]
#[should_panic]
fn dist_matvec_panics_on_bad_dims() {
    // Minimal 1x1 operator with serial communicator
    let local = CsrMatrix::from_csr(1, 1, vec![0, 1], vec![0], vec![1.0]);
    let part = vec![0, 1]; // single-rank partition
    let comm = UniverseComm::NoComm(NoComm);
    let op = DistCsrOp::from_local_rows(1, 0, &local, &part, comm).unwrap();

    let x = vec![1.0, 2.0]; // wrong length
    let mut y = vec![0.0];
    // Should panic (previously: silently ignored the Err)
    op.matvec(&x, &mut y);
}

#[test]
fn dist_try_matvec_returns_error_on_bad_dims() {
    let local = CsrMatrix::from_csr(1, 1, vec![0, 1], vec![0], vec![1.0]);
    let part = vec![0, 1];
    let comm = UniverseComm::NoComm(NoComm);
    let op = DistCsrOp::from_local_rows(1, 0, &local, &part, comm).unwrap();

    let x = vec![1.0, 2.0]; // bad
    let mut y = vec![0.0];

    let err = op.try_matvec(&x, &mut y).unwrap_err();
    match err {
        kryst::KError::InvalidInput(msg) => assert!(msg.to_lowercase().contains("dimension")),
        other => panic!("unexpected error: {:?}", other),
    }
}
