use super::*;
use crate::matrix::DistCsrOp;
use crate::matrix::sparse::CsrMatrix;
use crate::parallel::{NoComm, UniverseComm};

#[test]
#[cfg(not(feature = "complex"))]
fn jacobi_setup_with_dist_csr_local_block() {
    let local = CsrMatrix::from_csr(2, 2, vec![0, 1, 2], vec![0, 1], vec![2.0, 4.0]);
    let part_prefix = vec![0, 2];
    let comm = UniverseComm::NoComm(NoComm);
    let dist = DistCsrOp::from_local_rows(2, 0, &local, &part_prefix, comm).expect("dist csr");

    let mut pc = Jacobi::new();
    pc.setup(&dist).expect("jacobi setup");

    let rhs = [S::from_real(2.0), S::from_real(8.0)];
    let mut out = [S::zero(); 2];
    pc.apply(PcSide::Left, &rhs, &mut out)
        .expect("jacobi apply");

    assert!((out[0] - S::from_real(1.0)).abs() < 1e-12);
    assert!((out[1] - S::from_real(2.0)).abs() < 1e-12);
}
