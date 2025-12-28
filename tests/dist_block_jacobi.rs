#![cfg(all(feature = "backend-faer", not(feature = "complex")))]

use faer::Mat;
use kryst::matrix::dist_csr::DistCsrOp;
use kryst::matrix::sparse::CsrMatrix;
use kryst::parallel::{NoComm, UniverseComm};
use kryst::preconditioner::PcSide;
use kryst::preconditioner::dist::{
    DistVec, GlobalPcKind, LocalPcKind, MpiPcOptions, build_block_jacobi_pc,
};
use kryst::preconditioner::ilu::{Ilu, IluConfig};
use kryst::preconditioner::legacy::Preconditioner;

#[test]
fn distributed_block_jacobi_ilu_matches_serial() {
    let mut mat = Mat::zeros(3, 3);
    mat[(0, 0)] = 4.0;
    mat[(0, 1)] = -1.0;
    mat[(1, 0)] = -1.0;
    mat[(1, 1)] = 4.0;
    mat[(1, 2)] = -1.0;
    mat[(2, 1)] = -1.0;
    mat[(2, 2)] = 4.0;

    let mut ilu = Ilu::new();
    let mut serial_out = vec![0.0; 3];
    let rhs = vec![1.0, 2.0, 3.0];
    ilu.setup(&mat).expect("serial ILU setup");
    ilu.apply(PcSide::Left, &rhs, &mut serial_out)
        .expect("serial ILU apply");

    let csr = CsrMatrix::from_dense(&mat, 0.0).unwrap();
    let part = vec![0, 3];
    let comm = UniverseComm::NoComm(NoComm);
    let dist_op = DistCsrOp::from_local_rows(3, 0, &csr, &part, comm.clone()).unwrap();

    let mut mpi_opts = MpiPcOptions::default();
    mpi_opts.global_pc = GlobalPcKind::BlockJacobi;
    mpi_opts.local_pc = LocalPcKind::Ilu;
    mpi_opts.ilu_config = IluConfig::default();

    let pc = build_block_jacobi_pc(&dist_op, &mpi_opts)
        .unwrap()
        .expect("distributed block jacobi PC");

    let mut dist_vec = DistVec::new(comm, 0, 3, rhs.clone());
    pc.apply_global(PcSide::Left, &mut dist_vec)
        .expect("distributed apply");

    for (serial, local) in serial_out.iter().zip(dist_vec.local_view().iter()) {
        assert!((serial - local).abs() < 1e-10);
    }
}
