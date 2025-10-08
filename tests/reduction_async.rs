use kryst::algebra::prelude::*;
use kryst::parallel::{Comm, NoComm};
use kryst::utils::reduction::{AllreduceOps, ReductOptions};

#[test]
fn nocomm_allreduce_pair_ready_immediately() {
    let comm = NoComm;
    let opts = ReductOptions::default();
    let three = S::from_real(3.0).real();
    let four = S::from_real(4.0).real();
    let (mut handle, local) = comm.allreduce2_async(three, four, &opts).unwrap();
    assert_eq!(local, (three, four));
    assert_eq!(NoComm::test_pair(&mut handle), Some((three, four)));
    assert_eq!(NoComm::wait_pair(handle), (three, four));
}

#[test]
fn nocomm_allreduce_vec_ready_immediately() {
    let comm = NoComm;
    let opts = ReductOptions::default();
    let one = S::from_real(1.0).real();
    let two = S::from_real(2.0).real();
    let three = S::from_real(3.0).real();
    let expected = vec![one, two, three];
    let (mut handle, local) = comm.allreduce_n_async(expected.clone(), &opts).unwrap();
    assert_eq!(local, expected.clone());
    assert_eq!(NoComm::test_vec(&mut handle), Some(expected.clone()));
    assert_eq!(NoComm::wait_vec(handle), expected);
}

#[cfg(feature = "rayon")]
#[test]
fn rayon_allreduce_pair_async_completes() {
    let comm = kryst::parallel::rayon_comm::RayonComm::new();
    let opts = ReductOptions::default();
    let five = S::from_real(5.0).real();
    let seven = S::from_real(7.0).real();
    let (mut handle, local) = comm.allreduce2_async(five, seven, &opts).unwrap();
    assert_eq!(local, (five, seven));
    if let Some(res) = kryst::parallel::rayon_comm::RayonComm::test_pair(&mut handle) {
        assert_eq!(res, (five, seven));
    } else {
        let waited = kryst::parallel::rayon_comm::RayonComm::wait_pair(handle);
        assert_eq!(waited, (five, seven));
    }
}

#[cfg(feature = "rayon")]
#[test]
fn rayon_allreduce_vec_async_completes() {
    let comm = kryst::parallel::rayon_comm::RayonComm::new();
    let opts = ReductOptions::default();
    let one = S::from_real(1.0).real();
    let two = S::from_real(2.0).real();
    let three = S::from_real(3.0).real();
    let four = S::from_real(4.0).real();
    let expected = vec![one, two, three, four];
    let (mut handle, local) = comm.allreduce_n_async(expected.clone(), &opts).unwrap();
    assert_eq!(local, expected.clone());
    if let Some(res) = kryst::parallel::rayon_comm::RayonComm::test_vec(&mut handle) {
        assert_eq!(res, expected.clone());
    } else {
        let waited = kryst::parallel::rayon_comm::RayonComm::wait_vec(handle);
        assert_eq!(waited, expected);
    }
}

#[cfg(feature = "mpi")]
#[test]
fn mpi_allreduce_pair_matches_sum() {
    use kryst::parallel::mpi_comm::MpiComm;
    let comm = MpiComm::new();
    let opts = ReductOptions::default();
    let base = comm.rank() as f64;
    let local = (
        S::from_real(base + 1.0).real(),
        S::from_real(base + 2.0).real(),
    );
    let (handle, _) = comm.allreduce2_async(local.0, local.1, &opts).unwrap();
    let global = kryst::parallel::mpi_comm::MpiComm::wait_pair(handle);
    let size = comm.size() as f64;
    let size_r = S::from_real(size).real();
    assert_eq!(
        global.0,
        (size_r * S::from_real(size + 1.0).real()) / S::from_real(2.0).real()
    );
    assert_eq!(
        global.1,
        (size_r * S::from_real(size + 3.0).real()) / S::from_real(2.0).real()
    );
}
