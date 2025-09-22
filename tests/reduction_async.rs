use kryst::parallel::{Comm, NoComm};
use kryst::utils::reduction::{AllreduceOps, ReductOptions};

#[test]
fn nocomm_allreduce_pair_ready_immediately() {
    let comm = NoComm;
    let opts = ReductOptions::default();
    let (mut handle, local) = comm.allreduce2_async(3.0, 4.0, &opts).unwrap();
    assert_eq!(local, (3.0, 4.0));
    assert_eq!(NoComm::test_pair(&mut handle), Some((3.0, 4.0)));
    assert_eq!(NoComm::wait_pair(handle), (3.0, 4.0));
}

#[test]
fn nocomm_allreduce_vec_ready_immediately() {
    let comm = NoComm;
    let opts = ReductOptions::default();
    let (mut handle, local) = comm.allreduce_n_async(vec![1.0, 2.0, 3.0], &opts).unwrap();
    assert_eq!(local, vec![1.0, 2.0, 3.0]);
    assert_eq!(NoComm::test_vec(&mut handle), Some(vec![1.0, 2.0, 3.0]));
    assert_eq!(NoComm::wait_vec(handle), vec![1.0, 2.0, 3.0]);
}

#[cfg(feature = "rayon")]
#[test]
fn rayon_allreduce_pair_async_completes() {
    let comm = kryst::parallel::rayon_comm::RayonComm::new();
    let opts = ReductOptions::default();
    let (mut handle, local) = comm.allreduce2_async(5.0, 7.0, &opts).unwrap();
    assert_eq!(local, (5.0, 7.0));
    if let Some(res) = kryst::parallel::rayon_comm::RayonComm::test_pair(&mut handle) {
        assert_eq!(res, (5.0, 7.0));
    } else {
        let waited = kryst::parallel::rayon_comm::RayonComm::wait_pair(handle);
        assert_eq!(waited, (5.0, 7.0));
    }
}

#[cfg(feature = "rayon")]
#[test]
fn rayon_allreduce_vec_async_completes() {
    let comm = kryst::parallel::rayon_comm::RayonComm::new();
    let opts = ReductOptions::default();
    let (mut handle, local) = comm
        .allreduce_n_async(vec![1.0, 2.0, 3.0, 4.0], &opts)
        .unwrap();
    assert_eq!(local, vec![1.0, 2.0, 3.0, 4.0]);
    if let Some(res) = kryst::parallel::rayon_comm::RayonComm::test_vec(&mut handle) {
        assert_eq!(res, vec![1.0, 2.0, 3.0, 4.0]);
    } else {
        let waited = kryst::parallel::rayon_comm::RayonComm::wait_vec(handle);
        assert_eq!(waited, vec![1.0, 2.0, 3.0, 4.0]);
    }
}

#[cfg(feature = "mpi")]
#[test]
fn mpi_allreduce_pair_matches_sum() {
    use kryst::parallel::mpi_comm::MpiComm;
    let comm = MpiComm::new();
    let opts = ReductOptions::default();
    let local = (comm.rank() as f64 + 1.0, comm.rank() as f64 + 2.0);
    let (handle, _) = comm.allreduce2_async(local.0, local.1, &opts).unwrap();
    let global = kryst::parallel::mpi_comm::MpiComm::wait_pair(handle);
    let size = comm.size() as f64;
    assert_eq!(global.0, (size * (size + 1.0)) / 2.0);
    assert_eq!(global.1, (size * (size + 3.0)) / 2.0);
}
