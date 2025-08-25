#[cfg(feature = "mpi")]
use kryst::parallel::{Comm, MpiComm, UniverseComm};
#[cfg(feature = "mpi")]
use std::sync::Arc;

#[cfg(feature = "mpi")]
#[test]
fn split_roundtrips() {
    let comm = UniverseComm::Mpi(Arc::new(MpiComm::new()));
    let color = (comm.rank() % 2) as i32;
    let key = comm.rank() as i32;
    let sub = comm.split(color, key);
    assert!(sub.size() >= 1);
}

#[cfg(feature = "mpi")]
#[test]
fn scatter_gather_smoke() {
    let comm = MpiComm::new();
    let rank = comm.rank();
    let size = comm.size();
    let root = 0usize;
    let n = 2usize;
    let mut recv = vec![0i32; n];

    if rank == root {
        let global: Vec<i32> = (0..n * size).map(|x| x as i32).collect();
        comm.scatter(&global, &mut recv, root);
    } else {
        let empty: Vec<i32> = Vec::new();
        comm.scatter(&empty, &mut recv, root);
    }

    let mut gathered = Vec::new();
    comm.gather(&recv, &mut gathered, root);

    if rank == root {
        let global: Vec<i32> = (0..n * size).map(|x| x as i32).collect();
        assert_eq!(gathered, global);
    } else {
        assert!(gathered.is_empty());
    }
}
