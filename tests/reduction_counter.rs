mod support;
use support::reduce_counter::CountingComm;
use kryst::parallel::{Comm, NoComm, UniverseComm};
use std::sync::atomic::Ordering;

#[test]
fn counting_comm_counts_calls() {
    let base = UniverseComm::NoComm(NoComm);
    let comm = CountingComm::new(base);
    let (a,b) = comm.allreduce_sum2(1.0, 2.0);
    assert_eq!(a, 1.0);
    assert_eq!(b, 2.0);
    assert_eq!(comm.reduces.load(Ordering::Relaxed), 1);
    let _ = comm.allreduce_sum(3.0);
    assert_eq!(comm.reduces.load(Ordering::Relaxed), 2);
}
