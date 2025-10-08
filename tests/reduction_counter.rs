mod support;

use kryst::algebra::prelude::*;
use kryst::parallel::{Comm, NoComm, UniverseComm};
use std::sync::atomic::Ordering;
use support::reduce_counter::CountingComm;

#[test]
fn counting_comm_counts_calls() {
    let base = UniverseComm::NoComm(NoComm);
    let comm = CountingComm::new(base);
    let one = S::from_real(1.0).real();
    let two = S::from_real(2.0).real();
    let three = S::from_real(3.0).real();
    let (a, b) = comm.allreduce_sum2(one, two);
    assert_eq!(a, one);
    assert_eq!(b, two);
    assert_eq!(comm.reduces.load(Ordering::Relaxed), 1);
    let _ = comm.allreduce_sum(three);
    assert_eq!(comm.reduces.load(Ordering::Relaxed), 2);
}
