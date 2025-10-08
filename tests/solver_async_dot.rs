use kryst::algebra::blas::dot_conj;
use kryst::algebra::prelude::*;
use kryst::parallel::NoComm;
use kryst::solver::common::{dot2_async, dotn_async, nrm2_async};
use kryst::testkit::{ATOL, is_zero, s};
use kryst::utils::reduction::{AllreduceOps, ReductOptions};
use kryst::{assert_s_close, assert_vec_close};

#[test]
fn async_dot2_matches_blocking() {
    let comm = NoComm;
    let x1 = [R::from(1.0), R::from(2.0), R::from(3.0)];
    let y1 = [R::from(4.0), R::from(5.0), R::from(6.0)];
    let x2 = [R::from(0.5), R::from(1.5), R::from(-2.5)];
    let y2 = [R::from(2.0), R::from(-3.0), R::from(4.0)];
    let xs1: Vec<S> = x1.iter().copied().map(s).collect();
    let ys1: Vec<S> = y1.iter().copied().map(s).collect();
    let xs2: Vec<S> = x2.iter().copied().map(s).collect();
    let ys2: Vec<S> = y2.iter().copied().map(s).collect();
    let opts = ReductOptions::default();
    let mut async_pair = dot2_async(&comm, &x1, &y1, &x2, &y2, &opts);
    let expected0 = dot_conj(&xs1, &ys1);
    let expected1 = dot_conj(&xs2, &ys2);
    assert_s_close!("async dot2 local[0]", expected0, s(async_pair.local.0));
    assert_s_close!("async dot2 local[1]", expected1, s(async_pair.local.1));
    assert_eq!(
        NoComm::test_pair(&mut async_pair.handle),
        Some(async_pair.local)
    );
    let global = <NoComm as AllreduceOps>::wait_pair(async_pair.handle);
    assert_s_close!("async dot2 global[0]", expected0, s(global.0));
    assert_s_close!("async dot2 global[1]", expected1, s(global.1));
}

#[test]
fn async_dotn_matches_blocking() {
    let comm = NoComm;
    let v1 = [R::from(1.0), R::from(-1.0), R::from(2.0)];
    let w1 = [R::from(3.0), R::from(0.5), R::from(-4.0)];
    let v2 = [R::default(), R::from(2.0), R::from(1.0)];
    let w2 = [R::from(1.0), R::from(1.0), R::from(1.0)];
    let v1s: Vec<S> = v1.iter().copied().map(s).collect();
    let w1s: Vec<S> = w1.iter().copied().map(s).collect();
    let v2s: Vec<S> = v2.iter().copied().map(s).collect();
    let w2s: Vec<S> = w2.iter().copied().map(s).collect();
    let opts = ReductOptions::default();
    let mut async_vec = dotn_async(&comm, &[(&v1[..], &w1[..]), (&v2[..], &w2[..])], &opts);
    let expected0 = dot_conj(&v1s, &w1s);
    let expected1 = dot_conj(&v2s, &w2s);
    let local: Vec<S> = async_vec.local.iter().copied().map(s).collect();
    assert_vec_close!("async dotn local", &local, &[expected0, expected1]);
    assert_eq!(
        NoComm::test_vec(&mut async_vec.handle),
        Some(async_vec.local.clone())
    );
    let global = <NoComm as AllreduceOps>::wait_vec(async_vec.handle);
    let global_s: Vec<S> = global.iter().copied().map(s).collect();
    assert_vec_close!("async dotn global", &global_s, &[expected0, expected1]);
}

#[test]
fn async_norm_matches_blocking() {
    let comm = NoComm;
    let x = [R::from(1.0), R::from(2.0), R::from(2.0)];
    let xs: Vec<S> = x.iter().copied().map(s).collect();
    let opts = ReductOptions::default();
    let (handle, local) = nrm2_async(&comm, &x, &opts);
    let expected = dot_conj(&xs, &xs);
    assert_s_close!("async norm local", expected, s(local));
    let sumsq = <NoComm as AllreduceOps>::wait_pair(handle);
    assert_s_close!("async norm global sum", expected, s(sumsq.0));
    assert!(is_zero(s(sumsq.1), ATOL));
}
