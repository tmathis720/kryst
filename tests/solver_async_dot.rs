use kryst::parallel::NoComm;
use kryst::solver::common::{dot2_async, dotn_async, nrm2_async};
use kryst::utils::reduction::{AllreduceOps, ReductOptions};

#[test]
fn async_dot2_matches_blocking() {
    let comm = NoComm;
    let x1 = [1.0, 2.0, 3.0];
    let y1 = [4.0, 5.0, 6.0];
    let x2 = [0.5, 1.5, -2.5];
    let y2 = [2.0, -3.0, 4.0];
    let opts = ReductOptions::default();
    let mut async_pair = dot2_async(&comm, &x1, &y1, &x2, &y2, &opts);
    assert_eq!(
        async_pair.local.0,
        x1.iter().zip(&y1).map(|(a, b)| a * b).sum::<f64>()
    );
    assert_eq!(
        async_pair.local.1,
        x2.iter().zip(&y2).map(|(a, b)| a * b).sum::<f64>()
    );
    assert_eq!(
        NoComm::test_pair(&mut async_pair.handle),
        Some(async_pair.local)
    );
    let global = <NoComm as AllreduceOps>::wait_pair(async_pair.handle);
    assert_eq!(global, async_pair.local);
}

#[test]
fn async_dotn_matches_blocking() {
    let comm = NoComm;
    let v1 = [1.0, -1.0, 2.0];
    let w1 = [3.0, 0.5, -4.0];
    let v2 = [0.0, 2.0, 1.0];
    let w2 = [1.0, 1.0, 1.0];
    let opts = ReductOptions::default();
    let mut async_vec = dotn_async(&comm, &[(&v1[..], &w1[..]), (&v2[..], &w2[..])], &opts);
    let expected0 = v1.iter().zip(&w1).map(|(a, b)| a * b).sum::<f64>();
    let expected1 = v2.iter().zip(&w2).map(|(a, b)| a * b).sum::<f64>();
    assert_eq!(async_vec.local, vec![expected0, expected1]);
    assert_eq!(
        NoComm::test_vec(&mut async_vec.handle),
        Some(async_vec.local.clone())
    );
    let global = <NoComm as AllreduceOps>::wait_vec(async_vec.handle);
    assert_eq!(global, vec![expected0, expected1]);
}

#[test]
fn async_norm_matches_blocking() {
    let comm = NoComm;
    let x = [1.0, 2.0, 2.0];
    let opts = ReductOptions::default();
    let (handle, local) = nrm2_async(&comm, &x, &opts);
    let expected = x.iter().map(|v| v * v).sum::<f64>();
    assert_eq!(local, expected);
    let sumsq = <NoComm as AllreduceOps>::wait_pair(handle);
    assert_eq!(sumsq, (expected, 0.0));
}
