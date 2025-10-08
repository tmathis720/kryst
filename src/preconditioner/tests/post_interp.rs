use crate::algebra::prelude::*;
use crate::preconditioner::amg::{RowScaleMode, row_scaling};

#[test]
fn row_scaling_sum_to_one() {
    let agg = vec![0, 1];
    let r = 1;
    let pr = vec![0, 2, 4];
    let pc = vec![0, 1, 0, 1];
    let mut pv = vec![R::from(0.2), R::from(0.3), R::from(0.4), R::from(0.1)];
    row_scaling(
        RowScaleMode::SumToOne,
        r,
        None,
        &agg,
        None,
        &pr,
        &pc,
        &mut pv,
    )
    .unwrap();
    for i in 0..2 {
        let rs = pr[i];
        let re = pr[i + 1];
        let mut sum = R::default();
        for value in &pv[rs..re] {
            sum += *value;
        }
        assert!((sum - R::from(1.0)).abs() < R::from(1e-12));
    }
}

#[test]
fn row_scaling_to_nns() {
    let agg = vec![0, 0, 1, 1];
    let r = 2;
    let pr = vec![0, 2, 4, 6, 8];
    let pc = vec![0, 1, 0, 1, 2, 3, 2, 3];
    let mut pv = vec![
        R::from(0.2),
        R::from(0.4),
        R::from(0.3),
        R::from(0.3),
        R::from(0.7),
        R::from(0.2),
        R::from(0.1),
        R::from(0.8),
    ];
    let t0 = vec![R::from(1.0); 4];
    let t1 = vec![R::default(), R::from(1.0), R::default(), R::from(1.0)];
    let nns_vec = vec![t0, t1];
    let nns_refs: Vec<&[R]> = nns_vec.iter().map(|v| v.as_slice()).collect();
    row_scaling(
        RowScaleMode::ToNearNullspace,
        r,
        Some(&nns_refs),
        &agg,
        None,
        &pr,
        &pc,
        &mut pv,
    )
    .unwrap();
    for i in 0..4 {
        let rs = pr[i];
        let re = pr[i + 1];
        for alpha in 0..r {
            let mut sum = R::default();
            for k in rs..re {
                if pc[k] % r == alpha {
                    sum += pv[k];
                }
            }
            assert!((sum - nns_vec[alpha][i]).abs() < R::from(1e-12));
        }
    }
}
