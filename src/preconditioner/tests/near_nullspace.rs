use super::*;
use crate::preconditioner::amg::prolong::{smooth_tentative_sa_multi, TentativeP};
use crate::matrix::sparse::CsrMatrix;

#[test]
fn nns_tentative_reproduces_basis() {
    let n = 4;
    let a = CsrMatrix::identity(n);
    let d_inv = vec![1.0; n];
    let agg = vec![0, 0, 1, 1];
    let t0 = vec![1.0; n];
    let t1 = vec![0.0, 1.0, 0.0, 1.0];
    let tp = TentativeP {
        agg_of: agg.clone(),
        n_coarse: 2,
        num_functions: 2,
        nns: Some(vec![t0.clone(), t1.clone()]),
        comp_of: None,
    };
    let p = smooth_tentative_sa_multi(&a, &d_inv, &tp, 0.0, 0.0, 0, 0.0);
    assert_eq!(p.n, 4);
    for i in 0..n {
        let g = agg[i];
        let rs = p.row_ptr[i];
        let re = p.row_ptr[i + 1];
        for alpha in 0..2 {
            let col = g * 2 + alpha;
            let mut found = false;
            for k in rs..re {
                if p.col_idx[k] == col {
                    let expected = if alpha == 0 { t0[i] } else { t1[i] };
                    assert!((p.vals[k] - expected).abs() < 1e-12);
                    found = true;
                }
            }
            assert!(found);
        }
    }
}
