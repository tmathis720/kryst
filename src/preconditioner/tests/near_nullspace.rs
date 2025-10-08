use crate::algebra::prelude::*;
use crate::matrix::sparse::CsrMatrix;
use crate::preconditioner::amg::prolong::{TentativeP, smooth_tentative_sa_multi};

use crate::assert_s_close;

#[test]
fn nns_tentative_reproduces_basis() {
    let n = 4;
    let a = CsrMatrix::identity(n);
    let one = S::from_real(1.0).real();
    let zero = R::default();
    let d_inv = vec![one; n];
    let agg = vec![0, 0, 1, 1];
    let t0 = vec![one; n];
    let t1 = vec![zero, one, zero, one];
    let tp = TentativeP {
        agg_of: agg.clone(),
        n_coarse: 2,
        num_functions: 2,
        nns: Some(vec![t0.clone(), t1.clone()]),
        comp_of: None,
    };
    let p = smooth_tentative_sa_multi(&a, &d_inv, &tp, zero, zero, 0, zero);
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
                    assert_s_close!(
                        "tentative prolongator entry",
                        S::from_real(p.vals[k]),
                        S::from_real(expected)
                    );
                    found = true;
                }
            }
            assert!(found);
        }
    }
}
