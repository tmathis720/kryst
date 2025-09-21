use super::*;

fn simple_identity(n: usize) -> CsrMatrix<f64> {
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::with_capacity(n);
    let mut vals = Vec::with_capacity(n);
    row_ptr.push(0);
    for i in 0..n {
        col_idx.push(i);
        vals.push(1.0);
        row_ptr.push(col_idx.len());
    }
    CsrMatrix::from_csr(n, n, row_ptr, col_idx, vals)
}

#[test]
fn powdered_columns_retry() {
    let a = simple_identity(2);
    let diag_inv = vec![1.0, 1.0];
    let tp = TentativeP {
        agg_of: vec![0, 1],
        n_coarse: 2,
        num_functions: 1,
        nns: None,
        comp_of: None,
    };
    let mut p_csr = Pcsr {
        m: 2,
        n: 2,
        row_ptr: vec![0, 1, 2],
        col_idx: vec![0, 1],
        vals: vec![1e-16, 1e-16],
    };
    let ctx = LevelPostContext {
        r: 1,
        agg_of: &tp.agg_of,
        nns: None,
        a: Some(&a),
        d_inv: Some(&diag_inv),
    };
    let p0 = CsrMatrix::from_csr(
        p_csr.m,
        p_csr.n,
        p_csr.row_ptr.clone(),
        p_csr.col_idx.clone(),
        p_csr.vals.clone(),
    );
    let mut cfg = AMGConfig::default();
    cfg.rank_min_col_norm = 1e-12;
    let diag0 = check_p_rank_fast(&p0, &cfg).unwrap();
    assert!(diag0.suspect, "expected suspect rank before fix");

    match try_fix_rank(0, &a, &diag_inv, &tp, &ctx, &mut p_csr, &mut cfg).unwrap() {
        RankFixOutcome::Fixed => {
            let p_fixed = CsrMatrix::from_csr(
                p_csr.m,
                p_csr.n,
                p_csr.row_ptr.clone(),
                p_csr.col_idx.clone(),
                p_csr.vals.clone(),
            );
            let diag1 = check_p_rank_fast(&p_fixed, &cfg).unwrap();
            assert!(diag1.min_col_norm > cfg.rank_min_col_norm);
            assert!(!diag1.suspect);
        }
        RankFixOutcome::Unfixed => panic!("expected retry to succeed"),
    }
}

#[test]
fn sketch_condition_detects_colinear_columns() {
    let row_ptr = vec![0, 2, 4];
    let col_idx = vec![0, 1, 0, 1];
    let vals = vec![1.0, 1.0, 2.0, 2.0];
    let p = CsrMatrix::from_csr(2, 2, row_ptr.clone(), col_idx.clone(), vals);
    let mut cfg = AMGConfig::default();
    cfg.rank_cond_threshold = 10.0;
    let diag = check_p_rank_fast(&p, &cfg).unwrap();
    assert!(diag.suspect, "colinear columns should trigger suspect flag");
    assert!(!diag.cond_estimate.is_finite() || diag.cond_estimate > cfg.rank_cond_threshold);
}

#[test]
fn galerkin_identity_matches() {
    let a = CsrMatrix::from_csr(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![4.0, -1.0, -1.0, 4.0],
    );
    let p = simple_identity(2);
    let r = simple_identity(2);
    let a_c = a.clone();
    let (ok, worst) = galerkin_sample_check(&a, &p, &r, &a_c, 3, 1e-12, 0xBEEF).unwrap();
    assert!(
        ok,
        "Galekin identity should hold for identical hierarchy: worst={worst:e}"
    );
}

#[test]
fn galerkin_identity_detects_mismatch() {
    let a = CsrMatrix::from_csr(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![4.0, -1.0, -1.0, 4.0],
    );
    let p = simple_identity(2);
    let r = simple_identity(2);
    let a_c = CsrMatrix::from_csr(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![4.0, -2.0, -1.0, 4.0],
    );
    let (ok, worst) = galerkin_sample_check(&a, &p, &r, &a_c, 2, 1e-12, 0xC0FF).unwrap();
    assert!(!ok);
    assert!(worst > 0.0);
}

#[test]
fn rank_condition_deterministic() {
    let p = CsrMatrix::from_csr(3, 3, vec![0, 1, 2, 3], vec![0, 1, 2], vec![1.0, 0.5, 0.25]);
    let mut cfg = AMGConfig::default();
    cfg.rank_sketch_cols = 3;
    let (_, c1) = rank_condition_estimate(&p, cfg.rank_sketch_cols, 0xABC).unwrap();
    let (_, c2) = rank_condition_estimate(&p, cfg.rank_sketch_cols, 0xABC).unwrap();
    assert_eq!(c1, c2);
}
