#![cfg(not(feature = "complex"))]
use kryst::algebra::prelude::*;
use kryst::matrix::sparse::CsrMatrix;
use kryst::preconditioner::{PcSide, Preconditioner};
use proptest::prelude::*;

// Build a random strictly diagonally dominant symmetric CSR
fn random_strictly_diag_dominant_csr(n: usize, bandwidth: usize) -> CsrMatrix<R> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx: Vec<usize> = Vec::new();
    let mut vals: Vec<R> = Vec::new();
    row_ptr.push(0);
    for i in 0..n {
        // Assemble candidate column indices within the bandwidth
        let mut row_cols = Vec::new();
        row_cols.push(i);
        for k in 1..=bandwidth {
            if i + k < n {
                row_cols.push(i + k);
            }
            if k <= i {
                row_cols.push(i - k);
            }
        }
        row_cols.sort_unstable();
        row_cols.dedup();

        // Build (col, val) entries, compute off-diagonal sum for strict diagonal dominance
        let mut entries: Vec<(usize, R)> = Vec::with_capacity(row_cols.len());
        let mut sum_abs = R::default();
        for &j in &row_cols {
            if j == i {
                continue; // defer diagonal until after off-diagonal sum
            }
            let v = R::from(rng.gen_range(-0.5..0.5));
            entries.push((j, v));
            sum_abs += v.abs();
        }
        // strictly dominant diagonal
        entries.push((i, sum_abs + R::from(1.0)));
        // Ensure CSR row has sorted column indices
        entries.sort_unstable_by_key(|e| e.0);
        for (j, v) in entries {
            col_idx.push(j);
            vals.push(v);
        }
        row_ptr.push(col_idx.len());
    }
    CsrMatrix::from_csr(n, n, row_ptr, col_idx, vals)
}

proptest! {
  #[test]
  fn pc_apply_produces_finite(n in 5usize..30) {
    let a = random_strictly_diag_dominant_csr(n, 3);
    let mut jac = kryst::preconditioner::jacobi::Jacobi::new();
    prop_assert!(jac.setup(&a).is_ok());

    let x: Vec<R> = (0..n).map(|k| R::from((k as f64).sin())).collect();
    let mut y: Vec<R> = vec![R::default(); n];
    prop_assert!(jac.apply(PcSide::Left, &x, &mut y).is_ok());
    prop_assert!(y.iter().all(|v| v.is_finite()));
  }
}
