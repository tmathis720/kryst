use kryst::matrix::sparse::CsrMatrix;
use kryst::preconditioner::{Preconditioner, PcSide};
use proptest::prelude::*;

// Build a random strictly diagonally dominant symmetric CSR
fn random_strictly_diag_dominant_csr(n: usize, bandwidth: usize) -> CsrMatrix<f64> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx: Vec<usize> = Vec::new();
    let mut vals: Vec<f64> = Vec::new();
    row_ptr.push(0);
    for i in 0..n {
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
        let mut sum_abs = 0.0f64;
        let mut diag = 0.0f64;
        for &j in &row_cols {
            if j == i {
                diag = 0.0;
            } else {
                let v = rng.gen_range(-0.5..0.5);
                col_idx.push(j);
                vals.push(v);
                sum_abs += v.abs();
            }
        }
        // strictly dominant diagonal
        diag = sum_abs + 1.0;
        col_idx.push(i);
        vals.push(diag);
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

    let x: Vec<f64> = (0..n).map(|k| (k as f64).sin()).collect();
    let mut y = vec![0.0; n];
    prop_assert!(jac.apply(PcSide::Left, &x, &mut y).is_ok());
    prop_assert!(y.iter().all(|v| v.is_finite()));
  }
}

