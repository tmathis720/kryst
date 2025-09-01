// Minimal CSR fixtures for golden tests
use kryst::matrix::sparse::CsrMatrix;

pub fn csr_poisson_1d(n: usize) -> CsrMatrix<f64> {
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::new();
    let mut vals = Vec::new();
    row_ptr.push(0);
    for i in 0..n {
        if i > 0 {
            col_idx.push(i - 1);
            vals.push(-1.0);
        }
        col_idx.push(i);
        vals.push(2.0);
        if i + 1 < n {
            col_idx.push(i + 1);
            vals.push(-1.0);
        }
        row_ptr.push(col_idx.len());
    }
    CsrMatrix::from_csr(n, n, row_ptr, col_idx, vals)
}

#[allow(dead_code)]
pub fn csr_identity(n: usize) -> CsrMatrix<f64> {
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

