use crate::matrix::sparse::CsrMatrix;

#[derive(Clone, Debug)]
pub struct Strength {
    pub row_ptr: Vec<usize>,
    pub col_idx: Vec<usize>,
}

impl Strength {
    pub fn from_csr(a: &CsrMatrix<f64>, theta: f64, normalize: bool) -> Self {
        strength_csr(a, theta, normalize)
    }
}

pub fn strength_csr(a: &CsrMatrix<f64>, theta: f64, normalize: bool) -> Strength {
    let n = a.nrows();
    let rp = a.row_ptr();
    let cj = a.col_idx();
    let vv = a.values();

    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx: Vec<usize> = Vec::with_capacity(a.nnz());
    row_ptr.push(0);

    if normalize {
        // |a_ij| / sqrt(|a_ii||a_jj|)
        let mut diag = vec![0.0f64; n];
        for i in 0..n {
            let rs = rp[i]; let re = rp[i + 1];
            let mut aii = 0.0;
            for p in rs..re {
                if cj[p] == i { aii = vv[p].abs(); break; }
            }
            diag[i] = aii;
        }
        for i in 0..n {
            let rs = rp[i]; let re = rp[i + 1];
            let mut count = 0usize;
            for p in rs..re {
                let j = cj[p];
                if j == i { continue; }
                let denom = (diag[i] * diag[j]).sqrt();
                if denom > 0.0 {
                    let s = vv[p].abs() / denom;
                    if s >= theta { col_idx.push(j); count += 1; }
                }
            }
            row_ptr.push(row_ptr.last().unwrap() + count);
        }
    } else {
        // |a_ij| >= theta * max_off_i
        for i in 0..n {
            let rs = rp[i]; let re = rp[i + 1];
            let mut max_off = 0.0f64;
            for p in rs..re { let j = cj[p]; if j != i { max_off = max_off.max(vv[p].abs()); } }
            let thr = theta * max_off;
            let mut count = 0usize;
            for p in rs..re {
                let j = cj[p];
                if j != i && vv[p].abs() >= thr { col_idx.push(j); count += 1; }
            }
            row_ptr.push(row_ptr.last().unwrap() + count);
        }
    }

    Strength { row_ptr, col_idx }
}

