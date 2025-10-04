use crate::error::KError;
use crate::matrix::op::LinOp;
use crate::matrix::sparse::CsrMatrix;
use crate::preconditioner::stats::{PcIntrospect, PcStats};
use crate::preconditioner::{PcSide, Preconditioner};
use faer::Mat;
use std::sync::atomic::{AtomicU64, Ordering};

pub struct Jacobi {
    pub(crate) diag_inv: Vec<f64>,
    n: usize,
    applies: AtomicU64,
}
impl Default for Jacobi {
    fn default() -> Self {
        Self::new()
    }
}

impl Jacobi {
    pub fn new() -> Self {
        Self {
            diag_inv: Vec::new(),
            n: 0,
            applies: AtomicU64::new(0),
        }
    }

    fn recompute(&mut self, pmat: &dyn LinOp<S = f64>) -> Result<(), KError> {
        if let Some(csr) = pmat.as_any().downcast_ref::<CsrMatrix<f64>>() {
            let n = csr.nrows().min(csr.ncols());
            self.diag_inv.resize(n, 0.0);
            for i in 0..n {
                let rs = csr.row_ptr()[i];
                let re = csr.row_ptr()[i + 1];
                let mut aii = 0.0;
                for p in rs..re {
                    if csr.col_idx()[p] == i {
                        aii = csr.values()[p];
                        break;
                    }
                }
                self.diag_inv[i] = if aii.abs() > 1e-14 { 1.0 / aii } else { 0.0 };
            }
            self.n = n;
            return Ok(());
        }
        if let Some(d) = pmat.as_any().downcast_ref::<Mat<f64>>() {
            let n = d.nrows().min(d.ncols());
            self.diag_inv.resize(n, 0.0);
            for i in 0..n {
                let aii = d[(i, i)];
                self.diag_inv[i] = if aii.abs() > 1e-14 { 1.0 / aii } else { 0.0 };
            }
            self.n = n;
            return Ok(());
        }
        Err(KError::InvalidInput("Jacobi needs Dense or CSR".into()))
    }
}
impl Preconditioner for Jacobi {
    fn dims(&self) -> (usize, usize) {
        (self.n, self.n)
    }

    fn setup(&mut self, pmat: &dyn LinOp<S = f64>) -> Result<(), KError> {
        self.recompute(pmat)
    }
    fn supports_numeric_update(&self) -> bool {
        true
    }

    fn update_numeric(&mut self, pmat: &dyn LinOp<S = f64>) -> Result<(), KError> {
        self.recompute(pmat)
    }
    fn apply(&self, _side: PcSide, r: &[f64], z: &mut [f64]) -> Result<(), KError> {
        if r.len() != self.n || z.len() != self.n {
            return Err(KError::InvalidInput(format!(
                "Jacobi::apply dimension mismatch: n={}, r.len()={}, z.len()={}",
                self.n,
                r.len(),
                z.len()
            )));
        }
        for i in 0..self.n {
            z[i] = self.diag_inv[i] * r[i];
        }
        self.applies.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

impl crate::preconditioner::legacy::Preconditioner<Mat<f64>, Vec<f64>> for Jacobi {
    fn setup(&mut self, a: &Mat<f64>) -> Result<(), KError> {
        self.recompute(a)
    }
    fn apply(&self, side: PcSide, r: &Vec<f64>, z: &mut Vec<f64>) -> Result<(), KError> {
        crate::preconditioner::Preconditioner::apply(self, side, r.as_slice(), z.as_mut_slice())
    }
}

impl PcIntrospect for Jacobi {
    fn stats(&self) -> PcStats {
        PcStats {
            name: "Jacobi",
            n: self.n,
            build_ms: 0.0,
            nnz_pc: self.n,
            fill_ratio: 0.0,
            applies: self.applies.load(Ordering::Relaxed),
        }
    }
}
