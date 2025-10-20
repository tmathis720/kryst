#[cfg(feature = "complex")]
use crate::algebra::bridge::BridgeScratch;
use crate::algebra::prelude::*;
use crate::error::KError;
use crate::matrix::op::LinOp;
use crate::matrix::sparse::CsrMatrix;
#[cfg(feature = "complex")]
use crate::ops::kpc::KPreconditioner;
use crate::preconditioner::stats::{PcIntrospect, PcStats};
use crate::preconditioner::{PcSide, Preconditioner};
use faer::Mat;
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(feature = "rayon")]
use rayon::prelude::*;

pub struct Jacobi {
    pub(crate) diag_inv: Vec<S>,
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
            self.diag_inv.resize(n, S::zero());
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
                self.diag_inv[i] = if aii.abs() > 1e-14 {
                    S::from_real(1.0 / aii)
                } else {
                    S::zero()
                };
            }
            self.n = n;
            return Ok(());
        }
        if let Some(d) = pmat.as_any().downcast_ref::<Mat<f64>>() {
            let n = d.nrows().min(d.ncols());
            self.diag_inv.resize(n, S::zero());
            for i in 0..n {
                let aii = d[(i, i)];
                self.diag_inv[i] = if aii.abs() > 1e-14 {
                    S::from_real(1.0 / aii)
                } else {
                    S::zero()
                };
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
        #[cfg_attr(not(feature = "rayon"), allow(unused_mut))]
        let mut used_parallel = false;
        #[cfg(feature = "rayon")]
        {
            if r.len() >= crate::parallel_cfg::parallel_tune().min_len_vec {
                z.par_iter_mut()
                    .zip(r.par_iter().copied())
                    .zip(self.diag_inv.par_iter().copied())
                    .for_each(|((zi, ri), di)| {
                        *zi = di.real() * ri;
                    });
                used_parallel = true;
            }
        }
        if !used_parallel {
            for (zi, (&ri, &di)) in z.iter_mut().zip(r.iter().zip(self.diag_inv.iter())) {
                *zi = di.real() * ri;
            }
        }
        self.applies.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

#[cfg(feature = "complex")]
impl KPreconditioner for Jacobi {
    type Scalar = S;

    #[inline]
    fn dims(&self) -> (usize, usize) {
        (self.n, self.n)
    }

    fn apply_s(
        &self,
        _side: PcSide,
        x: &[S],
        y: &mut [S],
        _scratch: &mut BridgeScratch,
    ) -> Result<(), KError> {
        if x.len() != self.n || y.len() != self.n {
            return Err(KError::InvalidInput(format!(
                "Jacobi::apply_s dimension mismatch: n={}, x.len()={}, y.len()={}",
                self.n,
                x.len(),
                y.len()
            )));
        }

        let mut used_parallel = false;
        #[cfg(feature = "rayon")]
        {
            if x.len() >= crate::parallel_cfg::parallel_tune().min_len_vec {
                y.par_iter_mut()
                    .zip(x.par_iter().copied())
                    .zip(self.diag_inv.par_iter().copied())
                    .for_each(|((yi, xi), di)| {
                        *yi = di * xi;
                    });
                used_parallel = true;
            }
        }
        if !used_parallel {
            for (yi, (&xi, &di)) in y.iter_mut().zip(x.iter().zip(self.diag_inv.iter())) {
                *yi = di * xi;
            }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::bridge::BridgeScratch;
    use crate::ops::kpc::KPreconditioner;

    #[test]
    fn apply_s_matches_real_path() {
        let mut pc = Jacobi::new();
        let dense = Mat::<f64>::from_fn(2, 2, |i, j| if i == j { [4.0, 9.0][i] } else { 0.0 });
        pc.setup(&dense).expect("jacobi setup");

        let rhs_real = [8.0, 18.0];
        let mut out_real = [0.0; 2];
        pc.apply(PcSide::Left, &rhs_real, &mut out_real)
            .expect("jacobi apply real");

        let rhs_s: Vec<S> = rhs_real.iter().copied().map(S::from_real).collect();
        let mut out_s = vec![S::zero(); rhs_s.len()];
        let mut scratch = BridgeScratch::default();
        pc.apply_s(PcSide::Left, &rhs_s, &mut out_s, &mut scratch)
            .expect("jacobi apply_s");

        for (ys, &yr) in out_s.iter().zip(out_real.iter()) {
            assert!((ys.real() - yr).abs() < 1e-12);
        }
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
