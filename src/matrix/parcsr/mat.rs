use super::halo::HaloPlan;
use crate::algebra::prelude::*;
use crate::error::KError;
use crate::matrix::op::LinOp;
use crate::matrix::sparse::CsrMatrix;
use crate::parallel::{Comm, UniverseComm};
use std::sync::Arc;

/// Distributed CSR matrix split into on- and off-process blocks.
#[derive(Clone)]
pub struct ParCsrMatrix {
    pub comm: UniverseComm,
    pub row_start: usize,
    pub row_end: usize,
    pub global_n: usize,
    pub global_m: usize,
    pub a_diag: CsrMatrix<S>,
    pub a_off: CsrMatrix<S>,
    pub colmap_owned: Vec<usize>,
    pub colmap_ghost: Vec<usize>,
    pub halo: HaloPlan,
}

/// LinOp adapter that exposes a [`ParCsrMatrix`] through the common interface.
pub struct ParCsrOp {
    pub mat: Arc<ParCsrMatrix>,
}

impl ParCsrOp {
    pub fn new(mat: Arc<ParCsrMatrix>) -> Self {
        Self { mat }
    }

    pub fn from_owned(mat: ParCsrMatrix) -> Self {
        Self { mat: Arc::new(mat) }
    }
}

impl LinOp for ParCsrOp {
    type S = S;

    fn dims(&self) -> (usize, usize) {
        (self.mat.local_n(), self.mat.global_m)
    }

    fn matvec(&self, x: &[Self::S], y: &mut [Self::S]) {
        self.mat
            .spmv(x, y)
            .expect("ParCsrMatrix::spmv dimension mismatch");
    }

    fn try_matvec(&self, x: &[Self::S], y: &mut [Self::S]) -> Result<(), KError> {
        self.mat.spmv(x, y)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn comm(&self) -> UniverseComm {
        self.mat.comm.clone()
    }

    fn format(&self) -> crate::matrix::format::OpFormat {
        crate::matrix::format::OpFormat::Csr
    }
}

impl ParCsrMatrix {
    /// Number of locally owned rows.
    pub fn local_n(&self) -> usize {
        self.row_end - self.row_start
    }

    /// y = alpha*A*x + beta*y with two-phase halo exchange.
    pub fn spmv_scaled(
        &self,
        alpha: S,
        x_owned: &[S],
        beta: S,
        y_owned: &mut [S],
    ) -> Result<(), KError> {
        if x_owned.len() != self.local_n() || y_owned.len() != self.local_n() {
            return Err(KError::InvalidInput(
                "dimension mismatch in ParCsrMatrix::spmv".into(),
            ));
        }

        let mut x_ghost: Vec<S> = vec![S::zero(); self.colmap_ghost.len()];
        let mut recv_buf: Vec<S> = vec![S::zero(); self.halo.recv_idx.len()];
        let mut send_buf: Vec<S> = vec![S::zero(); self.halo.send_idx.len()];
        let mut reqs = self
            .halo
            .begin_exchange(&self.comm, x_owned, &mut send_buf, &mut recv_buf);

        self.a_diag.spmv_scaled(alpha, x_owned, beta, y_owned)?;

        self.comm.wait_all(&mut reqs);
        self.halo.unpack(&recv_buf, &mut x_ghost);

        self.a_off.spmv_scaled(alpha, &x_ghost, S::one(), y_owned)?;
        Ok(())
    }

    /// Convenience wrapper for y = A*x.
    pub fn spmv(&self, x_owned: &[S], y_owned: &mut [S]) -> Result<(), KError> {
        self.spmv_scaled(S::one(), x_owned, S::zero(), y_owned)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::matrix::sparse::CsrMatrix;
    use crate::parallel::{NoComm, UniverseComm};

    #[test]
    fn spmv_local_only() {
        // A = diag([2, 3])
        let a_diag = CsrMatrix::from_csr(
            2,
            2,
            vec![0, 1, 2],
            vec![0, 1],
            vec![S::from_real(2.0), S::from_real(3.0)],
        );
        let a_off = CsrMatrix::from_csr(2, 0, vec![0, 0, 0], Vec::new(), Vec::new());
        let halo = HaloPlan::default();
        let par = ParCsrMatrix {
            comm: UniverseComm::NoComm(NoComm),
            row_start: 0,
            row_end: 2,
            global_n: 2,
            global_m: 2,
            a_diag,
            a_off,
            colmap_owned: vec![0, 1],
            colmap_ghost: Vec::new(),
            halo,
        };
        let x = vec![S::from_real(1.0), S::from_real(2.0)];
        let mut y = vec![S::zero(); 2];
        par.spmv(&x, &mut y).unwrap();
        assert_eq!(y, vec![S::from_real(2.0), S::from_real(6.0)]);
    }
}
