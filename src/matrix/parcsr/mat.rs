use super::halo::HaloPlan;
use crate::algebra::prelude::*;
use crate::error::KError;
use crate::matrix::sparse::CsrMatrix;
use crate::parallel::{Comm, UniverseComm};

/// Distributed CSR matrix split into on- and off-process blocks.
#[derive(Clone)]
pub struct ParCsrMatrix {
    pub comm: UniverseComm,
    pub row_start: usize,
    pub row_end: usize,
    pub global_n: usize,
    pub global_m: usize,
    pub a_diag: CsrMatrix<f64>,
    pub a_off: CsrMatrix<f64>,
    pub colmap_owned: Vec<usize>,
    pub colmap_ghost: Vec<usize>,
    pub halo: HaloPlan,
}

impl ParCsrMatrix {
    /// Number of locally owned rows.
    pub fn local_n(&self) -> usize {
        self.row_end - self.row_start
    }

    /// y = alpha*A*x + beta*y with two-phase halo exchange.
    pub fn spmv_scaled(
        &self,
        alpha: f64,
        x_owned: &[f64],
        beta: f64,
        y_owned: &mut [f64],
    ) -> Result<(), KError> {
        if x_owned.len() != self.local_n() || y_owned.len() != self.local_n() {
            return Err(KError::InvalidInput(
                "dimension mismatch in ParCsrMatrix::spmv".into(),
            ));
        }

        let mut x_ghost: Vec<R> = vec![R::default(); self.colmap_ghost.len()];
        let mut recv_buf: Vec<R> = vec![R::default(); self.halo.recv_idx.len()];
        let mut send_buf: Vec<R> = vec![R::default(); self.halo.send_idx.len()];
        let mut reqs = self
            .halo
            .begin_exchange(&self.comm, x_owned, &mut send_buf, &mut recv_buf);

        self.a_diag.spmv_scaled(alpha, x_owned, beta, y_owned)?;

        self.comm.wait_all(&mut reqs);
        self.halo.unpack(&recv_buf, &mut x_ghost);

        self.a_off.spmv_scaled(alpha, &x_ghost, 1.0, y_owned)?;
        Ok(())
    }

    /// Convenience wrapper for y = A*x.
    pub fn spmv(&self, x_owned: &[f64], y_owned: &mut [f64]) -> Result<(), KError> {
        self.spmv_scaled(1.0, x_owned, 0.0, y_owned)
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
        let a_diag = CsrMatrix::from_csr(2, 2, vec![0, 1, 2], vec![0, 1], vec![2.0, 3.0]);
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
        let x = vec![R::from(1.0), R::from(2.0)];
        let mut y = vec![R::default(); 2];
        par.spmv(&x, &mut y).unwrap();
        assert_eq!(y, vec![R::from(2.0), R::from(6.0)]);
    }
}
