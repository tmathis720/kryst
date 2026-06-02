use super::halo::HaloPlan;
use crate::algebra::prelude::*;
use crate::error::KError;
use crate::matrix::dist_csr::DistCsrOp;
use crate::matrix::op::LinOp;
use crate::matrix::sparse::CsrMatrix;
use crate::parallel::UniverseComm;
use std::sync::{Arc, OnceLock};

/// Distributed CSR matrix split into on- and off-process blocks.
#[derive(Clone)]
pub struct ParCsrMatrix {
    pub comm: UniverseComm,
    pub row_start: usize,
    pub row_end: usize,
    pub global_n: usize,
    pub global_m: usize,
    #[deprecated(
        since = "1.1.0",
        note = "Legacy diag/off storage is compatibility-only; use canonical_dist_op()/DistCsrOp accessors. Planned removal after 2026-12-31"
    )]
    pub a_diag: CsrMatrix<S>,
    #[deprecated(
        since = "1.1.0",
        note = "Legacy diag/off storage is compatibility-only; use canonical_dist_op()/DistCsrOp accessors. Planned removal after 2026-12-31"
    )]
    pub a_off: CsrMatrix<S>,
    #[deprecated(
        since = "1.1.0",
        note = "Legacy column maps are compatibility-only; use canonical_dist_op()/DistCsrOp row_partition/layout metadata. Planned removal after 2026-12-31"
    )]
    pub colmap_owned: Vec<usize>,
    #[deprecated(
        since = "1.1.0",
        note = "Legacy column maps are compatibility-only; use canonical_dist_op()/DistCsrOp row_partition/layout metadata. Planned removal after 2026-12-31"
    )]
    pub colmap_ghost: Vec<usize>,
    #[deprecated(
        since = "1.1.0",
        note = "Legacy halo internals are compatibility-only; use canonical_dist_op()/DistCsrOp exchange paths. Planned removal after 2026-12-31"
    )]
    pub halo: HaloPlan,
    canonical: OnceLock<Arc<DistCsrOp>>,
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
        (self.mat.local_n(), self.mat.local_n())
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

    fn dist_layout(&self) -> Option<&crate::matrix::op::DistLayout> {
        self.mat
            .canonical_dist_op()
            .ok()
            .and_then(|op| op.dist_layout())
    }

    fn format(&self) -> crate::matrix::format::OpFormat {
        crate::matrix::format::OpFormat::Csr
    }
}

impl ParCsrMatrix {
    #[deprecated(
        since = "1.1.0",
        note = "Legacy diag/off constructor is compatibility-only; use DistCsrOp::from_local_rows and canonical distributed APIs. Planned removal after 2026-12-31"
    )]
    pub fn from_legacy_parts(
        comm: UniverseComm,
        row_start: usize,
        row_end: usize,
        global_n: usize,
        global_m: usize,
        a_diag: CsrMatrix<S>,
        a_off: CsrMatrix<S>,
        colmap_owned: Vec<usize>,
        colmap_ghost: Vec<usize>,
        halo: HaloPlan,
    ) -> Self {
        Self {
            comm,
            row_start,
            row_end,
            global_n,
            global_m,
            a_diag,
            a_off,
            colmap_owned,
            colmap_ghost,
            halo,
            canonical: OnceLock::new(),
        }
    }

    /// Number of locally owned rows.
    pub fn local_n(&self) -> usize {
        self.row_end - self.row_start
    }

    /// Canonical distributed operator backing this legacy wrapper.
    pub fn canonical_dist_op(&self) -> Result<&DistCsrOp, KError> {
        if let Some(op) = self.canonical.get() {
            return Ok(op.as_ref());
        }
        let op = Arc::new(DistCsrOp::from_parcsr(self)?);
        let _ = self.canonical.set(op);
        Ok(self
            .canonical
            .get()
            .expect("ParCsrMatrix canonical operator was set")
            .as_ref())
    }

    #[deprecated(
        since = "1.1.0",
        note = "ParCsr halo internals are legacy compatibility only; prefer canonical_dist_op(). Planned removal after 2026-12-31"
    )]
    pub fn legacy_halo_plan(&self) -> &HaloPlan {
        &self.halo
    }

    #[deprecated(
        since = "1.1.0",
        note = "Legacy diag block access is compatibility-only; prefer canonical_dist_op().local_block_csr(). Planned removal after 2026-12-31"
    )]
    pub fn legacy_diag_block(&self) -> &CsrMatrix<S> {
        &self.a_diag
    }

    #[deprecated(
        since = "1.1.0",
        note = "Legacy off block access is compatibility-only; prefer canonical_dist_op().local_matrix()/layout metadata. Planned removal after 2026-12-31"
    )]
    pub fn legacy_off_block(&self) -> &CsrMatrix<S> {
        &self.a_off
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
        let mut tmp = vec![S::zero(); self.local_n()];
        self.canonical_dist_op()?.try_matvec(x_owned, &mut tmp)?;
        for (yi, ai) in y_owned.iter_mut().zip(tmp.iter()) {
            *yi = alpha * *ai + beta * *yi;
        }
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
    use std::sync::Arc;

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
            canonical: OnceLock::new(),
        };
        let x = vec![S::from_real(1.0), S::from_real(2.0)];
        let mut y = vec![S::zero(); 2];
        par.spmv(&x, &mut y).unwrap();
        assert_eq!(y, vec![S::from_real(2.0), S::from_real(6.0)]);
    }

    #[cfg(feature = "complex")]
    #[test]
    fn spmv_complex_local_only() {
        // A = diag([2, 3]) with complex entries
        let a_diag = CsrMatrix::from_csr(
            2,
            2,
            vec![0, 1, 2],
            vec![0, 1],
            vec![S::from_parts(2.0, -1.0), S::from_parts(3.0, 0.5)],
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
            canonical: OnceLock::new(),
        };
        let x = vec![S::from_parts(1.0, 2.0), S::from_parts(-1.0, 0.5)];
        let mut y = vec![S::zero(); 2];
        par.spmv(&x, &mut y).unwrap();
        let expected = vec![
            S::from_parts(2.0, -1.0) * x[0],
            S::from_parts(3.0, 0.5) * x[1],
        ];
        assert_eq!(y, expected);
    }

    #[cfg(all(feature = "complex", feature = "mpi"))]
    #[test]
    fn spmv_complex_simple_halo_exchange() {
        use crate::parallel::{Comm, MpiComm};

        let comm = MpiComm::new();
        let rank = comm.rank();
        let size = comm.size();
        if size != 2 {
            return;
        }

        let comm = UniverseComm::Mpi(Arc::new(comm));
        let (row_start, row_end) = if rank == 0 { (0, 1) } else { (1, 2) };
        let (diag_val, off_val, colmap_owned, colmap_ghost) = if rank == 0 {
            (S::from_real(1.0), S::from_real(2.0), vec![0], vec![1])
        } else {
            (S::from_real(4.0), S::from_real(3.0), vec![1], vec![0])
        };

        let a_diag = CsrMatrix::from_csr(1, 1, vec![0, 1], vec![0], vec![diag_val]);
        let a_off = CsrMatrix::from_csr(1, 1, vec![0, 1], vec![0], vec![off_val]);
        let halo = HaloPlan {
            neighbors: vec![if rank == 0 { 1 } else { 0 }],
            send_ptr: vec![0, 1],
            send_idx: vec![0],
            recv_ptr: vec![0, 1],
            recv_idx: vec![0],
        };

        let par = ParCsrMatrix {
            comm,
            row_start,
            row_end,
            global_n: 2,
            global_m: 2,
            a_diag,
            a_off,
            colmap_owned,
            colmap_ghost,
            halo,
            canonical: OnceLock::new(),
        };

        let x0 = S::from_parts(1.0, 2.0);
        let x1 = S::from_parts(3.0, 4.0);
        let x_owned = if rank == 0 { vec![x0] } else { vec![x1] };
        let mut y = vec![S::zero(); 1];
        par.spmv(&x_owned, &mut y).unwrap();

        let expected = if rank == 0 {
            S::from_real(1.0) * x0 + S::from_real(2.0) * x1
        } else {
            S::from_real(3.0) * x0 + S::from_real(4.0) * x1
        };
        assert_eq!(y[0], expected);
    }
}
