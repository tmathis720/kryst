#![cfg(feature = "backend-nalgebra")]

use std::sync::Arc;

use crate::algebra::prelude::*;
use crate::matrix::format::OpFormat;
use crate::matrix::op::{LinOp, StructureId, ValuesId};
use crate::parallel::{NoComm, UniverseComm};

#[derive(Clone)]
pub struct NalgebraDenseOp {
    a: Arc<nalgebra::DMatrix<S>>,
    comm: UniverseComm,
    sid: StructureId,
    vid: ValuesId,
}

impl NalgebraDenseOp {
    pub fn new(a: Arc<nalgebra::DMatrix<S>>) -> Self {
        let p = Arc::as_ptr(&a) as *const () as usize as u64;
        Self {
            a,
            comm: UniverseComm::NoComm(NoComm),
            sid: StructureId(p),
            vid: ValuesId(p),
        }
    }

    pub fn with_comm(mut self, comm: UniverseComm) -> Self {
        self.comm = comm;
        self
    }

    pub fn inner(&self) -> &nalgebra::DMatrix<S> {
        self.a.as_ref()
    }

    pub fn mark_values_changed(&mut self) {
        self.vid = ValuesId(self.vid.0.wrapping_add(1).max(1));
    }

    pub fn mark_structure_changed(&mut self) {
        self.sid = StructureId(self.sid.0.wrapping_add(1).max(1));
        self.mark_values_changed();
    }
}

impl LinOp for NalgebraDenseOp {
    type S = S;

    fn dims(&self) -> (usize, usize) {
        (self.a.nrows(), self.a.ncols())
    }

    fn matvec(&self, x: &[S], y: &mut [S]) {
        let (m, n) = self.dims();
        debug_assert_eq!(x.len(), n);
        debug_assert_eq!(y.len(), m);
        for i in 0..m {
            let mut sum = S::zero();
            for j in 0..n {
                sum += self.a[(i, j)] * x[j];
            }
            y[i] = sum;
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn comm(&self) -> UniverseComm {
        self.comm.clone()
    }

    fn structure_id(&self) -> StructureId {
        self.sid
    }

    fn values_id(&self) -> ValuesId {
        self.vid
    }

    fn format(&self) -> OpFormat {
        OpFormat::Dense
    }
}

#[cfg(feature = "nalgebra-explicit-dense")]
use crate::error::KError;

#[cfg(feature = "nalgebra-explicit-dense")]
pub fn explicit_dense_from_linop(
    op: Arc<dyn LinOp<S = S>>,
) -> Result<Arc<dyn LinOp<S = S>>, KError> {
    const EXPLICIT_DENSE_MAX_N: usize = 1024;

    let (m, n) = op.dims();
    if m == 0 || n == 0 {
        return Err(KError::InvalidInput(
            "explicit dense materialization requires known dimensions".into(),
        ));
    }
    if m > EXPLICIT_DENSE_MAX_N || n > EXPLICIT_DENSE_MAX_N {
        return Err(KError::Unsupported(
            "explicit dense materialization is limited to 1024x1024",
        ));
    }
    let nn = m
        .checked_mul(n)
        .ok_or_else(|| KError::InvalidInput("dimension overflow in dense materialize".into()))?;

    let mut data = Vec::with_capacity(nn);
    let mut x = vec![S::zero(); n];
    let mut col = vec![S::zero(); m];
    for j in 0..n {
        x[j] = S::one();
        op.try_matvec(&x, &mut col)?;
        data.extend_from_slice(&col);
        x[j] = S::zero();
    }

    let mat = nalgebra::DMatrix::<S>::from_column_slice(m, n, &data);
    let dense = NalgebraDenseOp::new(Arc::new(mat)).with_comm(op.comm());
    Ok(Arc::new(dense))
}
