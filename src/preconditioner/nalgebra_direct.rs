#![cfg(feature = "backend-nalgebra")]

use crate::algebra::prelude::*;
use crate::error::KError;
use crate::matrix::op::LinOp;
use crate::parallel::Comm;
use crate::preconditioner::{PcDistributedSupport, PcSide, Preconditioner};
use nalgebra::{DMatrix, DVector, Dynamic};

fn extract_dense(op: &dyn LinOp<S = S>) -> Result<DMatrix<S>, KError> {
    let dense = op
        .as_any()
        .downcast_ref::<crate::matrix::op_nalgebra::NalgebraDenseOp>()
        .ok_or_else(|| {
            KError::InvalidInput("nalgebra direct PC requires NalgebraDenseOp".into())
        })?;
    Ok(dense.inner().clone())
}

fn ensure_trivial_comm(op: &dyn LinOp<S = S>) -> Result<(), KError> {
    if !op.comm().is_trivial() && op.comm().size() > 1 {
        return Err(KError::Unsupported(
            "nalgebra direct preconditioners require a trivial communicator",
        ));
    }
    Ok(())
}

pub struct NalgebraLuPc {
    lu: Option<nalgebra::linalg::LU<S, Dynamic, Dynamic>>,
    n: usize,
}

impl NalgebraLuPc {
    pub fn new() -> Self {
        Self { lu: None, n: 0 }
    }

    fn factorize(&mut self, op: &dyn LinOp<S = S>) -> Result<(), KError> {
        ensure_trivial_comm(op)?;
        let a = extract_dense(op)?;
        if a.nrows() != a.ncols() {
            return Err(KError::InvalidInput(
                "nalgebra LU requires a square matrix".into(),
            ));
        }
        let n = a.nrows();
        let lu = a.lu();
        self.lu = Some(lu);
        self.n = n;
        Ok(())
    }
}

impl Preconditioner for NalgebraLuPc {
    fn dims(&self) -> (usize, usize) {
        (self.n, self.n)
    }

    fn setup(&mut self, op: &dyn LinOp<S = S>) -> Result<(), KError> {
        self.factorize(op)
    }

    fn apply(&self, _side: PcSide, _r: &[S], _z: &mut [S]) -> Result<(), KError> {
        Err(KError::Unsupported(
            "NalgebraLuPc::apply is PREONLY-only; use SolverType::Preonly or call direct_solve",
        ))
    }

    fn direct_solve(&mut self, op: &dyn LinOp<S = S>, b: &[S], x: &mut [S]) -> Result<(), KError> {
        if self.lu.is_none() {
            self.factorize(op)?;
        }
        let lu = self
            .lu
            .as_ref()
            .ok_or_else(|| KError::SolveError("nalgebra LU factorization missing".into()))?;
        if b.len() != self.n || x.len() != self.n {
            return Err(KError::InvalidInput(
                "nalgebra LU solve dimension mismatch".into(),
            ));
        }
        let b_vec = DVector::from_column_slice(b);
        let sol = lu
            .solve(&b_vec)
            .ok_or_else(|| KError::FactorError("nalgebra LU solve failed".into()))?;
        x.copy_from_slice(sol.as_slice());
        Ok(())
    }

    fn required_format(&self) -> crate::matrix::format::OpFormat {
        crate::matrix::format::OpFormat::Dense
    }

    fn distributed_support(&self) -> PcDistributedSupport {
        PcDistributedSupport::LocalOnly
    }
}

pub struct NalgebraQrPc {
    qr: Option<nalgebra::linalg::QR<S, Dynamic, Dynamic>>,
    n: usize,
}

impl NalgebraQrPc {
    pub fn new() -> Self {
        Self { qr: None, n: 0 }
    }

    fn factorize(&mut self, op: &dyn LinOp<S = S>) -> Result<(), KError> {
        ensure_trivial_comm(op)?;
        let a = extract_dense(op)?;
        if a.nrows() != a.ncols() {
            return Err(KError::InvalidInput(
                "nalgebra QR requires a square matrix".into(),
            ));
        }
        let n = a.nrows();
        let qr = a.qr();
        self.qr = Some(qr);
        self.n = n;
        Ok(())
    }
}

impl Preconditioner for NalgebraQrPc {
    fn dims(&self) -> (usize, usize) {
        (self.n, self.n)
    }

    fn setup(&mut self, op: &dyn LinOp<S = S>) -> Result<(), KError> {
        self.factorize(op)
    }

    fn apply(&self, _side: PcSide, _r: &[S], _z: &mut [S]) -> Result<(), KError> {
        Err(KError::Unsupported(
            "NalgebraQrPc::apply is PREONLY-only; use SolverType::Preonly or call direct_solve",
        ))
    }

    fn direct_solve(&mut self, op: &dyn LinOp<S = S>, b: &[S], x: &mut [S]) -> Result<(), KError> {
        if self.qr.is_none() {
            self.factorize(op)?;
        }
        let qr = self
            .qr
            .as_ref()
            .ok_or_else(|| KError::SolveError("nalgebra QR factorization missing".into()))?;
        if b.len() != self.n || x.len() != self.n {
            return Err(KError::InvalidInput(
                "nalgebra QR solve dimension mismatch".into(),
            ));
        }
        let b_vec = DVector::from_column_slice(b);
        let sol = qr
            .solve(&b_vec)
            .ok_or_else(|| KError::FactorError("nalgebra QR solve failed".into()))?;
        x.copy_from_slice(sol.as_slice());
        Ok(())
    }

    fn required_format(&self) -> crate::matrix::format::OpFormat {
        crate::matrix::format::OpFormat::Dense
    }

    fn distributed_support(&self) -> PcDistributedSupport {
        PcDistributedSupport::LocalOnly
    }
}
