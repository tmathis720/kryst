#![cfg(feature = "backend-sprs")]
//! Sprs-backed sparse backend.

use std::any::Any;
use std::sync::Arc;

use sprs::{CSR, CsMat, TriMat};

use crate::algebra::prelude::*;
use crate::error::KError;
use crate::matrix::backend::SparseBackend;
use crate::matrix::dense_api::{DenseMatMut, DenseMatRef, DenseMatShape};
use crate::matrix::format::{BackendFormatSupport, OpFormat};
use crate::matrix::op::LinOp;

/// Marker type for the sprs backend.
pub struct SprsBackend;

/// Minimal row-major dense matrix for sprs-backed conversions.
#[derive(Clone, Debug, PartialEq)]
pub struct SprsDenseMat {
    nrows: usize,
    ncols: usize,
    data: Vec<f64>,
}

impl SprsDenseMat {
    pub fn from_row_major(nrows: usize, ncols: usize, data: Vec<f64>) -> Self {
        assert_eq!(
            data.len(),
            nrows * ncols,
            "row-major dense data length must equal nrows * ncols"
        );
        Self { nrows, ncols, data }
    }

    #[inline]
    fn idx(&self, i: usize, j: usize) -> usize {
        i * self.ncols + j
    }
}

impl DenseMatShape for SprsDenseMat {
    fn nrows(&self) -> usize {
        self.nrows
    }

    fn ncols(&self) -> usize {
        self.ncols
    }
}

impl DenseMatRef<f64> for SprsDenseMat {
    fn get(&self, i: usize, j: usize) -> f64 {
        self.data[self.idx(i, j)]
    }
}

impl DenseMatMut<f64> for SprsDenseMat {
    fn set(&mut self, i: usize, j: usize, val: f64) {
        let idx = self.idx(i, j);
        self.data[idx] = val;
    }
}

impl LinOp for SprsDenseMat {
    type S = f64;

    fn dims(&self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }

    fn matvec(&self, x: &[Self::S], y: &mut [Self::S]) {
        y.fill(0.0);
        for i in 0..self.nrows {
            let row_offset = i * self.ncols;
            let mut acc = 0.0;
            for j in 0..self.ncols {
                acc += self.data[row_offset + j] * x[j];
            }
            y[i] = acc;
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn format(&self) -> OpFormat {
        OpFormat::Dense
    }
}

impl LinOp for CsMat<f64> {
    type S = f64;

    fn dims(&self) -> (usize, usize) {
        (self.rows(), self.cols())
    }

    fn matvec(&self, x: &[Self::S], y: &mut [Self::S]) {
        y.fill(0.0);
        if self.storage() != CSR {
            let csr = self.to_csr();
            for (row_ind, row_vec) in csr.outer_iterator().enumerate() {
                let mut acc = 0.0;
                for (col_ind, val) in row_vec.iter() {
                    acc += val * x[col_ind];
                }
                y[row_ind] = acc;
            }
            return;
        }
        for (row_ind, row_vec) in self.outer_iterator().enumerate() {
            let mut acc = 0.0;
            for (col_ind, val) in row_vec.iter() {
                acc += val * x[col_ind];
            }
            y[row_ind] = acc;
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn format(&self) -> OpFormat {
        OpFormat::Csr
    }
}

impl SparseBackend<f64> for SprsBackend {
    const FORMAT_SUPPORT: BackendFormatSupport =
        BackendFormatSupport::new(true, true, false, false);

    type Csr = CsMat<f64>;
    type Csc = ();
    type Dense = SprsDenseMat;

    fn csr_from_dense(dense: &Self::Dense, drop_tol: f64) -> Result<Self::Csr, KError> {
        let mut triplet = TriMat::with_capacity((dense.nrows, dense.ncols), dense.data.len());
        for i in 0..dense.nrows {
            for j in 0..dense.ncols {
                let val = dense.get(i, j);
                if val.abs() > drop_tol {
                    triplet.add_triplet(i, j, val);
                }
            }
        }
        Ok(triplet.to_csr())
    }

    fn csc_from_csr(_csr: &Self::Csr, _drop_tol: f64) -> Self::Csc {
        ()
    }

    fn csr_from_csc(_csc: &Self::Csc, _drop_tol: f64) -> Self::Csr {
        unreachable!("sprs backend does not support CSC materialization")
    }

    fn dense_from_csr(csr: &Self::Csr) -> Result<Self::Dense, KError> {
        let csr = if csr.storage() == CSR {
            csr.clone()
        } else {
            csr.to_csr()
        };
        let mut data = vec![0.0; csr.rows() * csr.cols()];
        for (row_ind, row_vec) in csr.outer_iterator().enumerate() {
            for (col_ind, val) in row_vec.iter() {
                data[row_ind * csr.cols() + col_ind] += val;
            }
        }
        Ok(SprsDenseMat::from_row_major(csr.rows(), csr.cols(), data))
    }

    fn dense_from_csc(_csc: &Self::Csc) -> Result<Self::Dense, KError> {
        Err(KError::Unsupported(
            "sprs backend does not support CSC materialization",
        ))
    }
}

pub fn try_materialize<Scalar, RealScalar>(
    op: Arc<dyn LinOp<S = Scalar>>,
    want: OpFormat,
    drop_tol: RealScalar,
) -> Result<Arc<dyn LinOp<S = Scalar>>, KError>
where
    Scalar: KrystScalar<Real = f64> + 'static,
    RealScalar: Into<f64> + Copy,
{
    // This backend only supports f64 scalars, not complex
    let is_f64 = std::any::TypeId::of::<Scalar>() == std::any::TypeId::of::<f64>();
    if !is_f64 {
        return Err(KError::Unsupported(
            "sprs backend only supports f64 scalars, not complex",
        ));
    }

    if want.is_any() {
        return Ok(op);
    }
    if !<SprsBackend as SparseBackend<f64>>::FORMAT_SUPPORT.supports(want) {
        return Err(KError::Unsupported(
            "sprs backend does not support the requested format",
        ));
    }

    if let Some(csr) = op.as_any().downcast_ref::<CsMat<f64>>() {
        // Safe because we verified Scalar == f64 above
        match want {
            OpFormat::Csr => {
                let result = Arc::new(csr.clone()) as Arc<dyn LinOp<S = f64>>;
                // This is safe because we've verified that Scalar == f64
                #[allow(unsafe_code)]
                unsafe {
                    Ok(std::mem::transmute::<
                        Arc<dyn LinOp<S = f64>>,
                        Arc<dyn LinOp<S = Scalar>>,
                    >(result))
                }
            }
            OpFormat::Dense => {
                let dense = <SprsBackend as SparseBackend<f64>>::dense_from_csr(csr)?;
                let result = Arc::new(dense) as Arc<dyn LinOp<S = f64>>;
                #[allow(unsafe_code)]
                unsafe {
                    Ok(std::mem::transmute::<
                        Arc<dyn LinOp<S = f64>>,
                        Arc<dyn LinOp<S = Scalar>>,
                    >(result))
                }
            }
            OpFormat::Csc | OpFormat::BlockCsr | OpFormat::Any => Err(KError::Unsupported(
                "sprs backend cannot materialize the requested format",
            )),
        }
    } else if let Some(dense) = op.as_any().downcast_ref::<SprsDenseMat>() {
        match want {
            OpFormat::Csr => {
                let csr =
                    <SprsBackend as SparseBackend<f64>>::csr_from_dense(dense, drop_tol.into())?;
                let result = Arc::new(csr) as Arc<dyn LinOp<S = f64>>;
                #[allow(unsafe_code)]
                unsafe {
                    Ok(std::mem::transmute::<
                        Arc<dyn LinOp<S = f64>>,
                        Arc<dyn LinOp<S = Scalar>>,
                    >(result))
                }
            }
            OpFormat::Dense => {
                let result = Arc::new(dense.clone()) as Arc<dyn LinOp<S = f64>>;
                #[allow(unsafe_code)]
                unsafe {
                    Ok(std::mem::transmute::<
                        Arc<dyn LinOp<S = f64>>,
                        Arc<dyn LinOp<S = Scalar>>,
                    >(result))
                }
            }
            OpFormat::Csc | OpFormat::BlockCsr | OpFormat::Any => Err(KError::Unsupported(
                "sprs backend cannot materialize the requested format",
            )),
        }
    } else {
        Err(KError::Unsupported(
            "sprs backend cannot materialize the requested operator",
        ))
    }
}

pub fn try_materialize_ref<Scalar, RealScalar>(
    op: &dyn LinOp<S = Scalar>,
    want: OpFormat,
    drop_tol: RealScalar,
) -> Result<Arc<dyn LinOp<S = Scalar>>, KError>
where
    Scalar: KrystScalar<Real = f64> + 'static,
    RealScalar: Into<f64> + Copy,
{
    // This backend only supports f64 scalars, not complex
    let is_f64 = std::any::TypeId::of::<Scalar>() == std::any::TypeId::of::<f64>();
    if !is_f64 {
        return Err(KError::Unsupported(
            "sprs backend only supports f64 scalars, not complex",
        ));
    }

    if want.is_any() {
        return Err(KError::Unsupported(
            "sprs backend cannot materialize OpFormat::Any",
        ));
    }
    if !<SprsBackend as SparseBackend<f64>>::FORMAT_SUPPORT.supports(want) {
        return Err(KError::Unsupported(
            "sprs backend does not support the requested format",
        ));
    }

    if let Some(csr) = op.as_any().downcast_ref::<CsMat<f64>>() {
        match want {
            OpFormat::Csr => {
                let result = Arc::new(csr.clone()) as Arc<dyn LinOp<S = f64>>;
                #[allow(unsafe_code)]
                unsafe {
                    Ok(std::mem::transmute::<
                        Arc<dyn LinOp<S = f64>>,
                        Arc<dyn LinOp<S = Scalar>>,
                    >(result))
                }
            }
            OpFormat::Dense => {
                let dense = <SprsBackend as SparseBackend<f64>>::dense_from_csr(csr)?;
                let result = Arc::new(dense) as Arc<dyn LinOp<S = f64>>;
                #[allow(unsafe_code)]
                unsafe {
                    Ok(std::mem::transmute::<
                        Arc<dyn LinOp<S = f64>>,
                        Arc<dyn LinOp<S = Scalar>>,
                    >(result))
                }
            }
            OpFormat::Csc | OpFormat::BlockCsr | OpFormat::Any => Err(KError::Unsupported(
                "sprs backend cannot materialize the requested format",
            )),
        }
    } else if let Some(dense) = op.as_any().downcast_ref::<SprsDenseMat>() {
        match want {
            OpFormat::Csr => {
                let csr =
                    <SprsBackend as SparseBackend<f64>>::csr_from_dense(dense, drop_tol.into())?;
                let result = Arc::new(csr) as Arc<dyn LinOp<S = f64>>;
                #[allow(unsafe_code)]
                unsafe {
                    Ok(std::mem::transmute::<
                        Arc<dyn LinOp<S = f64>>,
                        Arc<dyn LinOp<S = Scalar>>,
                    >(result))
                }
            }
            OpFormat::Dense => {
                let result = Arc::new(dense.clone()) as Arc<dyn LinOp<S = f64>>;
                #[allow(unsafe_code)]
                unsafe {
                    Ok(std::mem::transmute::<
                        Arc<dyn LinOp<S = f64>>,
                        Arc<dyn LinOp<S = Scalar>>,
                    >(result))
                }
            }
            OpFormat::Csc | OpFormat::BlockCsr | OpFormat::Any => Err(KError::Unsupported(
                "sprs backend cannot materialize the requested format",
            )),
        }
    } else {
        Err(KError::Unsupported(
            "sprs backend cannot materialize the requested operator",
        ))
    }
}

#[cfg(all(test, not(feature = "complex")))]
mod tests {
    use super::*;
    use crate::matrix::backend;
    use crate::matrix::format::OpFormat;

    #[test]
    fn materialize_dense_and_csr() {
        let dense = SprsDenseMat::from_row_major(2, 2, vec![1.0, 0.0, 0.0, 2.0]);
        let op: Arc<dyn LinOp<S = S>> = Arc::new(dense.clone());

        let csr = backend::materialize(op.clone(), OpFormat::Csr, 0.0).unwrap();
        assert_eq!(csr.format(), OpFormat::Csr);
        let csr_ref = csr.as_any().downcast_ref::<CsMat<f64>>().unwrap();
        assert_eq!(csr_ref.rows(), 2);

        let dense_again =
            backend::materialize(Arc::new(csr_ref.clone()), OpFormat::Dense, 0.0).unwrap();
        assert_eq!(dense_again.format(), OpFormat::Dense);
        let dense_ref = dense_again.as_any().downcast_ref::<SprsDenseMat>().unwrap();
        assert_eq!(dense_ref.get(0, 0), 1.0);
        assert_eq!(dense_ref.get(1, 1), 2.0);
    }
}
