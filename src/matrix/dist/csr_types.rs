use crate::algebra::scalar::KrystScalar;
use crate::error::KError;
use crate::matrix::sparse::CsrMatrix;

/// Marker wrapper for a CSR slice with owned local rows and global columns.
///
/// This is the canonical shape extracted from a distributed row partition.
#[derive(Clone, Debug)]
pub struct DistRowCsr<S: KrystScalar> {
    csr: CsrMatrix<S>,
    row_offset: usize,
    n_global_cols: usize,
}

impl<S: KrystScalar> DistRowCsr<S> {
    pub fn new(csr: CsrMatrix<S>, row_offset: usize, n_global_cols: usize) -> Result<Self, KError> {
        if csr.ncols() != n_global_cols {
            return Err(KError::InvalidInput(format!(
                "DistRowCsr ncols mismatch: csr has {}, expected {n_global_cols}",
                csr.ncols()
            )));
        }
        Ok(Self {
            csr,
            row_offset,
            n_global_cols,
        })
    }

    pub fn as_csr(&self) -> &CsrMatrix<S> {
        &self.csr
    }

    pub fn row_offset(&self) -> usize {
        self.row_offset
    }

    pub fn n_global_cols(&self) -> usize {
        self.n_global_cols
    }
}

/// Marker wrapper for a locally owned square CSR block.
///
/// This type is intended for local factorizations (ILU/ILUT/ILUTP) that
/// require square ownership semantics.
#[derive(Clone, Debug)]
pub struct LocalSquareCsr<S: KrystScalar> {
    csr: CsrMatrix<S>,
}

impl<S: KrystScalar> LocalSquareCsr<S> {
    pub fn try_from_csr(csr: CsrMatrix<S>) -> Result<Self, KError> {
        if csr.nrows() != csr.ncols() {
            return Err(KError::InvalidInput(format!(
                "LocalSquareCsr requires square matrix, got {}x{}",
                csr.nrows(),
                csr.ncols()
            )));
        }
        Ok(Self { csr })
    }

    pub fn as_csr(&self) -> &CsrMatrix<S> {
        &self.csr
    }
}

impl<S: KrystScalar> TryFrom<CsrMatrix<S>> for LocalSquareCsr<S> {
    type Error = KError;

    fn try_from(value: CsrMatrix<S>) -> Result<Self, Self::Error> {
        Self::try_from_csr(value)
    }
}

#[cfg(test)]
mod tests {
    use super::LocalSquareCsr;
    use crate::matrix::sparse::CsrMatrix;

    #[test]
    fn local_square_csr_rejects_rectangular_with_clear_error() {
        let rectangular = CsrMatrix::from_csr(
            2,
            3,
            vec![0, 2, 4],
            vec![0, 1, 1, 2],
            vec![1.0, 2.0, 3.0, 4.0],
        );

        let err = LocalSquareCsr::try_from(rectangular).expect_err("rectangular must fail");
        let msg = err.to_string();
        assert!(
            msg.contains("LocalSquareCsr requires square matrix") && msg.contains("2x3"),
            "unexpected rejection text: {msg}"
        );
    }
}
