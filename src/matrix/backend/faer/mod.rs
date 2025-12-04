//! Faer-backed matrix implementations of the abstraction traits.

pub mod dense;
pub mod format;
pub mod sparse;

/// Marker type for the default Faer backend.
pub struct FaerBackend;

pub type DefaultDenseMat<S> = faer::Mat<S>;
pub type DefaultCsrMat<S> = crate::matrix::sparse::CsrMatrix<S>;
pub type DefaultCscMat<S> = crate::matrix::csc::CscMatrix<S>;

#[cfg(test)]
mod tests {
    use crate::matrix::dense_api::DenseMatRef;
    use crate::matrix::sparse::CsrMatrix;
    use crate::matrix::sparse_api::CsrMatRef;
    use crate::matrix::spmv;

    fn assert_dense_ref<T: DenseMatRef<f64>>() {}
    fn assert_csr_ref<T: CsrMatRef<f64>>() {}

    #[test]
    fn faer_mat_satisfies_dense_traits() {
        assert_dense_ref::<faer::Mat<f64>>();
    }

    #[test]
    fn csr_matrix_drives_generic_spmv() {
        assert_csr_ref::<CsrMatrix<f64>>();
        let a = CsrMatrix::from_csr(
            2,
            2,
            vec![0, 2, 3],
            vec![0, 1, 1],
            vec![1.0, 2.0, 3.0],
        );
        let x = vec![1.0, 1.0];
        let mut y = vec![0.0; 2];
        spmv::spmv_csr_serial(&a, &x, &mut y).unwrap();
        assert_eq!(y, vec![3.0, 3.0]);
    }
}
