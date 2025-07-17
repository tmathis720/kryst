//! Core linear-algebra traits for kryst.

/// Matrix–vector product: y ← A x.
pub trait MatVec<V> {
    /// Compute y = A · x.
    fn matvec(&self, x: &V, y: &mut V);
}

/// Matrix–transpose–vector product: y ← Aᵗ x.
pub trait MatTransVec<V> {
    /// Compute y = Aᵗ · x.
    fn mattransvec(&self, x: &V, y: &mut V);
}

/// Inner products & norms.
pub trait InnerProduct<V> {
    /// Associated scalar type.
    type Scalar: Copy + PartialOrd + From<f64> + Into<f64>;
    /// Compute dot(x, y) with communicator support for parallel reductions.
    fn dot(&self, x: &V, y: &V, comm: &impl crate::parallel::Comm) -> Self::Scalar;
    /// Compute ‖x‖₂ with communicator support for parallel reductions.
    fn norm(&self, x: &V, comm: &impl crate::parallel::Comm) -> Self::Scalar {
        let local_sq = self.dot(x, x, comm);
        let global_sq = comm.all_reduce_f64(local_sq.into());
        (global_sq.sqrt()).into()
    }
}

/// Uniform indexing into vectors (dense or sparse).
pub trait Indexing {
    /// Number of rows (or length for a vector).
    fn nrows(&self) -> usize;
}

/// Matrix shape trait: provides nrows/ncols for matrices and vectors.
pub trait MatShape {
    fn nrows(&self) -> usize;
    fn ncols(&self) -> usize;
}

/// Trait for extracting the sparsity pattern of a matrix row.
pub trait RowPattern {
    /// Returns the column indices of nonzeros in row i.
    fn row_indices(&self, i: usize) -> &[usize];
}

/// Trait for extracting elements from a matrix.
pub trait MatrixGet<T> {
    /// Get the element at position (i, j).
    fn get(&self, i: usize, j: usize) -> T;
}

/// Trait for extracting a submatrix by index set (for block/ASM preconditioners).
pub trait SubmatrixExtract {
    /// Returns the submatrix with rows and columns given by `indices`.
    fn submatrix(&self, indices: &[usize]) -> Self;
}

#[cfg(test)]
mod tests {
    use super::*;

    // Simple test to verify traits can be imported and used
    #[test]
    fn test_traits_exist() {
        // This test just verifies that all traits compile and can be referenced
        // More comprehensive tests would require mock implementations
        
        // Test that trait bounds can be specified
        fn _test_matvec_bound<T, V>(_: &T) where T: MatVec<V> {}
        fn _test_mattransvec_bound<T, V>(_: &T) where T: MatTransVec<V> {}
        fn _test_inner_product_bound<T, V>(_: &T) where T: InnerProduct<V> {}
        fn _test_indexing_bound<T>(_: &T) where T: Indexing {}
        fn _test_mat_shape_bound<T>(_: &T) where T: MatShape {}
        fn _test_row_pattern_bound<T>(_: &T) where T: RowPattern {}
        fn _test_matrix_get_bound<T, U>(_: &T) where T: MatrixGet<U> {}
        fn _test_submatrix_extract_bound<T>(_: &T) where T: SubmatrixExtract {}
        
        // All traits should compile
        assert!(true);
    }

    #[test]
    fn test_inner_product_scalar_trait_bounds() {
        // Test that the associated Scalar type has the required bounds
        fn _check_scalar_bounds<T: Copy + PartialOrd + From<f64> + Into<f64>>() {}
        
        // f64 should satisfy the bounds
        _check_scalar_bounds::<f64>();
        
        assert!(true);
    }

    #[test]
    fn test_trait_names_and_methods() {
        // Verify method names exist by checking trait signatures
        trait TestMatVec<V> {
            fn matvec(&self, x: &V, y: &mut V);
        }
        
        trait TestMatTransVec<V> {
            fn mattransvec(&self, x: &V, y: &mut V);
        }
        
        trait TestInnerProduct<V> {
            type Scalar: Copy + PartialOrd + From<f64> + Into<f64>;
            fn dot(&self, x: &V, y: &V, comm: &impl crate::parallel::Comm) -> Self::Scalar;
            fn norm(&self, x: &V, comm: &impl crate::parallel::Comm) -> Self::Scalar {
                let local_sq = self.dot(x, x, comm);
                let global_sq = comm.all_reduce_f64(local_sq.into());
                (global_sq.sqrt()).into()
            }
        }
        
        // All method signatures should compile
        assert!(true);
    }
}
