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
    type Scalar: Copy + PartialOrd + From<f64>;
    /// Compute dot(x, y).
    fn dot(&self, x: &V, y: &V) -> Self::Scalar;
    /// Compute ‖x‖₂.
    fn norm(&self, x: &V) -> Self::Scalar;
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
