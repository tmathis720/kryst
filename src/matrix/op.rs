use crate::parallel::{NoComm, UniverseComm};
use faer::traits::ComplexField;
use std::any::Any;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StructureId(pub u64);
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ValuesId(pub u64);

/// Format-agnostic linear operator.
pub trait LinOp: Send + Sync + Any {
    type S: ComplexField;

    /// Dimensions (rows, cols).
    fn dims(&self) -> (usize, usize);

    /// Compute y = A x.
    fn matvec(&self, x: &[Self::S], y: &mut [Self::S]);

    /// Whether this operator supports `t_matvec`.
    fn supports_transpose(&self) -> bool {
        false
    }

    /// Optional transpose/adjoint matvec. Default panics if unsupported.
    fn t_matvec(&self, _x: &[Self::S], _y: &mut [Self::S]) {
        panic!("LinOp::t_matvec called but supports_transpose() == false");
    }

    /// Downcast hook for specialized solvers/preconditioners.
    fn as_any(&self) -> &dyn Any;

    /// Changes when the nonzero pattern / shape changes.
    /// Default 0 -> unknown; higher layers may fall back to pointer identity.
    fn structure_id(&self) -> StructureId {
        StructureId(0)
    }

    /// Changes when only numerical values change.
    /// Default 0 -> unknown.
    fn values_id(&self) -> ValuesId {
        ValuesId(0)
    }

    /// Parallel communicator for this operator.
    ///
    /// Returns the communicator that owns the operator, used by distributed PCs
    /// and solvers. Local/dense operators return [`UniverseComm::NoComm`].
    ///
    /// # Invariants
    /// - `A.comm() == P.comm()` is enforced by [`KspContext::set_operators`].
    /// - PCs obtain their communicator from the operator passed to [`Preconditioner::setup`].
    fn comm(&self) -> UniverseComm {
        UniverseComm::NoComm(NoComm)
    }
}

/// Simple bumpable counters for LinOp implementors/wrappers.
#[derive(Default)]
pub struct ChangeIds {
    pub sid: AtomicU64,
    pub vid: AtomicU64,
}
impl ChangeIds {
    pub fn structure_id(&self) -> StructureId {
        StructureId(self.sid.load(Ordering::Relaxed))
    }
    pub fn values_id(&self) -> ValuesId {
        ValuesId(self.vid.load(Ordering::Relaxed))
    }
    pub fn bump_structure(&self) {
        self.sid.fetch_add(1, Ordering::Relaxed);
    }
    pub fn bump_values(&self) {
        self.vid.fetch_add(1, Ordering::Relaxed);
    }
}

// --- Optional wrappers for dense and CSR matrices -------------------------
use crate::matrix::{csc::CscMatrix, sparse::CsrMatrix};
use faer::Mat;

pub struct DenseOp {
    mat: Arc<Mat<f64>>,
    ids: ChangeIds,
}
impl DenseOp {
    pub fn new(mat: Arc<Mat<f64>>) -> Self {
        let ids = ChangeIds::default();
        ids.bump_structure();
        ids.bump_values();
        Self { mat, ids }
    }
    pub fn mark_structure_changed(&self) {
        self.ids.bump_structure();
    }
    pub fn mark_values_changed(&self) {
        self.ids.bump_values();
    }
    pub fn inner(&self) -> &Mat<f64> {
        &self.mat
    }
}
impl LinOp for DenseOp {
    type S = f64;
    fn dims(&self) -> (usize, usize) {
        (self.mat.nrows(), self.mat.ncols())
    }
    fn matvec(&self, x: &[f64], y: &mut [f64]) {
        let _ = crate::matrix::utils::parallel_mat_vec(&self.mat, x, y);
    }
    fn supports_transpose(&self) -> bool {
        true
    }
    fn t_matvec(&self, x: &[f64], y: &mut [f64]) {
        assert_eq!(x.len(), self.mat.nrows());
        assert_eq!(y.len(), self.mat.ncols());
        for j in 0..self.mat.ncols() {
            let mut sum = 0.0;
            for i in 0..self.mat.nrows() {
                sum += self.mat[(i, j)] * x[i];
            }
            y[j] = sum;
        }
    }
    fn as_any(&self) -> &dyn Any {
        &*self.mat
    }
    fn structure_id(&self) -> StructureId {
        self.ids.structure_id()
    }
    fn values_id(&self) -> ValuesId {
        self.ids.values_id()
    }
}

pub struct CsrOp {
    csr: Arc<CsrMatrix<f64>>,
    ids: ChangeIds,
}
impl CsrOp {
    pub fn new(csr: Arc<CsrMatrix<f64>>) -> Self {
        let ids = ChangeIds::default();
        ids.bump_structure();
        ids.bump_values();
        Self { csr, ids }
    }
    pub fn mark_structure_changed(&self) {
        self.ids.bump_structure();
    }
    pub fn mark_values_changed(&self) {
        self.ids.bump_values();
    }
    pub fn inner(&self) -> &CsrMatrix<f64> {
        &self.csr
    }
}
impl LinOp for CsrOp {
    type S = f64;
    fn dims(&self) -> (usize, usize) {
        (self.csr.nrows(), self.csr.ncols())
    }
    fn matvec(&self, x: &[f64], y: &mut [f64]) {
        self.csr.spmv(x, y);
    }
    fn supports_transpose(&self) -> bool {
        true
    }
    fn t_matvec(&self, x: &[f64], y: &mut [f64]) {
        assert_eq!(x.len(), self.csr.nrows());
        assert_eq!(y.len(), self.csr.ncols());
        y.fill(0.0);
        let rp = self.csr.row_ptr();
        let ci = self.csr.col_idx();
        let vv = self.csr.values();
        for i in 0..self.csr.nrows() {
            let xi = x[i];
            for idx in rp[i]..rp[i + 1] {
                y[ci[idx]] += vv[idx] * xi;
            }
        }
    }
    fn as_any(&self) -> &dyn Any {
        &*self.csr
    }
    fn structure_id(&self) -> StructureId {
        self.ids.structure_id()
    }
    fn values_id(&self) -> ValuesId {
        self.ids.values_id()
    }
}

// --- Direct adapters ------------------------------------------------------
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

impl LinOp for Mat<f64> {
    type S = f64;
    fn dims(&self) -> (usize, usize) {
        (self.nrows(), self.ncols())
    }
    fn matvec(&self, x: &[f64], y: &mut [f64]) {
        assert_eq!(x.len(), self.ncols());
        assert_eq!(y.len(), self.nrows());
        for i in 0..self.nrows() {
            let mut sum = 0.0;
            for j in 0..self.ncols() {
                sum += self[(i, j)] * x[j];
            }
            y[i] = sum;
        }
    }
    fn supports_transpose(&self) -> bool {
        true
    }
    fn t_matvec(&self, x: &[f64], y: &mut [f64]) {
        assert_eq!(x.len(), self.nrows());
        assert_eq!(y.len(), self.ncols());
        for j in 0..self.ncols() {
            let mut sum = 0.0;
            for i in 0..self.nrows() {
                sum += self[(i, j)] * x[i];
            }
            y[j] = sum;
        }
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn structure_id(&self) -> StructureId {
        let mut h = DefaultHasher::new();
        self.nrows().hash(&mut h);
        self.ncols().hash(&mut h);
        StructureId(h.finish())
    }
    fn values_id(&self) -> ValuesId {
        ValuesId(0)
    }
}

use crate::matrix::sparse::SparseMatrix;
impl LinOp for CsrMatrix<f64> {
    type S = f64;
    fn dims(&self) -> (usize, usize) {
        (self.nrows(), self.ncols())
    }
    fn matvec(&self, x: &[f64], y: &mut [f64]) {
        self.spmv(x, y);
    }
    fn supports_transpose(&self) -> bool {
        true
    }
    fn t_matvec(&self, x: &[f64], y: &mut [f64]) {
        assert_eq!(x.len(), self.nrows());
        assert_eq!(y.len(), self.ncols());
        y.fill(0.0);
        let rp = self.row_ptr();
        let ci = self.col_idx();
        let vv = self.values();
        for i in 0..self.nrows() {
            let xi = x[i];
            for idx in rp[i]..rp[i + 1] {
                y[ci[idx]] += vv[idx] * xi;
            }
        }
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn structure_id(&self) -> StructureId {
        let mut h = DefaultHasher::new();
        self.row_ptr().hash(&mut h);
        self.col_idx().hash(&mut h);
        StructureId(h.finish())
    }
    fn values_id(&self) -> ValuesId {
        ValuesId(0)
    }
}

impl LinOp for CscMatrix<f64> {
    type S = f64;

    fn dims(&self) -> (usize, usize) {
        (self.nrows(), self.ncols())
    }

    fn matvec(&self, x: &[f64], y: &mut [f64]) {
        assert_eq!(x.len(), self.ncols());
        assert_eq!(y.len(), self.nrows());
        y.fill(0.0);
        let cp = self.col_ptr();
        let ri = self.row_idx();
        let vv = self.values();
        for j in 0..self.ncols() {
            let xj = x[j];
            for p in cp[j]..cp[j + 1] {
                y[ri[p]] += vv[p] * xj;
            }
        }
    }

    fn supports_transpose(&self) -> bool {
        true
    }

    fn t_matvec(&self, x: &[f64], y: &mut [f64]) {
        assert_eq!(x.len(), self.nrows());
        assert_eq!(y.len(), self.ncols());
        y.fill(0.0);
        let cp = self.col_ptr();
        let ri = self.row_idx();
        let vv = self.values();
        for j in 0..self.ncols() {
            let mut sum = 0.0;
            for p in cp[j]..cp[j + 1] {
                sum += vv[p] * x[ri[p]];
            }
            y[j] = sum;
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn structure_id(&self) -> StructureId {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut h = DefaultHasher::new();
        self.col_ptr().hash(&mut h);
        self.row_idx().hash(&mut h);
        StructureId(h.finish())
    }

    fn values_id(&self) -> ValuesId {
        ValuesId(0)
    }
}
/// Wrap any LinOp with a communicator without changing its behavior.
pub struct WithCommOp<T: LinOp + ?Sized> {
    inner: Arc<T>,
    comm: UniverseComm,
}

impl<T: LinOp + ?Sized> WithCommOp<T> {
    pub fn new(inner: Arc<T>, comm: UniverseComm) -> Self {
        Self { inner, comm }
    }
    pub fn inner(&self) -> &T {
        &self.inner
    }
}

impl<T: LinOp + ?Sized> LinOp for WithCommOp<T> {
    type S = T::S;

    #[inline]
    fn dims(&self) -> (usize, usize) {
        self.inner.dims()
    }
    #[inline]
    fn matvec(&self, x: &[Self::S], y: &mut [Self::S]) {
        self.inner.matvec(x, y)
    }
    #[inline]
    fn supports_transpose(&self) -> bool {
        self.inner.supports_transpose()
    }
    #[inline]
    fn t_matvec(&self, x: &[Self::S], y: &mut [Self::S]) {
        self.inner.t_matvec(x, y)
    }
    #[inline]
    fn as_any(&self) -> &dyn Any {
        self.inner.as_any()
    }
    #[inline]
    fn structure_id(&self) -> StructureId {
        self.inner.structure_id()
    }
    #[inline]
    fn values_id(&self) -> ValuesId {
        self.inner.values_id()
    }
    #[inline]
    fn comm(&self) -> UniverseComm {
        self.comm.clone()
    }
}

/// Ergonomic helper for call sites
pub fn wrap_with_comm<T>(op: Arc<T>, comm: UniverseComm) -> Arc<dyn LinOp<S = T::S>>
where
    T: LinOp + ?Sized + 'static,
{
    Arc::new(WithCommOp::new(op, comm)) as Arc<dyn LinOp<S = T::S>>
}
