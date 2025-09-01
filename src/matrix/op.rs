use crate::error::KError;
use crate::parallel::{Comm, NoComm, UniverseComm};
use faer::traits::ComplexField;
use std::any::Any;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

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

    /// Fallible matvec. Default delegates to `matvec` and returns `Ok(())`.
    /// Implementations that can detect/return errors should override this.
    fn try_matvec(&self, x: &[Self::S], y: &mut [Self::S]) -> Result<(), KError> {
        self.matvec(x, y);
        Ok(())
    }

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
    /// - `A.comm() == P.comm()` is enforced by [`KspContext::try_set_operators`]
    ///   (and `set_operators` panics on mismatch).
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

/// Wrap your concrete dense matrix with `DenseOp` to provide stable
/// `StructureId`/`ValuesId` so conversions and preconditioner reuse can be cached.
pub struct DenseOp {
    mat: Arc<Mat<f64>>,
    ids: ChangeIds,
    comm: UniverseComm,
}
impl DenseOp {
    /// Wrap a dense matrix so changes can be tracked via [`mark_structure_changed`] and
    /// [`mark_values_changed`]. This enables correct caching and preconditioner reuse across
    /// nonlinear or time-stepping updates.
    pub fn new(mat: Arc<Mat<f64>>) -> Self {
        let ids = ChangeIds::default();
        ids.bump_structure();
        ids.bump_values();
        Self {
            mat,
            ids,
            comm: UniverseComm::NoComm(NoComm),
        }
    }
    /// Attach a communicator to this operator.
    pub fn with_comm(mut self, comm: UniverseComm) -> Self {
        self.comm = comm;
        self
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
        for (j, yj) in y.iter_mut().enumerate().take(self.mat.ncols()) {
            let mut sum = 0.0;
            for (i, xi) in x.iter().enumerate().take(self.mat.nrows()) {
                sum += self.mat[(i, j)] * *xi;
            }
            *yj = sum;
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
    fn comm(&self) -> UniverseComm {
        self.comm.clone()
    }
}

/// CSR-backed linear operator.
///
/// Wrap your concrete CSR matrix with `CsrOp` to provide stable
/// `StructureId`/`ValuesId` so conversions and preconditioner reuse can be cached.
///
/// # Threading policy
///
/// When built with the `rayon` feature, [`CsrOp::matvec`] may switch to a
/// parallel SpMV path. The decision is:
///
/// 1. The operator is local-only (`self.comm.size() == 1`).
/// 2. The current Rayon pool has > 1 threads.
/// 3. The matrix is large enough: `nrows >= KRYST_PAR_CUTOFF`
///    (default [`crate::parallel::threads::DEFAULT_PAR_CUTOFF`]).
///
/// If any of these are false, SpMV runs single-threaded.
///
/// ## Tuning knobs (environment variables)
///
/// - `KRYST_THREADS` (only with `rayon`): sets the total number of Rayon threads.
///   If unset, we fall back to `RAYON_NUM_THREADS`, then to `num_cpus`.
///   When using MPI, the pool is sized per rank as
///   `max(1, total_threads / mpi_size)`.
///
/// - `KRYST_PAR_CUTOFF`: minimum `nrows` to enable the parallel SpMV path.
///   Default: [`crate::parallel::threads::DEFAULT_PAR_CUTOFF`].
///
/// ## Examples
/// ```no_run
/// # // Shell example: enable a bigger pool and lower the cutoff
/// # std::env::set_var("KRYST_THREADS", "16");
/// # std::env::set_var("KRYST_PAR_CUTOFF", "2048");
/// use kryst::matrix::sparse::CsrMatrix;
/// use kryst::matrix::op::CsrOp;
/// use std::sync::Arc;
///
/// // Build/own a CSR, then wrap it as a LinOp.
/// let csr = CsrMatrix::identity(10_000);
/// let op  = CsrOp::new(Arc::new(csr));
///
/// // y = A * x; will use Rayon if compiled with `rayon` and nrows>=cutoff.
/// let x = vec![1.0; 10_000];
/// let mut y = vec![0.0; 10_000];
/// op.matvec(&x, &mut y);
/// ```
///
/// ## Notes
/// - If you run under MPI and the communicator has size > 1, the parallel
///   path is disabled in [`CsrOp::matvec`] (it’s intended for shared-memory).
/// - See [`crate::parallel::threads`] for details on pool sizing and MPI.
pub struct CsrOp {
    csr: Arc<CsrMatrix<f64>>,
    ids: ChangeIds,
    comm: UniverseComm,
    #[cfg(feature = "transpose-cache")]
    t_cache: parking_lot::RwLock<Option<(ValuesId, Arc<CscMatrix<f64>>)>>,
}
impl CsrOp {
    pub fn new(csr: Arc<CsrMatrix<f64>>) -> Self {
        let ids = ChangeIds::default();
        ids.bump_structure();
        ids.bump_values();
        Self {
            csr,
            ids,
            comm: UniverseComm::NoComm(NoComm),
            #[cfg(feature = "transpose-cache")]
            t_cache: parking_lot::RwLock::new(None),
        }
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
    /// Attach a communicator to this operator.
    pub fn with_comm(mut self, comm: UniverseComm) -> Self {
        self.comm = comm;
        self
    }
}
impl LinOp for CsrOp {
    type S = f64;
    fn dims(&self) -> (usize, usize) {
        (self.csr.nrows(), self.csr.ncols())
    }
    fn matvec(&self, x: &[f64], y: &mut [f64]) {
        #[cfg(feature = "rayon")]
        {
            let local_only = self.comm.size() == 1;
            let threads = crate::parallel::threads::current_rayon_threads();
            let cutoff = crate::parallel::threads::env_usize(
                "KRYST_PAR_CUTOFF",
                crate::parallel::threads::DEFAULT_PAR_CUTOFF,
            );
            let big_enough = self.csr.nrows() >= cutoff;

            if local_only && threads > 1 && big_enough {
                #[cfg(feature = "logging")]
                log::trace!(
                    "CsrOp::matvec using Rayon (rows={}, threads={}, cutoff={})",
                    self.csr.nrows(),
                    threads,
                    cutoff,
                );
                let _ = crate::matrix::spmv::spmv_csr_parallel(self.csr.as_ref(), x, y);
                return;
            } else {
                #[cfg(feature = "logging")]
                log::trace!(
                    "CsrOp::matvec serial path (local_only={}, threads={}, rows={}, cutoff={})",
                    local_only,
                    threads,
                    self.csr.nrows(),
                    cutoff,
                );
            }
        }

        self.csr.spmv(x, y);
    }
    fn supports_transpose(&self) -> bool {
        true
    }
    fn t_matvec(&self, x: &[f64], y: &mut [f64]) {
        #[cfg(feature = "transpose-cache")]
        {
            if let Some(csc) = self.ensure_csc_view() {
                let _ = crate::matrix::spmv::t_spmv_csr_parallel(
                    self.csr.as_ref(),
                    crate::matrix::spmv::TBackend::Csc(&csc),
                    x,
                    y,
                );
                return;
            }
        }
        let _ = crate::matrix::spmv::t_spmv_csr_parallel(
            self.csr.as_ref(),
            crate::matrix::spmv::TBackend::CsrGather,
            x,
            y,
        );
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
    fn comm(&self) -> UniverseComm {
        self.comm.clone()
    }
}

#[cfg(feature = "transpose-cache")]
impl CsrOp {
    pub fn ensure_csc_view(&self) -> Option<Arc<CscMatrix<f64>>> {
        use crate::matrix::format::AsFormat;
        let vid = self.values_id();
        {
            let guard = self.t_cache.read();
            if let Some((cached_vid, csc)) = &*guard {
                if *cached_vid == vid {
                    return Some(csc.clone());
                }
            }
        }
        let csc = AsFormat::to_csc_cached(self.csr.as_ref(), 0.0);
        {
            let mut guard = self.t_cache.write();
            *guard = Some((vid, csc.clone()));
        }
        Some(csc)
    }
}

// --- Direct adapters ------------------------------------------------------
// If the `mat-values-fingerprint` feature is enabled, `Mat<f64>::values_id()` computes an
// O(m*n) fingerprint of the numeric values to strengthen cache invalidation for users who
// do not wrap with `DenseOp`. By default (feature off), `values_id()` returns 0 for `Mat<f64>`
// and callers should prefer `DenseOp` + `mark_values_changed()` for precise reuse.
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
        #[cfg(feature = "mat-values-fingerprint")]
        {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let mut h = DefaultHasher::new();
            let (m, n) = (self.nrows(), self.ncols());
            m.hash(&mut h);
            n.hash(&mut h);
            // Full scan for correctness; opt-in via feature due to cost.
            for i in 0..m {
                for j in 0..n {
                    self[(i, j)].to_bits().hash(&mut h);
                }
            }
            ValuesId(h.finish())
        }
        #[cfg(not(feature = "mat-values-fingerprint"))]
        {
            ValuesId(0)
        }
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
    fn try_matvec(&self, x: &[Self::S], y: &mut [Self::S]) -> Result<(), KError> {
        self.inner.try_matvec(x, y)
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
