#[cfg(feature = "complex")]
use crate::algebra::bridge::BridgeScratch;
use crate::algebra::prelude::*;
use crate::error::KError;
#[cfg(feature = "backend-faer")]
use crate::matrix::csr::CsrMatrix as ScalarCsrMatrix;
use crate::matrix::format::OpFormat;
#[cfg(feature = "backend-faer")]
use crate::matrix::spmv::plan::{self as spmv_plan, SpmvPlan as ScalarSpmvPlan, SpmvTuning};
#[cfg(feature = "complex")]
use crate::ops::klinop::KLinOp;
use crate::parallel::{NoComm, UniverseComm};
use std::any::Any;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// PETSc-like contiguous ownership ranges for distributed operators.
#[derive(Clone, Debug)]
pub struct DistLayout {
    pub global_rows: usize,
    pub global_cols: usize,
    pub row_start: usize,
    pub row_end: usize,
    pub col_start: usize,
    pub col_end: usize,
}

/// Opaque handle returned by a halo exchange start.
pub struct HaloHandle {
    inner: Box<dyn Any + Send>,
}

impl HaloHandle {
    pub fn new<T: Any + Send>(value: T) -> Self {
        Self {
            inner: Box::new(value),
        }
    }

    pub fn downcast<T: Any + Send>(self) -> Result<T, Self> {
        match self.inner.downcast::<T>() {
            Ok(value) => Ok(*value),
            Err(inner) => Err(Self { inner }),
        }
    }
}

/// Optional halo exchange capability for vectors living on the operator layout.
pub trait HaloExchange<S>: Send + Sync {
    fn begin(&self, x_local: &[S]) -> Result<HaloHandle, KError>;
    fn end(&self, handle: HaloHandle, x_with_halo: &mut [S]) -> Result<(), KError>;
}

/// Opaque identifier for an operator's structural pattern.
///
/// Bump this when the sparsity pattern or dimensions change. Returning
/// `StructureId(0)` means "unknown" and forces conservative cache invalidation.
///
/// Typical patterns:
/// - Immutable matrix: keep the ID fixed.
/// - Re-assemble with a new pattern: bump `StructureId` (and usually `ValuesId`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StructureId(pub u64);

/// Opaque identifier for an operator's numeric values.
///
/// Bump this when numeric values change but the structure is unchanged.
/// Returning `ValuesId(0)` means "unknown" and forces conservative refreshes.
///
/// Typical patterns:
/// - Fixed structure, changing values each step: bump `ValuesId` only.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ValuesId(pub u64);

/// Format-agnostic linear operator.
///
/// # Thread-safety / reentrancy
/// Unless otherwise stated, `LinOp` implementations are not required to support
/// concurrent `matvec` calls on the same instance. Callers must avoid invoking
/// `matvec`/`try_matvec` on the same operator from multiple threads at once.
///
/// # Change IDs
/// - `structure_id()` must change when the sparsity pattern or dimensions change.
/// - `values_id()` must change when only the numeric values change.
/// - Returning `StructureId(0)` or `ValuesId(0)` means "unknown" and causes
///   downstream caches (format/PC conversions) to fall back to pointer identity.
///   Wrappers that mutate matrices in place should call
///   [`mark_structure_changed`] / [`mark_values_changed`] to keep caches valid.
pub trait LinOp: Send + Sync + Any {
    type S: KrystScalar;

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
    /// - `A.comm()` and `P.comm()` must be congruent; enforced by [`KspContext::try_set_operators`]
    ///   (and `set_operators` panics on mismatch).
    /// - PCs obtain their communicator from the operator passed to [`Preconditioner::setup`].
    fn comm(&self) -> UniverseComm {
        UniverseComm::NoComm(NoComm)
    }

    /// Optional distributed layout metadata for MPI-enabled operators.
    fn dist_layout(&self) -> Option<&DistLayout> {
        None
    }

    /// Optional halo exchange support for operator-aligned vectors.
    fn halo_exchange(&self) -> Option<&dyn HaloExchange<Self::S>> {
        None
    }

    /// Operator storage format, if known.
    fn format(&self) -> OpFormat {
        OpFormat::Any
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

/// Scalar-generic CSR linear operator backed by [`ScalarCsrMatrix`].
///
/// This wrapper tracks [`StructureId`]/[`ValuesId`] so callers can mutate the
/// owned matrix and keep format caches valid. After any in-place edits,
/// call [`mark_structure_changed`] or [`mark_values_changed`] before reusing
/// the operator in cached conversions.
///
/// The operator wires the scalar-polymorphic sparse matrix storage into the
/// [`LinOp`] trait by constructing an [`SpmvPlan`](ScalarSpmvPlan) at creation
/// time. Real builds continue to select SIMD kernels when available while
/// complex builds transparently fall back to the portable scalar path.
#[cfg(feature = "backend-faer")]
pub struct GenericCsrOp<S: KrystScalar> {
    matrix: Arc<ScalarCsrMatrix<S>>,
    plan: ScalarSpmvPlan<S>,
    ids: ChangeIds,
    comm: UniverseComm,
    layout: Option<DistLayout>,
}

#[cfg(feature = "backend-faer")]
impl<S: KrystScalar> GenericCsrOp<S> {
    /// Wraps an [`Arc`] around the provided matrix and builds an SpMV plan
    /// using the supplied tuning parameters.
    pub fn new(matrix: Arc<ScalarCsrMatrix<S>>, tuning: &SpmvTuning) -> Self {
        let plan = spmv_plan::build(matrix.as_ref(), tuning);
        let ids = ChangeIds::default();
        ids.bump_structure();
        ids.bump_values();
        Self {
            matrix,
            plan,
            ids,
            comm: UniverseComm::NoComm(NoComm),
            layout: None,
        }
    }

    /// Builds an operator from an owned matrix by first wrapping it in an
    /// [`Arc`] for cheap cloning.
    pub fn from_matrix(matrix: ScalarCsrMatrix<S>, tuning: &SpmvTuning) -> Self {
        Self::new(Arc::new(matrix), tuning)
    }

    /// Lifts a real-valued CSR matrix into the active scalar domain.
    ///
    /// This clones the underlying structure and converts the numeric values
    /// through [`KrystScalar::from_real`], enabling solvers in complex builds to
    /// reuse pre-existing `f64` sparsity patterns without reassembly.
    pub fn from_real_csr(real: &crate::matrix::sparse::CsrMatrix<f64>, tuning: &SpmvTuning) -> Self
    where
        S: KrystScalar<Real = f64>,
    {
        let owned = ScalarCsrMatrix::<S>::from_real_csr(real);
        Self::from_matrix(owned, tuning)
    }

    /// Returns the underlying CSR matrix.
    pub fn matrix(&self) -> &ScalarCsrMatrix<S> {
        self.matrix.as_ref()
    }

    /// Returns a reference to the cached SpMV plan.
    pub fn plan(&self) -> &ScalarSpmvPlan<S> {
        &self.plan
    }

    /// Rebuilds the internal SpMV plan using the latest tuning parameters.
    pub fn rebuild_plan(&mut self, tuning: &SpmvTuning) {
        self.plan = spmv_plan::build(self.matrix.as_ref(), tuning);
        self.ids.bump_values();
    }

    /// Attaches a communicator to the operator, mirroring the legacy API.
    pub fn with_comm(mut self, comm: UniverseComm) -> Self {
        self.comm = comm;
        self
    }

    /// Attach a distributed layout to this operator.
    pub fn with_layout(mut self, layout: DistLayout) -> Self {
        self.layout = Some(layout);
        self
    }

    /// Marks the sparsity pattern as changed for cache bookkeeping.
    pub fn mark_structure_changed(&self) {
        self.ids.bump_structure();
    }

    /// Marks only the numeric values as changed.
    pub fn mark_values_changed(&self) {
        self.ids.bump_values();
    }
}

#[cfg(feature = "backend-faer")]
impl<S: KrystScalar> LinOp for GenericCsrOp<S> {
    type S = S;

    fn dims(&self) -> (usize, usize) {
        self.matrix.dims()
    }

    fn matvec(&self, x: &[S], y: &mut [S]) {
        let (m, n) = self.matrix.dims();
        debug_assert_eq!(x.len(), n);
        debug_assert_eq!(y.len(), m);
        self.plan.apply_scaled(S::one(), x, S::zero(), y);
    }

    fn try_matvec(&self, x: &[S], y: &mut [S]) -> Result<(), KError> {
        let (m, n) = self.matrix.dims();
        if x.len() != n || y.len() != m {
            return Err(KError::InvalidInput(format!(
                "GenericCsrOp::matvec dimension mismatch: A={}x{}, x.len()={}, y.len()={}",
                m,
                n,
                x.len(),
                y.len()
            )));
        }
        self.plan.apply_scaled(S::one(), x, S::zero(), y);
        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
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
    fn dist_layout(&self) -> Option<&DistLayout> {
        self.layout.as_ref()
    }
    fn format(&self) -> OpFormat {
        OpFormat::Csr
    }
}

/// Convenience trait for callers that only operate on real scalars.
pub trait LinOpF64 {
    fn dims(&self) -> (usize, usize);
    fn matvec(&self, x: &[f64], y: &mut [f64]);
}

#[cfg(feature = "backend-faer")]
impl LinOpF64 for GenericCsrOp<f64> {
    #[inline]
    fn dims(&self) -> (usize, usize) {
        <Self as LinOp>::dims(self)
    }

    #[inline]
    fn matvec(&self, x: &[f64], y: &mut [f64]) {
        <Self as LinOp>::matvec(self, x, y)
    }
}

#[cfg(feature = "complex")]
#[cfg(all(feature = "backend-faer", feature = "complex"))]
impl KLinOp for GenericCsrOp<num_complex::Complex64> {
    type Scalar = num_complex::Complex64;

    #[inline]
    fn dims(&self) -> (usize, usize) {
        self.matrix.dims()
    }

    #[inline]
    fn matvec_s(
        &self,
        x: &[num_complex::Complex64],
        y: &mut [num_complex::Complex64],
        _scratch: &mut BridgeScratch,
    ) {
        <Self as LinOp>::matvec(self, x, y);
    }

    #[inline]
    fn supports_t_matvec_s(&self) -> bool {
        <Self as LinOp>::supports_transpose(self)
    }

    #[inline]
    fn t_matvec_s(
        &self,
        x: &[num_complex::Complex64],
        y: &mut [num_complex::Complex64],
        _scratch: &mut BridgeScratch,
    ) {
        <Self as LinOp>::t_matvec(self, x, y);
    }
}

// --- Optional wrappers for dense and CSR matrices -------------------------
#[cfg(feature = "backend-faer")]
use crate::matrix::csc::CscMatrix;
use crate::matrix::sparse::CsrMatrix;
#[cfg(feature = "backend-faer")]
use faer::Mat;

/// Wrap your concrete dense matrix with `DenseOp` to provide stable
/// `StructureId`/`ValuesId` so conversions and preconditioner reuse can be cached.
///
/// Callers that modify the underlying matrix in place must invoke
/// [`mark_structure_changed`] or [`mark_values_changed`] as appropriate so
/// cached conversions/preconditioners can detect the new contents.
#[cfg(feature = "backend-faer")]
pub struct DenseOp<S: KrystScalar> {
    mat: Arc<Mat<S>>,
    ids: ChangeIds,
    comm: UniverseComm,
    layout: Option<DistLayout>,
}
#[cfg(feature = "backend-faer")]
impl<S: KrystScalar> DenseOp<S> {
    /// Wrap a dense matrix so changes can be tracked via [`mark_structure_changed`] and
    /// [`mark_values_changed`]. This enables correct caching and preconditioner reuse across
    /// nonlinear or time-stepping updates.
    pub fn new(mat: Arc<Mat<S>>) -> Self {
        let ids = ChangeIds::default();
        ids.bump_structure();
        ids.bump_values();
        Self {
            mat,
            ids,
            comm: UniverseComm::NoComm(NoComm),
            layout: None,
        }
    }
    /// Attach a communicator to this operator.
    pub fn with_comm(mut self, comm: UniverseComm) -> Self {
        self.comm = comm;
        self
    }
    /// Attach a distributed layout to this operator.
    pub fn with_layout(mut self, layout: DistLayout) -> Self {
        self.layout = Some(layout);
        self
    }
    pub fn mark_structure_changed(&self) {
        self.ids.bump_structure();
    }
    pub fn mark_values_changed(&self) {
        self.ids.bump_values();
    }
    pub fn inner(&self) -> &Mat<S> {
        &self.mat
    }
}
#[cfg(feature = "backend-faer")]
impl<S: KrystScalar> LinOp for DenseOp<S> {
    type S = S;
    fn dims(&self) -> (usize, usize) {
        (self.mat.nrows(), self.mat.ncols())
    }
    fn matvec(&self, x: &[S], y: &mut [S]) {
        if let Err(err) = self.try_matvec(x, y) {
            debug_assert!(false, "DenseOp::matvec dimension mismatch: {err}");
            panic!("{err}");
        }
    }
    fn try_matvec(&self, x: &[S], y: &mut [S]) -> Result<(), KError> {
        if x.len() != self.mat.ncols() || y.len() != self.mat.nrows() {
            return Err(KError::InvalidInput(format!(
                "DenseOp::matvec dimension mismatch: A={}x{}, x.len()={}, y.len()={}",
                self.mat.nrows(),
                self.mat.ncols(),
                x.len(),
                y.len()
            )));
        }
        let cols = self.mat.ncols();
        for (i, yi) in y.iter_mut().enumerate().take(self.mat.nrows()) {
            let mut sum = S::zero();
            for j in 0..cols {
                sum = sum + self.mat[(i, j)] * x[j];
            }
            *yi = sum;
        }
        Ok(())
    }
    fn supports_transpose(&self) -> bool {
        true
    }
    fn t_matvec(&self, x: &[S], y: &mut [S]) {
        if let Err(err) = try_t_matvec_impl("DenseOp::t_matvec", self.mat.as_ref(), x, y) {
            debug_assert!(false, "{err}");
            panic!("{err}");
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
    fn dist_layout(&self) -> Option<&DistLayout> {
        self.layout.as_ref()
    }
    fn format(&self) -> OpFormat {
        OpFormat::Dense
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
/// # unsafe { std::env::set_var("KRYST_THREADS", "16"); }
/// # unsafe { std::env::set_var("KRYST_PAR_CUTOFF", "2048"); }
/// use kryst::matrix::sparse::CsrMatrix;
/// use kryst::matrix::op::CsrOp;
/// use kryst::LinOp;
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
/// - Threaded SpMV honours [`ParallelTune`](crate::algebra::parallel_cfg::ParallelTune)
///   thresholds and falls back to the serial kernel for small systems via
///   the canonical [`crate::matrix::spmv`] entry points.
/// - See [`crate::parallel::threads`] for details on Rayon pool sizing.
/// - After any in-place update to the wrapped CSR matrix, call
///   [`mark_structure_changed`] or [`mark_values_changed`] so caches keyed on
///   [`StructureId`] / [`ValuesId`] stay valid.
#[cfg(feature = "backend-faer")]
pub struct CsrOp<Scalar = S> {
    csr: Arc<CsrMatrix<Scalar>>,
    ids: ChangeIds,
    comm: UniverseComm,
    layout: Option<DistLayout>,
    #[cfg(feature = "transpose-cache")]
    t_cache: parking_lot::RwLock<Option<(ValuesId, Arc<CscMatrix<Scalar>>)>>,
}
#[cfg(feature = "backend-faer")]
impl<Scalar> CsrOp<Scalar> {
    pub fn new(csr: Arc<CsrMatrix<Scalar>>) -> Self {
        let ids = ChangeIds::default();
        ids.bump_structure();
        ids.bump_values();
        Self {
            csr,
            ids,
            comm: UniverseComm::NoComm(NoComm),
            layout: None,
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
    pub fn inner(&self) -> &CsrMatrix<Scalar> {
        &self.csr
    }
    /// Attach a communicator to this operator.
    pub fn with_comm(mut self, comm: UniverseComm) -> Self {
        self.comm = comm;
        self
    }
    /// Attach a distributed layout to this operator.
    pub fn with_layout(mut self, layout: DistLayout) -> Self {
        self.layout = Some(layout);
        self
    }
}
#[cfg(feature = "backend-faer")]
impl<S: KrystScalar> LinOp for CsrOp<S> {
    type S = S;
    fn dims(&self) -> (usize, usize) {
        (self.csr.nrows(), self.csr.ncols())
    }
    fn matvec(&self, x: &[S], y: &mut [S]) {
        if let Err(err) = self.try_matvec(x, y) {
            debug_assert!(false, "CsrOp::matvec dimension mismatch: {err}");
        }
    }
    fn try_matvec(&self, x: &[S], y: &mut [S]) -> Result<(), KError> {
        let (m, n) = (self.csr.nrows(), self.csr.ncols());
        if x.len() != n || y.len() != m {
            return Err(KError::InvalidInput(format!(
                "CsrOp::matvec dimension mismatch: A={}x{}, x.len()={}, y.len()={}",
                m,
                n,
                x.len(),
                y.len()
            )));
        }
        if let Err(err) = crate::matrix::spmv::csr_matvec_par(&*self.csr, x, y) {
            #[cfg(feature = "logging")]
            log::trace!("CsrOp::matvec fallback to serial SpMV: {err}");
            #[cfg(not(feature = "logging"))]
            let _ = &err;
            return crate::matrix::spmv::csr_matvec(&*self.csr, x, y);
        }
        Ok(())
    }
    fn supports_transpose(&self) -> bool {
        true
    }
    fn t_matvec(&self, x: &[S], y: &mut [S]) {
        #[cfg(all(feature = "transpose-cache", not(feature = "complex")))]
        {
            if let Some(csc) = self.ensure_csc_view() {
                let _ = crate::matrix::spmv::csr_t_matvec_par(
                    self.csr.as_ref(),
                    crate::matrix::spmv::TBackend::Csc(csc.as_ref()),
                    x,
                    y,
                );
                return;
            }
        }
        let _ = crate::matrix::spmv::csr_t_matvec_par(
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
    fn dist_layout(&self) -> Option<&DistLayout> {
        self.layout.as_ref()
    }
    fn format(&self) -> OpFormat {
        OpFormat::Csr
    }
}

#[cfg(feature = "backend-faer")]
impl LinOpF64 for CsrOp<f64> {
    #[inline]
    fn dims(&self) -> (usize, usize) {
        <Self as LinOp>::dims(self)
    }

    #[inline]
    fn matvec(&self, x: &[f64], y: &mut [f64]) {
        <Self as LinOp>::matvec(self, x, y)
    }
}

impl LinOpF64 for dyn LinOp<S = f64> + '_ {
    #[inline]
    fn dims(&self) -> (usize, usize) {
        LinOp::dims(self)
    }

    #[inline]
    fn matvec(&self, x: &[f64], y: &mut [f64]) {
        LinOp::matvec(self, x, y)
    }
}

#[cfg(all(feature = "transpose-cache", not(feature = "complex")))]
impl<Scalar: KrystScalar> CsrOp<Scalar> {
    pub fn ensure_csc_view(&self) -> Option<Arc<CscMatrix<Scalar>>> {
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
        let csc = self.csr.to_csc_cached(Scalar::zero().real());
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

#[cfg(feature = "backend-faer")]
fn try_t_matvec_impl<S: KrystScalar>(
    label: &str,
    mat: &Mat<S>,
    x: &[S],
    y: &mut [S],
) -> Result<(), KError> {
    let (m, n) = (mat.nrows(), mat.ncols());
    if x.len() != m || y.len() != n {
        return Err(KError::InvalidInput(format!(
            "{label} dimension mismatch: A={}x{}, x.len()={}, y.len()={}",
            m,
            n,
            x.len(),
            y.len()
        )));
    }
    for j in 0..n {
        let mut sum = S::zero();
        for i in 0..m {
            sum = sum + mat[(i, j)].conj() * x[i];
        }
        y[j] = sum;
    }
    Ok(())
}

/// Generic LinOp implementation for Faer dense matrices.
#[cfg(feature = "backend-faer")]
impl<S: KrystScalar> LinOp for Mat<S> {
    type S = S;
    fn dims(&self) -> (usize, usize) {
        (self.nrows(), self.ncols())
    }
    fn matvec(&self, x: &[S], y: &mut [S]) {
        if let Err(err) = self.try_matvec(x, y) {
            debug_assert!(false, "Mat::matvec dimension mismatch: {err}");
            panic!("{err}");
        }
    }
    fn try_matvec(&self, x: &[S], y: &mut [S]) -> Result<(), KError> {
        let (m, n) = self.dims();
        match (x.len(), y.len()) {
            (nx, my) if nx == n && my == m => {
                for i in 0..m {
                    let mut sum = S::zero();
                    for j in 0..n {
                        sum = sum + self[(i, j)] * x[j];
                    }
                    y[i] = sum;
                }
            }
            (mx, ny) if mx == m && ny == n => {
                for j in 0..n {
                    let mut sum = S::zero();
                    for i in 0..m {
                        sum = sum + self[(i, j)].conj() * x[i];
                    }
                    y[j] = sum;
                }
            }
            (nx, ny) if nx == n && ny == n => {
                let mut tmp = vec![S::zero(); m];
                for i in 0..m {
                    let mut sum = S::zero();
                    for j in 0..n {
                        sum = sum + self[(i, j)] * x[j];
                    }
                    tmp[i] = sum;
                }
                for j in 0..n {
                    let mut sum = S::zero();
                    for i in 0..m {
                        sum = sum + self[(i, j)].conj() * tmp[i];
                    }
                    y[j] = sum;
                }
            }
            (mx, my) if mx == m && my == m => {
                let mut tmp = vec![S::zero(); n];
                for j in 0..n {
                    let mut sum = S::zero();
                    for i in 0..m {
                        sum = sum + self[(i, j)].conj() * x[i];
                    }
                    tmp[j] = sum;
                }
                for i in 0..m {
                    let mut sum = S::zero();
                    for j in 0..n {
                        sum = sum + self[(i, j)] * tmp[j];
                    }
                    y[i] = sum;
                }
            }
            (lx, ly) => {
                return Err(KError::InvalidInput(format!(
                    "Mat::matvec dimension mismatch: A is {}x{}, x.len() = {}, y.len() = {}",
                    m, n, lx, ly
                )));
            }
        }
        Ok(())
    }
    fn supports_transpose(&self) -> bool {
        true
    }
    fn t_matvec(&self, x: &[S], y: &mut [S]) {
        if let Err(err) = try_t_matvec_impl("Mat::t_matvec", self, x, y) {
            debug_assert!(false, "{err}");
            panic!("{err}");
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
            // Note: complex values don't have to_bits, so this only works for real
            #[cfg(not(feature = "complex"))]
            for i in 0..m {
                for j in 0..n {
                    self[(i, j)].to_bits().hash(&mut h);
                }
            }
            #[cfg(feature = "complex")]
            {
                // For complex, use a simple fallback: 0 (unknown)
                // Higher layers may use pointer identity instead
            }
            ValuesId(h.finish())
        }
        #[cfg(not(feature = "mat-values-fingerprint"))]
        {
            ValuesId(0)
        }
    }
    fn format(&self) -> OpFormat {
        OpFormat::Dense
    }
}

#[cfg(feature = "backend-faer")]
impl LinOpF64 for Mat<f64> {
    #[inline]
    fn dims(&self) -> (usize, usize) {
        <Self as LinOp>::dims(self)
    }

    #[inline]
    fn matvec(&self, x: &[f64], y: &mut [f64]) {
        <Self as LinOp>::matvec(self, x, y)
    }
}

impl<S: KrystScalar> LinOp for CsrMatrix<S> {
    type S = S;
    fn dims(&self) -> (usize, usize) {
        (self.nrows(), self.ncols())
    }
    fn matvec(&self, x: &[S], y: &mut [S]) {
        self.spmv(x, y);
    }
    fn try_matvec(&self, x: &[S], y: &mut [S]) -> Result<(), KError> {
        self.try_spmv(x, y)
    }
    fn supports_transpose(&self) -> bool {
        true
    }
    fn t_matvec(&self, x: &[S], y: &mut [S]) {
        if let Err(err) = self.spmv_transpose_scaled(S::one(), x, S::zero(), y) {
            debug_assert!(false, "CsrMatrix::t_matvec dimension mismatch: {err}");
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
    fn format(&self) -> OpFormat {
        OpFormat::Csr
    }
}

impl LinOpF64 for CsrMatrix<f64> {
    #[inline]
    fn dims(&self) -> (usize, usize) {
        <Self as LinOp>::dims(self)
    }

    #[inline]
    fn matvec(&self, x: &[f64], y: &mut [f64]) {
        <Self as LinOp>::matvec(self, x, y)
    }
}

#[cfg(feature = "backend-faer")]
impl<S: KrystScalar> LinOp for CscMatrix<S> {
    type S = S;

    fn dims(&self) -> (usize, usize) {
        (self.nrows(), self.ncols())
    }

    fn matvec(&self, x: &[S], y: &mut [S]) {
        self.spmv(x, y);
    }

    fn supports_transpose(&self) -> bool {
        true
    }

    fn t_matvec(&self, x: &[S], y: &mut [S]) {
        CscMatrix::t_matvec(self, x, y);
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
    fn format(&self) -> OpFormat {
        OpFormat::Csc
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
    fn format(&self) -> OpFormat {
        self.inner.format()
    }
    #[inline]
    fn comm(&self) -> UniverseComm {
        self.comm.clone()
    }
    fn dist_layout(&self) -> Option<&DistLayout> {
        self.inner.dist_layout()
    }
    fn halo_exchange(&self) -> Option<&dyn HaloExchange<Self::S>> {
        self.inner.halo_exchange()
    }
}

/// Ergonomic helper for call sites
pub fn wrap_with_comm<T>(op: Arc<T>, comm: UniverseComm) -> Arc<dyn LinOp<S = T::S>>
where
    T: LinOp + ?Sized + 'static,
{
    Arc::new(WithCommOp::new(op, comm)) as Arc<dyn LinOp<S = T::S>>
}

#[cfg(all(test, feature = "backend-faer"))]
mod tests {
    use super::*;
    use crate::error::KError;
    use crate::matrix::sparse::CsrMatrix as RealCsrMatrix;
    use crate::matrix::spmv::scalar::spmv_csr_scalar;

    #[test]
    fn generic_csr_op_matches_scalar_kernel() {
        let matrix = Arc::new(ScalarCsrMatrix::new(
            3,
            3,
            vec![0, 2, 4, 5],
            vec![0, 2, 1, 2, 0],
            vec![1.0, -2.0, 3.5, 0.5, 4.0],
        ));
        let tuning = SpmvTuning {
            allow_simd: false,
            ..Default::default()
        };
        let op = GenericCsrOp::new(matrix.clone(), &tuning);

        let x = vec![0.75, -1.25, 2.0];
        let (m, _) = matrix.dims();
        let mut y = vec![0.0; m];
        LinOp::matvec(&op, &x, &mut y);

        let mut y_ref = vec![0.0; m];
        spmv_csr_scalar(matrix.as_ref(), &x, &mut y_ref);

        for (lhs, rhs) in y.iter().zip(y_ref.iter()) {
            assert!((lhs - rhs).abs() < 1e-12);
        }
    }

    #[test]
    fn generic_csr_op_from_real_csr_matches_matrix() {
        let real = RealCsrMatrix::from_csr(
            3,
            3,
            vec![0, 2, 4, 5],
            vec![0, 2, 1, 2, 0],
            vec![1.0, -2.0, 3.5, 0.5, 4.0],
        );
        let tuning = SpmvTuning {
            allow_simd: false,
            ..Default::default()
        };
        let op = GenericCsrOp::<f64>::from_real_csr(&real, &tuning);

        let x = vec![0.75, -1.25, 2.0];
        let mut y = vec![0.0; real.nrows()];
        LinOp::matvec(&op, &x, &mut y);

        let mut y_ref = vec![0.0; real.nrows()];
        real.spmv_scaled(1.0, &x, 0.0, &mut y_ref)
            .expect("real CSR spmv");

        for (lhs, rhs) in y.iter().zip(y_ref.iter()) {
            assert!((lhs - rhs).abs() < 1e-12);
        }
    }

    #[cfg(feature = "complex")]
    #[test]
    fn generic_csr_op_complex_matches_scalar_kernel() {
        use num_complex::Complex64;

        let matrix = Arc::new(ScalarCsrMatrix::new(
            2,
            3,
            vec![0, 2, 3],
            vec![0, 1, 2],
            vec![S::from_real(1.0), S::from_real(-0.5), S::from_real(2.25)],
        ));
        let tuning = SpmvTuning {
            allow_simd: false,
            ..Default::default()
        };
        let op = GenericCsrOp::new(matrix.clone(), &tuning);

        let x: Vec<S> = vec![
            S::from_real(1.0),
            S::from_real(-2.0),
            Complex64::new(0.5, 0.75),
        ];
        let (m, _) = matrix.dims();
        let mut y = vec![S::zero(); m];
        LinOp::matvec(&op, &x, &mut y);

        let mut y_ref = vec![S::zero(); m];
        spmv_csr_scalar(matrix.as_ref(), &x, &mut y_ref);

        for (lhs, rhs) in y.iter().zip(y_ref.iter()) {
            assert!((lhs - rhs).abs() < 1e-12);
        }
    }

    #[cfg(feature = "complex")]
    #[test]
    fn generic_csr_op_complex_from_real_csr_matches_scalar_kernel() {
        use num_complex::Complex64;

        let real =
            RealCsrMatrix::from_csr(2, 3, vec![0, 2, 3], vec![0, 1, 2], vec![1.0, -0.5, 2.25]);
        let tuning = SpmvTuning {
            allow_simd: false,
            ..Default::default()
        };
        let op = GenericCsrOp::<S>::from_real_csr(&real, &tuning);

        let x: Vec<S> = vec![
            S::from_real(1.0),
            S::from_parts(-2.0, 0.25),
            Complex64::new(0.5, 0.75),
        ];
        let mut y = vec![S::zero(); real.nrows()];
        LinOp::matvec(&op, &x, &mut y);

        let mut y_ref = vec![S::zero(); real.nrows()];
        spmv_csr_scalar(op.matrix(), &x, &mut y_ref);

        for (lhs, rhs) in y.iter().zip(y_ref.iter()) {
            assert!((lhs - rhs).abs() < 1e-12);
        }
    }

    #[test]
    fn csr_op_try_matvec_reports_dim_mismatch() {
        let csr = Arc::new(CsrMatrix::identity(2));
        let op = CsrOp::new(csr);
        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![0.0; 2];
        let err = op.try_matvec(&x, &mut y).unwrap_err();
        assert!(matches!(err, KError::InvalidInput(_)));
    }

    #[test]
    fn dense_op_reports_dim_mismatch() {
        let mat = Arc::new(Mat::<f64>::zeros(2, 3));
        let op = DenseOp::new(mat.clone());
        let x = vec![1.0; 4];
        let mut y = vec![0.0; 2];
        let err = op.try_matvec(&x, &mut y).unwrap_err();
        assert!(matches!(err, KError::InvalidInput(_)));

        let x = vec![1.0; 2];
        let mut y = vec![0.0; 2];
        let err = try_t_matvec_impl("DenseOp::t_matvec", mat.as_ref(), &x, &mut y).unwrap_err();
        assert!(matches!(err, KError::InvalidInput(_)));
    }

    #[test]
    fn mat_reports_dim_mismatch() {
        let mat = Mat::<f64>::zeros(2, 3);
        let x = vec![1.0; 3];
        let mut y = vec![0.0; 1];
        let err = LinOp::try_matvec(&mat, &x, &mut y).unwrap_err();
        assert!(matches!(err, KError::InvalidInput(_)));

        let x = vec![1.0; 2];
        let mut y = vec![0.0; 2];
        let err = try_t_matvec_impl("Mat::t_matvec", &mat, &x, &mut y).unwrap_err();
        assert!(matches!(err, KError::InvalidInput(_)));
    }
}
