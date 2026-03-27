//! Backend-aware format conversion traits and caches.
//!
//! Backend implementations advertise their matrix-format coverage via
//! [`BackendFormatSupport`]. This keeps format conversion expectations explicit
//! when adding new backends—update the backend's `FORMAT_SUPPORT` and ensure
//! the `AsFormat` implementations cover the advertised formats.
//! NOTE: Every conversion cache key includes the operator's [`StructureId`]
//! and [`ValuesId`]. A value of `0` indicates "unknown" and disables precise
//! invalidation, so wrappers like [`DenseOp`] / [`CsrOp`] are recommended when
//! you mutate matrices in-place and need cache stability.
use std::any::Any;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex, Weak};

use once_cell::sync::Lazy;

use crate::algebra::scalar::KrystScalar;
use crate::matrix::backend::SparseBackend;
use crate::matrix::op::{StructureId, ValuesId};

/// Backend-neutral operator formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpFormat {
    Any,
    Dense,
    Csr,
    Csc,
    BlockCsr,
}

/// Declares which operator formats a backend can materialize.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackendFormatSupport {
    pub dense: bool,
    pub csr: bool,
    pub csc: bool,
    pub block_csr: bool,
}

impl BackendFormatSupport {
    pub const fn new(dense: bool, csr: bool, csc: bool, block_csr: bool) -> Self {
        Self {
            dense,
            csr,
            csc,
            block_csr,
        }
    }

    pub const fn supports(self, format: OpFormat) -> bool {
        match format {
            OpFormat::Dense => self.dense,
            OpFormat::Csr => self.csr,
            OpFormat::Csc => self.csc,
            OpFormat::BlockCsr => self.block_csr,
            OpFormat::Any => true,
        }
    }
}

impl OpFormat {
    #[inline]
    pub fn is_any(self) -> bool {
        matches!(self, OpFormat::Any)
    }
}

/// Backend-specific format hints used by conversion helpers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FormatHint {
    Csr,
    Dense,
    Csc,
}

impl From<FormatHint> for OpFormat {
    fn from(hint: FormatHint) -> Self {
        match hint {
            FormatHint::Csr => OpFormat::Csr,
            FormatHint::Dense => OpFormat::Dense,
            FormatHint::Csc => OpFormat::Csc,
        }
    }
}

impl OpFormat {
    pub fn to_format_hint(self) -> Option<FormatHint> {
        match self {
            OpFormat::Csr => Some(FormatHint::Csr),
            OpFormat::Dense => Some(FormatHint::Dense),
            OpFormat::Csc => Some(FormatHint::Csc),
            OpFormat::Any | OpFormat::BlockCsr => None,
        }
    }
}

/// Trait for converting matrices into specific formats under a backend.
pub trait AsFormat<S: KrystScalar, B: SparseBackend<S>> {
    /// Borrow as CSR if already in that format.
    fn as_csr(&self) -> Option<&B::Csr> {
        None
    }

    /// Convert to CSR and cache the result.
    fn to_csr_cached(&self, drop_tol: S::Real) -> Arc<B::Csr>;

    /// Borrow as CSC if already in that format.
    fn as_csc(&self) -> Option<&B::Csc> {
        None
    }

    /// Convert to CSC and cache the result.
    fn to_csc_cached(&self, drop_tol: S::Real) -> Arc<B::Csc>;

    /// Identifier for structure-driven cache invalidation.
    ///
    /// Returning `StructureId(0)` means "unknown" and falls back to pointer identity
    /// based caches. Wrap your matrices with [`DenseOp`] / [`CsrOp`] and call
    /// [`mark_structure_changed`] when mutating the sparsity pattern if you need
    /// accurate cache keys.
    fn structure_id_for_cache(&self) -> StructureId {
        StructureId(0)
    }

    /// Identifier for value-driven cache invalidation.
    ///
    /// Value changes should bump the ID via [`mark_values_changed`] to keep
    /// cached conversions valid. A `ValuesId(0)` disables precise invalidation
    /// (e.g., raw `faer::Mat` without `mat-values-fingerprint`).
    fn values_id_for_cache(&self) -> ValuesId {
        ValuesId(0)
    }

    /// Advertise which formats the backend can materialize for this scalar.
    fn backend_format_support() -> BackendFormatSupport {
        B::FORMAT_SUPPORT
    }
}

/// Convenience alias for the active backend's [`AsFormat`] trait object.
#[cfg(feature = "backend-faer")]
pub type DefaultAsFormat<S> = dyn AsFormat<S, crate::matrix::backend::DefaultBackend>;

/// Backend-agnostic cache key for format conversions.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FormatKey {
    pub base_ptr: usize,
    pub structure_id: u64,
    pub values_id: u64,
    pub drop_tol_bits: u64,
}

impl PartialEq for FormatKey {
    fn eq(&self, other: &Self) -> bool {
        self.base_ptr == other.base_ptr
            && self.structure_id == other.structure_id
            && self.values_id == other.values_id
            && self.drop_tol_bits == other.drop_tol_bits
    }
}
impl Eq for FormatKey {}
impl Hash for FormatKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.base_ptr.hash(state);
        self.structure_id.hash(state);
        self.values_id.hash(state);
        self.drop_tol_bits.hash(state);
    }
}

#[inline]
pub(crate) fn format_key_from_ptr(
    base_ptr: usize,
    structure_id: StructureId,
    values_id: ValuesId,
    drop_tol: f64,
) -> FormatKey {
    FormatKey {
        base_ptr,
        structure_id: structure_id.0,
        values_id: values_id.0,
        drop_tol_bits: drop_tol.to_bits(),
    }
}

type CacheMap = HashMap<FormatKey, Weak<dyn Any + Send + Sync>>;

/// Global cache of CSR conversions keyed by [`FormatKey`].
pub(crate) static CSR_CACHE: Lazy<Mutex<CacheMap>> = Lazy::new(|| Mutex::new(HashMap::new()));

/// Global cache of CSC conversions keyed by [`FormatKey`].
pub(crate) static CSC_CACHE: Lazy<Mutex<CacheMap>> = Lazy::new(|| Mutex::new(HashMap::new()));

#[inline]
pub(crate) fn get_or_insert_csr<T: 'static + Send + Sync>(
    key: FormatKey,
    build: impl FnOnce() -> Arc<T>,
) -> Arc<T> {
    if let Some(existing) = CSR_CACHE
        .lock()
        .unwrap()
        .get(&key)
        .and_then(|w| w.upgrade())
        && let Ok(typed) = existing.downcast::<T>() {
            return typed;
        }

    let arc: Arc<T> = build();
    let erased: Arc<dyn Any + Send + Sync> = arc.clone();
    let mut cache = CSR_CACHE.lock().unwrap();
    cache.insert(key, Arc::downgrade(&erased));
    arc
}

#[inline]
pub(crate) fn get_or_insert_csc<T: 'static + Send + Sync>(
    key: FormatKey,
    build: impl FnOnce() -> Arc<T>,
) -> Arc<T> {
    if let Some(existing) = CSC_CACHE
        .lock()
        .unwrap()
        .get(&key)
        .and_then(|w| w.upgrade())
        && let Ok(typed) = existing.downcast::<T>() {
            return typed;
        }

    let arc: Arc<T> = build();
    let erased: Arc<dyn Any + Send + Sync> = arc.clone();
    let mut cache = CSC_CACHE.lock().unwrap();
    cache.insert(key, Arc::downgrade(&erased));
    arc
}
