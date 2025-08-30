use once_cell::sync::Lazy;
use std::{
    collections::HashMap,
    hash::{Hash, Hasher},
    sync::{Arc, Mutex, Weak},
};

use crate::matrix::{csc::CscMatrix, sparse::CsrMatrix};

/// High-level format hints that preconditioners can request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FormatHint {
    Csr,
    Dense,
    Csc,
}

/// Trait for converting matrices into specific formats.
pub trait AsFormat {
    /// Borrow as CSR if already in that format.
    fn as_csr(&self) -> Option<&CsrMatrix<f64>> {
        None
    }

    /// Convert to CSR and cache the result.
    fn to_csr_cached(&self, drop_tol: f64) -> Arc<CsrMatrix<f64>>;

    /// Borrow as CSC if already in that format.
    fn as_csc(&self) -> Option<&CscMatrix<f64>> {
        None
    }

    /// Convert to CSC and cache the result.
    fn to_csc_cached(&self, drop_tol: f64) -> Arc<CscMatrix<f64>>;
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct CsrKey {
    pub base_ptr: usize,
    pub structure_id: u64,
    pub values_id: u64,
    pub drop_tol_bits: u64,
}

impl PartialEq for CsrKey {
    fn eq(&self, other: &Self) -> bool {
        self.base_ptr == other.base_ptr
            && self.structure_id == other.structure_id
            && self.values_id == other.values_id
            && self.drop_tol_bits == other.drop_tol_bits
    }
}
impl Eq for CsrKey {}
impl Hash for CsrKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.base_ptr.hash(state);
        self.structure_id.hash(state);
        self.values_id.hash(state);
        self.drop_tol_bits.hash(state);
    }
}

/// Global cache of dense->CSR conversions.
pub(crate) static CSR_CACHE: Lazy<Mutex<HashMap<CsrKey, Weak<CsrMatrix<f64>>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

pub(crate) fn key_from_ptr(ptr: usize, structure_id: u64, values_id: u64, drop_tol: f64) -> CsrKey {
    CsrKey {
        base_ptr: ptr,
        structure_id,
        values_id,
        drop_tol_bits: drop_tol.to_bits(),
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct CscKey {
    pub base_ptr: usize,
    pub structure_id: u64,
    pub values_id: u64,
    pub drop_tol_bits: u64,
}

impl PartialEq for CscKey {
    fn eq(&self, other: &Self) -> bool {
        self.base_ptr == other.base_ptr
            && self.structure_id == other.structure_id
            && self.values_id == other.values_id
            && self.drop_tol_bits == other.drop_tol_bits
    }
}
impl Eq for CscKey {}
impl Hash for CscKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.base_ptr.hash(state);
        self.structure_id.hash(state);
        self.values_id.hash(state);
        self.drop_tol_bits.hash(state);
    }
}

/// Global cache of dense->CSC conversions.
pub(crate) static CSC_CACHE: Lazy<Mutex<HashMap<CscKey, Weak<CscMatrix<f64>>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

pub(crate) fn csc_key_from_ptr(
    ptr: usize,
    structure_id: u64,
    values_id: u64,
    drop_tol: f64,
) -> CscKey {
    CscKey {
        base_ptr: ptr,
        structure_id,
        values_id,
        drop_tol_bits: drop_tol.to_bits(),
    }
}
