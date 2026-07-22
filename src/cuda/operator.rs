use super::runtime::{CudaRuntime, CudaSpmvAlgorithm, cuda_error, map_driver, status_to_result};
use super::vector::{CudaVector, DeviceBuffer, DeviceScalar, host_to_device};
use crate::algebra::prelude::*;
use crate::error::{CudaErrorKind, KError};
use crate::matrix::op::{StructureId, ValuesId};
use crate::matrix::sparse::CsrMatrix;
use crate::parallel::UniverseComm;
use cudarc::cusparse;
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use std::any::Any;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CudaOperation {
    #[default]
    NonTranspose,
    Transpose,
    ConjugateTranspose,
}

pub trait CudaLinOp: Send + Sync + Any {
    fn dims(&self) -> (usize, usize);
    fn apply(
        &self,
        operation: CudaOperation,
        x: &CudaVector,
        y: &mut CudaVector,
    ) -> Result<(), KError>;
    fn prepare(&self) -> Result<(), KError> {
        Ok(())
    }
    fn supports_transpose(&self) -> bool {
        false
    }
    fn device_ordinal(&self) -> usize;
    /// Communicator for rank-global scalar reductions. Ordinary single-device
    /// operators return `None`; distributed operators return their rank group.
    fn communicator(&self) -> Option<&UniverseComm> {
        None
    }
    fn structure_id(&self) -> StructureId {
        StructureId(0)
    }
    fn values_id(&self) -> ValuesId {
        ValuesId(0)
    }
    fn as_any(&self) -> &dyn Any;
}

enum DeviceIndices {
    I32 {
        rows: CudaSlice<i32>,
        cols: CudaSlice<i32>,
    },
    I64 {
        rows: CudaSlice<i64>,
        cols: CudaSlice<i64>,
    },
}

impl DeviceIndices {
    fn pointers(&self, runtime: &CudaRuntime) -> (u64, u64) {
        match self {
            DeviceIndices::I32 { rows, cols } => {
                let (rows, _rr) = rows.device_ptr(runtime.stream());
                let (cols, _cr) = cols.device_ptr(runtime.stream());
                (rows, cols)
            }
            DeviceIndices::I64 { rows, cols } => {
                let (rows, _rr) = rows.device_ptr(runtime.stream());
                let (cols, _cr) = cols.device_ptr(runtime.stream());
                (rows, cols)
            }
        }
    }

    fn index_type(&self) -> cusparse::sys::cusparseIndexType_t {
        match self {
            DeviceIndices::I32 { .. } => cusparse::sys::cusparseIndexType_t::CUSPARSE_INDEX_32I,
            DeviceIndices::I64 { .. } => cusparse::sys::cusparseIndexType_t::CUSPARSE_INDEX_64I,
        }
    }
}

struct SpMatDescriptor {
    raw: cusparse::sys::cusparseSpMatDescr_t,
    runtime: Arc<CudaRuntime>,
}

unsafe impl Send for SpMatDescriptor {}
unsafe impl Sync for SpMatDescriptor {}

impl Drop for SpMatDescriptor {
    fn drop(&mut self) {
        let _ = self.runtime.stream().context().bind_to_thread();
        unsafe {
            let _ = cusparse::sys::cusparseDestroySpMat(self.raw);
        }
    }
}

struct SpmvResources {
    x: cusparse::sys::cusparseDnVecDescr_t,
    y: cusparse::sys::cusparseDnVecDescr_t,
    buffer: Option<CudaSlice<u8>>,
    runtime: Arc<CudaRuntime>,
}

unsafe impl Send for SpmvResources {}

impl Drop for SpmvResources {
    fn drop(&mut self) {
        let _ = self.runtime.stream().context().bind_to_thread();
        unsafe {
            let _ = cusparse::sys::cusparseDestroyDnVec(self.x);
            let _ = cusparse::sys::cusparseDestroyDnVec(self.y);
        }
    }
}

#[derive(Default)]
struct SpmvCache {
    forward: Option<SpmvResources>,
    transpose: Option<SpmvResources>,
}

pub struct CudaCsrOp {
    // Descriptor must be destroyed before its backing allocations.
    descriptor: SpMatDescriptor,
    _indices: DeviceIndices,
    values: Mutex<DeviceBuffer>,
    runtime: Arc<CudaRuntime>,
    nrows: usize,
    ncols: usize,
    nnz: usize,
    row_offsets_host: Vec<usize>,
    col_indices_host: Vec<usize>,
    values_host: RwLock<Vec<S>>,
    diagonal_host: RwLock<Vec<S>>,
    cache: Mutex<SpmvCache>,
    structure_id: AtomicU64,
    values_id: AtomicU64,
}

impl std::fmt::Debug for CudaCsrOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaCsrOp")
            .field("dims", &(self.nrows, self.ncols))
            .field("nnz", &self.nnz)
            .field("device_ordinal", &self.device_ordinal())
            .field("structure_id", &self.structure_id())
            .field("values_id", &self.values_id())
            .finish()
    }
}

impl CudaCsrOp {
    pub fn from_host(runtime: Arc<CudaRuntime>, matrix: &CsrMatrix<S>) -> Result<Self, KError> {
        Self::from_csr_parts(
            runtime,
            matrix.nrows(),
            matrix.ncols(),
            matrix.row_ptr(),
            matrix.col_idx(),
            matrix.values(),
        )
    }

    pub fn from_csr_parts(
        runtime: Arc<CudaRuntime>,
        nrows: usize,
        ncols: usize,
        row_offsets: &[usize],
        col_indices: &[usize],
        values: &[S],
    ) -> Result<Self, KError> {
        validate_csr(nrows, ncols, row_offsets, col_indices, values)?;
        let use_i32 = nrows <= i32::MAX as usize
            && ncols <= i32::MAX as usize
            && values.len() <= i32::MAX as usize
            && row_offsets.iter().all(|&v| v <= i32::MAX as usize)
            && col_indices.iter().all(|&v| v <= i32::MAX as usize);

        let indices = if use_i32 {
            let rows: Vec<i32> = row_offsets.iter().map(|&v| v as i32).collect();
            let cols: Vec<i32> = col_indices.iter().map(|&v| v as i32).collect();
            let rows = runtime
                .stream()
                .clone_htod(&rows)
                .map_err(|e| map_driver("upload CUDA CSR row offsets", e))?;
            let cols = runtime
                .stream()
                .clone_htod(&cols)
                .map_err(|e| map_driver("upload CUDA CSR column indices", e))?;
            runtime
                .diagnostics_ref()
                .allocation(row_offsets.len() * std::mem::size_of::<i32>());
            runtime
                .diagnostics_ref()
                .allocation(col_indices.len() * std::mem::size_of::<i32>());
            runtime
                .diagnostics_ref()
                .htod(row_offsets.len() * std::mem::size_of::<i32>());
            runtime
                .diagnostics_ref()
                .htod(col_indices.len() * std::mem::size_of::<i32>());
            DeviceIndices::I32 { rows, cols }
        } else {
            let rows: Vec<i64> = row_offsets
                .iter()
                .map(|&v| i64::try_from(v))
                .collect::<Result<_, _>>()
                .map_err(|_| KError::InvalidInput("CSR row offset exceeds i64".into()))?;
            let cols: Vec<i64> = col_indices
                .iter()
                .map(|&v| i64::try_from(v))
                .collect::<Result<_, _>>()
                .map_err(|_| KError::InvalidInput("CSR column index exceeds i64".into()))?;
            let rows = runtime
                .stream()
                .clone_htod(&rows)
                .map_err(|e| map_driver("upload CUDA CSR row offsets", e))?;
            let cols = runtime
                .stream()
                .clone_htod(&cols)
                .map_err(|e| map_driver("upload CUDA CSR column indices", e))?;
            runtime
                .diagnostics_ref()
                .allocation(row_offsets.len() * std::mem::size_of::<i64>());
            runtime
                .diagnostics_ref()
                .allocation(col_indices.len() * std::mem::size_of::<i64>());
            runtime
                .diagnostics_ref()
                .htod(row_offsets.len() * std::mem::size_of::<i64>());
            runtime
                .diagnostics_ref()
                .htod(col_indices.len() * std::mem::size_of::<i64>());
            DeviceIndices::I64 { rows, cols }
        };

        let converted = host_to_device(values);
        let device_values = runtime
            .stream()
            .clone_htod(converted.as_ref())
            .map_err(|e| map_driver("upload CUDA CSR values", e))?;
        runtime
            .diagnostics_ref()
            .allocation(values.len() * std::mem::size_of::<DeviceScalar>());
        runtime
            .diagnostics_ref()
            .htod(values.len() * std::mem::size_of::<DeviceScalar>());

        let (row_ptr, col_ptr) = indices.pointers(&runtime);
        let raw = {
            let (value_ptr, _vr) = device_values.device_ptr(runtime.stream());
            let mut raw = std::ptr::null_mut();
            unsafe {
                status_to_result(
                    cusparse::sys::cusparseCreateCsr(
                        &mut raw,
                        nrows as i64,
                        ncols as i64,
                        values.len() as i64,
                        row_ptr as *mut c_void,
                        col_ptr as *mut c_void,
                        value_ptr as *mut c_void,
                        indices.index_type(),
                        indices.index_type(),
                        cusparse::sys::cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO,
                        cuda_data_type(),
                    ),
                    "create cuSPARSE CSR descriptor",
                )?;
            }
            raw
        };
        runtime.diagnostics_ref().library_call();

        Ok(Self {
            descriptor: SpMatDescriptor {
                raw,
                runtime: runtime.clone(),
            },
            _indices: indices,
            values: Mutex::new(device_values),
            runtime,
            nrows,
            ncols,
            nnz: values.len(),
            row_offsets_host: row_offsets.to_vec(),
            col_indices_host: col_indices.to_vec(),
            values_host: RwLock::new(values.to_vec()),
            diagonal_host: RwLock::new(extract_diagonal(nrows, row_offsets, col_indices, values)),
            cache: Mutex::new(SpmvCache::default()),
            structure_id: AtomicU64::new(1),
            values_id: AtomicU64::new(1),
        })
    }

    pub fn runtime(&self) -> &Arc<CudaRuntime> {
        &self.runtime
    }

    pub fn nnz(&self) -> usize {
        self.nnz
    }

    pub fn diagonal_host(&self) -> Result<Vec<S>, KError> {
        self.diagonal_host
            .read()
            .map(|diagonal| diagonal.clone())
            .map_err(|_| {
                cuda_error(
                    CudaErrorKind::Library,
                    "read CUDA CSR diagonal",
                    "diagonal cache lock was poisoned",
                )
            })
    }

    pub(crate) fn host_csr_parts(&self) -> Result<(Vec<usize>, Vec<usize>, Vec<S>), KError> {
        let values = self
            .values_host
            .read()
            .map_err(|_| {
                cuda_error(
                    CudaErrorKind::Library,
                    "read CUDA CSR host values",
                    "host value cache lock was poisoned",
                )
            })?
            .clone();
        Ok((
            self.row_offsets_host.clone(),
            self.col_indices_host.clone(),
            values,
        ))
    }

    pub(crate) fn raw_descriptor(&self) -> cusparse::sys::cusparseSpMatDescr_t {
        self.descriptor.raw
    }

    /// Upload new numeric values without rebuilding the CSR structure or its
    /// cuSPARSE descriptor. Existing `Arc<CudaCsrOp>` owners observe the new
    /// [`ValuesId`] on their next setup/solve.
    pub fn update_values(&self, values: &[S]) -> Result<(), KError> {
        if values.len() != self.nnz {
            return Err(KError::InvalidInput(format!(
                "CUDA CSR numeric update length mismatch: {} vs {}",
                values.len(),
                self.nnz
            )));
        }
        let converted = host_to_device(values);
        let mut device_values = self.values.lock().map_err(|_| {
            cuda_error(
                CudaErrorKind::Library,
                "lock CUDA CSR values",
                "numeric value buffer lock was poisoned",
            )
        })?;
        self.runtime
            .stream()
            .memcpy_htod(converted.as_ref(), &mut *device_values)
            .map_err(|e| map_driver("update CUDA CSR values", e))?;
        self.runtime
            .diagnostics_ref()
            .htod(values.len() * std::mem::size_of::<DeviceScalar>());
        let mut diagonal = self.diagonal_host.write().map_err(|_| {
            cuda_error(
                CudaErrorKind::Library,
                "update CUDA CSR diagonal",
                "diagonal cache lock was poisoned",
            )
        })?;
        *diagonal = extract_diagonal(
            self.nrows,
            &self.row_offsets_host,
            &self.col_indices_host,
            values,
        );
        let mut host_values = self.values_host.write().map_err(|_| {
            cuda_error(
                CudaErrorKind::Library,
                "update CUDA CSR host values",
                "host value cache lock was poisoned",
            )
        })?;
        host_values.copy_from_slice(values);
        self.values_id.fetch_add(1, Ordering::Release);
        Ok(())
    }

    fn resources(&self, operation: CudaOperation) -> Result<(), KError> {
        let mut cache = self.cache.lock().map_err(|_| {
            cuda_error(
                CudaErrorKind::Library,
                "lock CUDA SpMV cache",
                "SpMV cache mutex was poisoned",
            )
        })?;
        let slot = match operation {
            CudaOperation::NonTranspose => &mut cache.forward,
            CudaOperation::Transpose | CudaOperation::ConjugateTranspose => &mut cache.transpose,
        };
        if slot.is_none() {
            *slot = Some(self.build_resources(operation)?);
        }
        Ok(())
    }

    fn build_resources(&self, operation: CudaOperation) -> Result<SpmvResources, KError> {
        let (x_len, y_len) = match operation {
            CudaOperation::NonTranspose => (self.ncols, self.nrows),
            CudaOperation::Transpose | CudaOperation::ConjugateTranspose => {
                (self.nrows, self.ncols)
            }
        };
        let x_dummy = CudaVector::zeros(self.runtime.clone(), x_len)?;
        let mut y_dummy = CudaVector::zeros(self.runtime.clone(), y_len)?;
        let (xp, _xr) = x_dummy.buffer().device_ptr(self.runtime.stream());
        let (yp, _yw) = y_dummy.buffer_mut().device_ptr_mut(self.runtime.stream());
        let mut x_desc = std::ptr::null_mut();
        let mut y_desc = std::ptr::null_mut();
        unsafe {
            status_to_result(
                cusparse::sys::cusparseCreateDnVec(
                    &mut x_desc,
                    x_len as i64,
                    xp as *mut c_void,
                    cuda_data_type(),
                ),
                "create cuSPARSE input vector descriptor",
            )?;
            if let Err(error) = status_to_result(
                cusparse::sys::cusparseCreateDnVec(
                    &mut y_desc,
                    y_len as i64,
                    yp as *mut c_void,
                    cuda_data_type(),
                ),
                "create cuSPARSE output vector descriptor",
            ) {
                let _ = cusparse::sys::cusparseDestroyDnVec(x_desc);
                return Err(error);
            }
        }

        let alpha = host_scalar(S::one());
        let beta = host_scalar(S::zero());
        let mut buffer_size = 0usize;
        let handles = self.runtime.handles()?;
        unsafe {
            status_to_result(
                cusparse::sys::cusparseSpMV_bufferSize(
                    handles.cusparse,
                    operation.into_sys(),
                    alpha.as_ptr(),
                    self.descriptor.raw,
                    x_desc,
                    beta.as_ptr(),
                    y_desc,
                    cuda_data_type(),
                    self.spmv_algorithm(),
                    &mut buffer_size,
                ),
                "query cuSPARSE SpMV workspace",
            )?;
        }
        drop(handles);
        self.runtime.diagnostics_ref().library_call();
        let buffer = if buffer_size == 0 {
            None
        } else {
            let buffer = self
                .runtime
                .stream()
                .alloc_zeros::<u8>(buffer_size)
                .map_err(|e| map_driver("allocate cuSPARSE SpMV workspace", e))?;
            self.runtime.diagnostics_ref().allocation(buffer_size);
            Some(buffer)
        };
        Ok(SpmvResources {
            x: x_desc,
            y: y_desc,
            buffer,
            runtime: self.runtime.clone(),
        })
    }

    fn spmv_algorithm(&self) -> cusparse::sys::cusparseSpMVAlg_t {
        match (
            self.runtime.options().deterministic,
            self.runtime.options().spmv_algorithm,
        ) {
            (true, _) => cusparse::sys::cusparseSpMVAlg_t::CUSPARSE_SPMV_CSR_ALG2,
            (_, CudaSpmvAlgorithm::Deterministic) => {
                cusparse::sys::cusparseSpMVAlg_t::CUSPARSE_SPMV_CSR_ALG2
            }
            (_, CudaSpmvAlgorithm::Auto | CudaSpmvAlgorithm::Fast) => {
                cusparse::sys::cusparseSpMVAlg_t::CUSPARSE_SPMV_CSR_ALG1
            }
        }
    }
}

impl CudaLinOp for CudaCsrOp {
    fn dims(&self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }

    fn apply(
        &self,
        operation: CudaOperation,
        x: &CudaVector,
        y: &mut CudaVector,
    ) -> Result<(), KError> {
        if x.device_ordinal() != self.device_ordinal()
            || y.device_ordinal() != self.device_ordinal()
        {
            return Err(cuda_error(
                CudaErrorKind::DeviceMismatch,
                "apply CUDA CSR operator",
                "operator and vectors must reside on the same CUDA device",
            ));
        }
        let (x_len, y_len) = match operation {
            CudaOperation::NonTranspose => (self.ncols, self.nrows),
            CudaOperation::Transpose | CudaOperation::ConjugateTranspose => {
                (self.nrows, self.ncols)
            }
        };
        if x.len() != x_len || y.len() != y_len {
            return Err(KError::InvalidInput(format!(
                "CUDA SpMV dimension mismatch: op={operation:?}, A={}x{}, x={}, y={}",
                self.nrows,
                self.ncols,
                x.len(),
                y.len()
            )));
        }
        self.resources(operation)?;
        let mut cache = self.cache.lock().map_err(|_| {
            cuda_error(
                CudaErrorKind::Library,
                "lock CUDA SpMV cache",
                "SpMV cache mutex was poisoned",
            )
        })?;
        let resources = match operation {
            CudaOperation::NonTranspose => cache.forward.as_mut().unwrap(),
            CudaOperation::Transpose | CudaOperation::ConjugateTranspose => {
                cache.transpose.as_mut().unwrap()
            }
        };
        let (xp, _xr) = x.buffer().device_ptr(self.runtime.stream());
        let (yp, _yw) = y.buffer_mut().device_ptr_mut(self.runtime.stream());
        let (buffer_ptr, _buffer_guard) = if let Some(buffer) = resources.buffer.as_mut() {
            let (ptr, guard) = buffer.device_ptr_mut(self.runtime.stream());
            (ptr as *mut c_void, Some(guard))
        } else {
            (std::ptr::null_mut(), None)
        };
        unsafe {
            status_to_result(
                cusparse::sys::cusparseDnVecSetValues(resources.x, xp as *mut c_void),
                "set cuSPARSE input vector",
            )?;
            status_to_result(
                cusparse::sys::cusparseDnVecSetValues(resources.y, yp as *mut c_void),
                "set cuSPARSE output vector",
            )?;
        }
        let alpha = host_scalar(S::one());
        let beta = host_scalar(S::zero());
        let handles = self.runtime.handles()?;
        unsafe {
            status_to_result(
                cusparse::sys::cusparseSpMV(
                    handles.cusparse,
                    operation.into_sys(),
                    alpha.as_ptr(),
                    self.descriptor.raw,
                    resources.x,
                    beta.as_ptr(),
                    resources.y,
                    cuda_data_type(),
                    self.spmv_algorithm(),
                    buffer_ptr,
                ),
                "execute cuSPARSE SpMV",
            )?;
        }
        drop(handles);
        self.runtime.diagnostics_ref().library_call();
        self.runtime.maybe_sync_debug()
    }

    fn prepare(&self) -> Result<(), KError> {
        self.resources(CudaOperation::NonTranspose)?;
        self.resources(CudaOperation::Transpose)?;
        Ok(())
    }

    fn supports_transpose(&self) -> bool {
        true
    }

    fn device_ordinal(&self) -> usize {
        self.runtime.device_ordinal()
    }

    fn structure_id(&self) -> StructureId {
        StructureId(self.structure_id.load(Ordering::Relaxed))
    }

    fn values_id(&self) -> ValuesId {
        ValuesId(self.values_id.load(Ordering::Acquire))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl CudaOperation {
    fn into_sys(self) -> cusparse::sys::cusparseOperation_t {
        match self {
            CudaOperation::NonTranspose => {
                cusparse::sys::cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE
            }
            CudaOperation::Transpose => {
                cusparse::sys::cusparseOperation_t::CUSPARSE_OPERATION_TRANSPOSE
            }
            CudaOperation::ConjugateTranspose => {
                cusparse::sys::cusparseOperation_t::CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE
            }
        }
    }
}

pub(crate) enum HostScalar {
    #[cfg(not(feature = "complex"))]
    Real(f64),
    #[cfg(feature = "complex")]
    Complex(cudarc::cublas::sys::double2),
}

impl HostScalar {
    pub(crate) fn as_ptr(&self) -> *const c_void {
        match self {
            #[cfg(not(feature = "complex"))]
            HostScalar::Real(value) => value as *const f64 as *const c_void,
            #[cfg(feature = "complex")]
            HostScalar::Complex(value) => {
                value as *const cudarc::cublas::sys::double2 as *const c_void
            }
        }
    }
}

#[cfg(not(feature = "complex"))]
pub(crate) fn host_scalar(value: S) -> HostScalar {
    HostScalar::Real(value)
}

#[cfg(feature = "complex")]
pub(crate) fn host_scalar(value: S) -> HostScalar {
    HostScalar::Complex(cudarc::cublas::sys::double2 {
        x: value.real(),
        y: value.imag(),
    })
}

#[cfg(not(feature = "complex"))]
pub(crate) fn cuda_data_type() -> cusparse::sys::cudaDataType {
    cusparse::sys::cudaDataType::CUDA_R_64F
}

#[cfg(feature = "complex")]
pub(crate) fn cuda_data_type() -> cusparse::sys::cudaDataType {
    cusparse::sys::cudaDataType::CUDA_C_64F
}

fn validate_csr(
    nrows: usize,
    ncols: usize,
    row_offsets: &[usize],
    col_indices: &[usize],
    values: &[S],
) -> Result<(), KError> {
    if row_offsets.len() != nrows.saturating_add(1) {
        return Err(KError::InvalidInput(format!(
            "CUDA CSR row_offsets length {} does not equal nrows + 1 ({})",
            row_offsets.len(),
            nrows.saturating_add(1)
        )));
    }
    if col_indices.len() != values.len() {
        return Err(KError::InvalidInput(format!(
            "CUDA CSR column/value length mismatch: {} vs {}",
            col_indices.len(),
            values.len()
        )));
    }
    if row_offsets.first().copied().unwrap_or(0) != 0
        || row_offsets.last().copied().unwrap_or(0) != values.len()
        || !row_offsets.windows(2).all(|w| w[0] <= w[1])
    {
        return Err(KError::InvalidInput(
            "CUDA CSR row offsets are not a valid zero-based partition".into(),
        ));
    }
    if col_indices.iter().any(|&column| column >= ncols) {
        return Err(KError::InvalidInput(
            "CUDA CSR column index exceeds matrix dimensions".into(),
        ));
    }
    Ok(())
}

fn extract_diagonal(
    nrows: usize,
    row_offsets: &[usize],
    col_indices: &[usize],
    values: &[S],
) -> Vec<S> {
    let mut diagonal = vec![S::zero(); nrows];
    for row in 0..nrows {
        for position in row_offsets[row]..row_offsets[row + 1] {
            if col_indices[position] == row {
                diagonal[row] = values[position];
                break;
            }
        }
    }
    diagonal
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_malformed_csr_before_cuda_is_touched() {
        assert!(validate_csr(2, 2, &[0, 1], &[0], &[S::one()]).is_err());
        assert!(validate_csr(1, 1, &[0, 2], &[0], &[S::one()]).is_err());
        assert!(validate_csr(1, 1, &[0, 1], &[1], &[S::one()]).is_err());
        assert!(validate_csr(2, 2, &[0, 1, 0], &[0], &[S::one()]).is_err());
    }

    #[test]
    fn accepts_empty_and_regular_csr_shapes() {
        assert!(validate_csr(0, 0, &[0], &[], &[]).is_ok());
        assert!(
            validate_csr(
                2,
                2,
                &[0, 2, 4],
                &[0, 1, 0, 1],
                &[S::one(), S::one(), S::one(), S::one()],
            )
            .is_ok()
        );
    }
}
