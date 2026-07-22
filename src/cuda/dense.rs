use super::operator::{CudaLinOp, CudaOperation};
use super::runtime::{CudaRuntime, blas_status, cuda_error, map_driver};
use super::vector::{CudaVector, DeviceBuffer, DeviceScalar, host_to_device};
use crate::algebra::prelude::*;
use crate::error::{CudaErrorKind, KError};
use crate::matrix::op::{StructureId, ValuesId};
use cudarc::cublas;
use cudarc::driver::{DevicePtr, DevicePtrMut};
use std::any::Any;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Device-resident column-major dense matrix applied through cuBLAS GEMV.
pub struct CudaDenseOp {
    runtime: Arc<CudaRuntime>,
    nrows: usize,
    ncols: usize,
    values: Mutex<DeviceBuffer>,
    structure_id: AtomicU64,
    values_id: AtomicU64,
}

impl std::fmt::Debug for CudaDenseOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaDenseOp")
            .field("dims", &(self.nrows, self.ncols))
            .field("device_ordinal", &self.device_ordinal())
            .field("structure_id", &self.structure_id())
            .field("values_id", &self.values_id())
            .finish()
    }
}

impl CudaDenseOp {
    pub fn from_column_major(
        runtime: Arc<CudaRuntime>,
        nrows: usize,
        ncols: usize,
        values: &[S],
    ) -> Result<Self, KError> {
        let expected = nrows.checked_mul(ncols).ok_or_else(|| {
            KError::InvalidInput("CUDA dense matrix dimensions overflow usize".into())
        })?;
        if values.len() != expected {
            return Err(KError::InvalidInput(format!(
                "CUDA dense values length {} does not equal {nrows}x{ncols}",
                values.len()
            )));
        }
        let converted = host_to_device(values);
        let device_values = runtime
            .stream()
            .clone_htod(converted.as_ref())
            .map_err(|error| map_driver("upload CUDA dense matrix", error))?;
        let bytes = values.len() * std::mem::size_of::<DeviceScalar>();
        runtime.diagnostics_ref().allocation(bytes);
        runtime.diagnostics_ref().htod(bytes);
        Ok(Self {
            runtime,
            nrows,
            ncols,
            values: Mutex::new(device_values),
            structure_id: AtomicU64::new(1),
            values_id: AtomicU64::new(1),
        })
    }

    pub fn from_row_major(
        runtime: Arc<CudaRuntime>,
        nrows: usize,
        ncols: usize,
        values: &[S],
    ) -> Result<Self, KError> {
        let expected = nrows.checked_mul(ncols).ok_or_else(|| {
            KError::InvalidInput("CUDA dense matrix dimensions overflow usize".into())
        })?;
        if values.len() != expected {
            return Err(KError::InvalidInput(format!(
                "CUDA dense values length {} does not equal {nrows}x{ncols}",
                values.len()
            )));
        }
        let mut column_major = vec![S::zero(); expected];
        for row in 0..nrows {
            for column in 0..ncols {
                column_major[column * nrows + row] = values[row * ncols + column];
            }
        }
        Self::from_column_major(runtime, nrows, ncols, &column_major)
    }

    pub fn update_values_column_major(&self, values: &[S]) -> Result<(), KError> {
        if values.len() != self.nrows * self.ncols {
            return Err(KError::InvalidInput(format!(
                "CUDA dense numeric update length mismatch: {} vs {}",
                values.len(),
                self.nrows * self.ncols
            )));
        }
        let converted = host_to_device(values);
        let mut device_values = self.values.lock().map_err(|_| {
            cuda_error(
                CudaErrorKind::Library,
                "lock CUDA dense values",
                "numeric value buffer lock was poisoned",
            )
        })?;
        self.runtime
            .stream()
            .memcpy_htod(converted.as_ref(), &mut *device_values)
            .map_err(|error| map_driver("update CUDA dense values", error))?;
        self.runtime
            .diagnostics_ref()
            .htod(values.len() * std::mem::size_of::<DeviceScalar>());
        self.values_id.fetch_add(1, Ordering::Release);
        Ok(())
    }
}

impl CudaLinOp for CudaDenseOp {
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
                "apply CUDA dense operator",
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
                "CUDA GEMV dimension mismatch: op={operation:?}, A={}x{}, x={}, y={}",
                self.nrows,
                self.ncols,
                x.len(),
                y.len()
            )));
        }
        if y_len == 0 {
            return Ok(());
        }
        if x_len == 0 {
            return y.fill_zero();
        }

        let values = self.values.lock().map_err(|_| {
            cuda_error(
                CudaErrorKind::Library,
                "lock CUDA dense values",
                "numeric value buffer lock was poisoned",
            )
        })?;
        let stream = self.runtime.stream();
        let (ap, _ar) = values.device_ptr(stream);
        let (xp, _xr) = x.buffer().device_ptr(stream);
        let (yp, _yw) = y.buffer_mut().device_ptr_mut(stream);
        let handles = self.runtime.handles()?;
        let status = unsafe {
            dense_gemv(
                handles.cublas,
                operation,
                self.nrows,
                self.ncols,
                ap,
                xp,
                yp,
            )
        };
        drop(handles);
        blas_status(status, "cuBLAS GEMV")?;
        self.runtime.diagnostics_ref().library_call();
        self.runtime.maybe_sync_debug()
    }

    fn supports_transpose(&self) -> bool {
        true
    }

    fn device_ordinal(&self) -> usize {
        self.runtime.device_ordinal()
    }

    fn structure_id(&self) -> StructureId {
        StructureId(self.structure_id.load(Ordering::Acquire))
    }

    fn values_id(&self) -> ValuesId {
        ValuesId(self.values_id.load(Ordering::Acquire))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

fn cublas_operation(operation: CudaOperation) -> cublas::sys::cublasOperation_t {
    match operation {
        CudaOperation::NonTranspose => cublas::sys::cublasOperation_t::CUBLAS_OP_N,
        CudaOperation::Transpose => cublas::sys::cublasOperation_t::CUBLAS_OP_T,
        CudaOperation::ConjugateTranspose => cublas::sys::cublasOperation_t::CUBLAS_OP_C,
    }
}

#[cfg(not(feature = "complex"))]
unsafe fn dense_gemv(
    handle: cublas::sys::cublasHandle_t,
    operation: CudaOperation,
    nrows: usize,
    ncols: usize,
    a: u64,
    x: u64,
    y: u64,
) -> cublas::sys::cublasStatus_t {
    let alpha = 1.0;
    let beta = 0.0;
    unsafe {
        cublas::sys::cublasDgemv_v2_64(
            handle,
            cublas_operation(operation),
            nrows as i64,
            ncols as i64,
            &alpha,
            a as *const f64,
            nrows.max(1) as i64,
            x as *const f64,
            1,
            &beta,
            y as *mut f64,
            1,
        )
    }
}

#[cfg(feature = "complex")]
unsafe fn dense_gemv(
    handle: cublas::sys::cublasHandle_t,
    operation: CudaOperation,
    nrows: usize,
    ncols: usize,
    a: u64,
    x: u64,
    y: u64,
) -> cublas::sys::cublasStatus_t {
    let alpha = cublas::sys::double2 { x: 1.0, y: 0.0 };
    let beta = cublas::sys::double2 { x: 0.0, y: 0.0 };
    unsafe {
        cublas::sys::cublasZgemv_v2_64(
            handle,
            cublas_operation(operation),
            nrows as i64,
            ncols as i64,
            &alpha,
            a as *const cublas::sys::cuDoubleComplex,
            nrows.max(1) as i64,
            x as *const cublas::sys::cuDoubleComplex,
            1,
            &beta,
            y as *mut cublas::sys::cuDoubleComplex,
            1,
        )
    }
}
