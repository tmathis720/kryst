use super::operator::{cuda_data_type, host_scalar};
use super::runtime::{CudaRuntime, map_driver_kind, status_to_result};
use super::{CudaCsrOp, CudaVector};
use crate::algebra::prelude::*;
use crate::error::{CudaErrorKind, KError};
use cudarc::cusparse;
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use std::ffi::c_void;
use std::sync::{Arc, Mutex};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Triangle {
    LowerUnit,
    UpperNonUnit,
}

struct SpSvResources {
    descriptor: cusparse::sys::cusparseSpSVDescr_t,
    x: cusparse::sys::cusparseDnVecDescr_t,
    y: cusparse::sys::cusparseDnVecDescr_t,
    _buffer: Option<CudaSlice<u8>>,
    runtime: Arc<CudaRuntime>,
}

unsafe impl Send for SpSvResources {}

impl Drop for SpSvResources {
    fn drop(&mut self) {
        let _ = self.runtime.stream().context().bind_to_thread();
        unsafe {
            let _ = cusparse::sys::cusparseSpSV_destroyDescr(self.descriptor);
            let _ = cusparse::sys::cusparseDestroyDnVec(self.x);
            let _ = cusparse::sys::cusparseDestroyDnVec(self.y);
        }
    }
}

pub(crate) struct CudaTriangularSolve {
    factor: CudaCsrOp,
    resources: Mutex<SpSvResources>,
    runtime: Arc<CudaRuntime>,
    n: usize,
}

impl CudaTriangularSolve {
    pub(crate) fn new(
        runtime: Arc<CudaRuntime>,
        n: usize,
        rows: &[usize],
        columns: &[usize],
        values: &[S],
        triangle: Triangle,
    ) -> Result<Self, KError> {
        let factor = CudaCsrOp::from_csr_parts(runtime.clone(), n, n, rows, columns, values)?;
        set_triangular_attributes(&factor, triangle)?;
        let resources = build_resources(&runtime, &factor, n)?;
        Ok(Self {
            factor,
            resources: Mutex::new(resources),
            runtime,
            n,
        })
    }

    pub(crate) fn update_values(&self, values: &[S]) -> Result<(), KError> {
        self.factor.update_values(values)
    }

    pub(crate) fn solve(&self, x: &CudaVector, y: &mut CudaVector) -> Result<(), KError> {
        x.ensure_compatible(y)?;
        if x.len() != self.n {
            return Err(KError::InvalidInput(format!(
                "CUDA triangular solve expected length {}, got {}",
                self.n,
                x.len()
            )));
        }
        if x.device_ordinal() != self.runtime.device_ordinal() {
            return Err(KError::InvalidInput(
                "CUDA triangular factor and vectors use different devices".into(),
            ));
        }
        let resources = self.resources.lock().map_err(|_| {
            KError::SolveError("CUDA triangular solve workspace mutex was poisoned".into())
        })?;
        let (xp, _xr) = x.buffer().device_ptr(self.runtime.stream());
        let (yp, _yw) = y.buffer_mut().device_ptr_mut(self.runtime.stream());
        unsafe {
            status_to_result(
                cusparse::sys::cusparseDnVecSetValues(resources.x, xp as *mut c_void),
                "set cuSPARSE SpSV input vector",
            )?;
            status_to_result(
                cusparse::sys::cusparseDnVecSetValues(resources.y, yp as *mut c_void),
                "set cuSPARSE SpSV output vector",
            )?;
        }
        let alpha = host_scalar(S::one());
        let handles = self.runtime.handles()?;
        unsafe {
            status_to_result(
                cusparse::sys::cusparseSpSV_solve(
                    handles.cusparse,
                    cusparse::sys::cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE,
                    alpha.as_ptr(),
                    self.factor.raw_descriptor(),
                    resources.x,
                    resources.y,
                    cuda_data_type(),
                    cusparse::sys::cusparseSpSVAlg_t::CUSPARSE_SPSV_ALG_DEFAULT,
                    resources.descriptor,
                ),
                "execute cuSPARSE SpSV",
            )?;
        }
        drop(handles);
        self.runtime.diagnostics_ref().library_call();
        self.runtime.maybe_sync_debug()
    }
}

fn set_triangular_attributes(factor: &CudaCsrOp, triangle: Triangle) -> Result<(), KError> {
    let mut fill = match triangle {
        Triangle::LowerUnit => cusparse::sys::cusparseFillMode_t::CUSPARSE_FILL_MODE_LOWER,
        Triangle::UpperNonUnit => cusparse::sys::cusparseFillMode_t::CUSPARSE_FILL_MODE_UPPER,
    };
    let mut diagonal = match triangle {
        Triangle::LowerUnit => cusparse::sys::cusparseDiagType_t::CUSPARSE_DIAG_TYPE_UNIT,
        Triangle::UpperNonUnit => cusparse::sys::cusparseDiagType_t::CUSPARSE_DIAG_TYPE_NON_UNIT,
    };
    unsafe {
        status_to_result(
            cusparse::sys::cusparseSpMatSetAttribute(
                factor.raw_descriptor(),
                cusparse::sys::cusparseSpMatAttribute_t::CUSPARSE_SPMAT_FILL_MODE,
                (&mut fill as *mut cusparse::sys::cusparseFillMode_t).cast(),
                std::mem::size_of_val(&fill),
            ),
            "set cuSPARSE triangular fill mode",
        )?;
        status_to_result(
            cusparse::sys::cusparseSpMatSetAttribute(
                factor.raw_descriptor(),
                cusparse::sys::cusparseSpMatAttribute_t::CUSPARSE_SPMAT_DIAG_TYPE,
                (&mut diagonal as *mut cusparse::sys::cusparseDiagType_t).cast(),
                std::mem::size_of_val(&diagonal),
            ),
            "set cuSPARSE triangular diagonal type",
        )?;
    }
    Ok(())
}

fn build_resources(
    runtime: &Arc<CudaRuntime>,
    factor: &CudaCsrOp,
    n: usize,
) -> Result<SpSvResources, KError> {
    let x_dummy = CudaVector::zeros(runtime.clone(), n)?;
    let mut y_dummy = CudaVector::zeros(runtime.clone(), n)?;
    let (xp, _xr) = x_dummy.buffer().device_ptr(runtime.stream());
    let (yp, _yw) = y_dummy.buffer_mut().device_ptr_mut(runtime.stream());
    let mut x_desc = std::ptr::null_mut();
    let mut y_desc = std::ptr::null_mut();
    let mut solve_desc = std::ptr::null_mut();
    unsafe {
        status_to_result(
            cusparse::sys::cusparseCreateDnVec(
                &mut x_desc,
                n as i64,
                xp as *mut c_void,
                cuda_data_type(),
            ),
            "create cuSPARSE SpSV input descriptor",
        )?;
        if let Err(error) = status_to_result(
            cusparse::sys::cusparseCreateDnVec(
                &mut y_desc,
                n as i64,
                yp as *mut c_void,
                cuda_data_type(),
            ),
            "create cuSPARSE SpSV output descriptor",
        ) {
            let _ = cusparse::sys::cusparseDestroyDnVec(x_desc);
            return Err(error);
        }
        if let Err(error) = status_to_result(
            cusparse::sys::cusparseSpSV_createDescr(&mut solve_desc),
            "create cuSPARSE SpSV descriptor",
        ) {
            let _ = cusparse::sys::cusparseDestroyDnVec(x_desc);
            let _ = cusparse::sys::cusparseDestroyDnVec(y_desc);
            return Err(error);
        }
    }

    let mut resources = SpSvResources {
        descriptor: solve_desc,
        x: x_desc,
        y: y_desc,
        _buffer: None,
        runtime: runtime.clone(),
    };

    let alpha = host_scalar(S::one());
    let mut buffer_size = 0usize;
    let handles = runtime.handles()?;
    unsafe {
        status_to_result(
            cusparse::sys::cusparseSpSV_bufferSize(
                handles.cusparse,
                cusparse::sys::cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE,
                alpha.as_ptr(),
                factor.raw_descriptor(),
                resources.x,
                resources.y,
                cuda_data_type(),
                cusparse::sys::cusparseSpSVAlg_t::CUSPARSE_SPSV_ALG_DEFAULT,
                resources.descriptor,
                &mut buffer_size,
            ),
            "query cuSPARSE SpSV workspace",
        )?;
    }
    drop(handles);
    resources._buffer = if buffer_size == 0 {
        None
    } else {
        let allocation = runtime
            .stream()
            .alloc_zeros::<u8>(buffer_size)
            .map_err(|error| {
                map_driver_kind(
                    CudaErrorKind::Allocation,
                    "allocate cuSPARSE SpSV workspace",
                    error,
                )
            })?;
        runtime.diagnostics_ref().allocation(buffer_size);
        Some(allocation)
    };
    let (buffer_pointer, buffer_guard) = if let Some(buffer) = resources._buffer.as_mut() {
        let (pointer, guard) = buffer.device_ptr_mut(runtime.stream());
        (pointer as *mut c_void, Some(guard))
    } else {
        (std::ptr::null_mut(), None)
    };
    let handles = runtime.handles()?;
    unsafe {
        status_to_result(
            cusparse::sys::cusparseSpSV_analysis(
                handles.cusparse,
                cusparse::sys::cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE,
                alpha.as_ptr(),
                factor.raw_descriptor(),
                resources.x,
                resources.y,
                cuda_data_type(),
                cusparse::sys::cusparseSpSVAlg_t::CUSPARSE_SPSV_ALG_DEFAULT,
                resources.descriptor,
                buffer_pointer,
            ),
            "analyze cuSPARSE SpSV factor",
        )?;
    }
    drop(handles);
    drop(buffer_guard);
    runtime.diagnostics_ref().library_call();
    Ok(resources)
}
