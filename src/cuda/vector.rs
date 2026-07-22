use super::runtime::{CudaRuntime, map_driver, map_driver_kind};
use crate::algebra::prelude::*;
use crate::error::{CudaErrorKind, KError};
use cudarc::driver::CudaSlice;
#[cfg(feature = "complex")]
use cudarc::driver::{DeviceRepr, ValidAsZeroBits};
use std::sync::Arc;

#[cfg(not(feature = "complex"))]
pub(crate) type DeviceScalar = f64;

#[cfg(feature = "complex")]
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct DeviceScalar {
    pub re: f64,
    pub im: f64,
}

#[cfg(feature = "complex")]
unsafe impl DeviceRepr for DeviceScalar {}
#[cfg(feature = "complex")]
unsafe impl ValidAsZeroBits for DeviceScalar {}

pub(crate) type DeviceBuffer = CudaSlice<DeviceScalar>;

pub struct CudaVector {
    runtime: Arc<CudaRuntime>,
    data: DeviceBuffer,
}

impl std::fmt::Debug for CudaVector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaVector")
            .field("len", &self.len())
            .field("device_ordinal", &self.device_ordinal())
            .finish()
    }
}

impl CudaVector {
    pub fn zeros(runtime: Arc<CudaRuntime>, len: usize) -> Result<Self, KError> {
        let data = runtime
            .stream()
            .alloc_zeros::<DeviceScalar>(len)
            .map_err(|e| map_driver_kind(CudaErrorKind::Allocation, "allocate CUDA vector", e))?;
        runtime
            .diagnostics_ref()
            .allocation(len.saturating_mul(std::mem::size_of::<DeviceScalar>()));
        Ok(Self { runtime, data })
    }

    pub fn from_host(runtime: Arc<CudaRuntime>, host: &[S]) -> Result<Self, KError> {
        let converted = host_to_device(host);
        let data = runtime
            .stream()
            .clone_htod(converted.as_ref())
            .map_err(|e| map_driver("copy CUDA vector from host", e))?;
        let bytes = host
            .len()
            .saturating_mul(std::mem::size_of::<DeviceScalar>());
        runtime.diagnostics_ref().allocation(bytes);
        runtime.diagnostics_ref().htod(bytes);
        Ok(Self { runtime, data })
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn device_ordinal(&self) -> usize {
        self.runtime.device_ordinal()
    }

    pub fn runtime(&self) -> &Arc<CudaRuntime> {
        &self.runtime
    }

    pub fn copy_from_host(&mut self, host: &[S]) -> Result<(), KError> {
        if host.len() != self.len() {
            return Err(KError::InvalidInput(format!(
                "CUDA vector upload length mismatch: {} vs {}",
                host.len(),
                self.len()
            )));
        }
        let converted = host_to_device(host);
        self.runtime
            .stream()
            .memcpy_htod(converted.as_ref(), &mut self.data)
            .map_err(|e| map_driver("copy CUDA vector from host", e))?;
        self.runtime.diagnostics_ref().htod(self.data.num_bytes());
        Ok(())
    }

    pub fn copy_to_host(&self, host: &mut [S]) -> Result<(), KError> {
        if host.len() != self.len() {
            return Err(KError::InvalidInput(format!(
                "CUDA vector download length mismatch: {} vs {}",
                host.len(),
                self.len()
            )));
        }
        let values = self
            .runtime
            .stream()
            .clone_dtoh(&self.data)
            .map_err(|e| map_driver("copy CUDA vector to host", e))?;
        device_to_host(&values, host);
        self.runtime.diagnostics_ref().dtoh(self.data.num_bytes());
        self.runtime.diagnostics_ref().synchronization();
        Ok(())
    }

    pub fn to_host(&self) -> Result<Vec<S>, KError> {
        let mut host = vec![S::zero(); self.len()];
        self.copy_to_host(&mut host)?;
        Ok(host)
    }

    pub fn try_clone(&self) -> Result<Self, KError> {
        let data = self
            .data
            .try_clone()
            .map_err(|e| map_driver("clone CUDA vector", e))?;
        self.runtime
            .diagnostics_ref()
            .allocation(self.data.num_bytes());
        self.runtime.diagnostics_ref().dtod(self.data.num_bytes());
        Ok(Self {
            runtime: self.runtime.clone(),
            data,
        })
    }

    pub fn fill_zero(&mut self) -> Result<(), KError> {
        self.runtime
            .stream()
            .memset_zeros(&mut self.data)
            .map_err(|e| map_driver("zero CUDA vector", e))?;
        self.runtime.diagnostics_ref().kernel_launch();
        self.runtime.maybe_sync_debug()
    }

    pub(crate) fn buffer(&self) -> &DeviceBuffer {
        &self.data
    }

    pub(crate) fn buffer_mut(&mut self) -> &mut DeviceBuffer {
        &mut self.data
    }

    pub(crate) fn ensure_compatible(&self, other: &Self) -> Result<(), KError> {
        if self.device_ordinal() != other.device_ordinal() {
            return Err(super::runtime::cuda_error(
                CudaErrorKind::DeviceMismatch,
                "validate CUDA vectors",
                format!(
                    "vectors are on devices {} and {}",
                    self.device_ordinal(),
                    other.device_ordinal()
                ),
            ));
        }
        if self.len() != other.len() {
            return Err(KError::InvalidInput(format!(
                "CUDA vector length mismatch: {} vs {}",
                self.len(),
                other.len()
            )));
        }
        Ok(())
    }
}

#[cfg(not(feature = "complex"))]
pub(crate) fn host_to_device(host: &[S]) -> std::borrow::Cow<'_, [DeviceScalar]> {
    std::borrow::Cow::Borrowed(host)
}

#[cfg(feature = "complex")]
pub(crate) fn host_to_device(host: &[S]) -> std::borrow::Cow<'_, [DeviceScalar]> {
    std::borrow::Cow::Owned(
        host.iter()
            .map(|&value| DeviceScalar {
                re: value.real(),
                im: value.imag(),
            })
            .collect(),
    )
}

#[cfg(not(feature = "complex"))]
pub(crate) fn device_to_host(device: &[DeviceScalar], host: &mut [S]) {
    host.copy_from_slice(device);
}

#[cfg(feature = "complex")]
pub(crate) fn device_to_host(device: &[DeviceScalar], host: &mut [S]) {
    for (dst, src) in host.iter_mut().zip(device) {
        *dst = S::from_parts(src.re, src.im);
    }
}

#[cfg(all(test, feature = "complex"))]
mod tests {
    use super::*;

    #[test]
    fn device_complex_layout_matches_cuda_double2() {
        assert_eq!(
            std::mem::size_of::<DeviceScalar>(),
            std::mem::size_of::<cudarc::cublas::sys::double2>()
        );
        assert_eq!(
            std::mem::align_of::<DeviceScalar>(),
            std::mem::align_of::<cudarc::cublas::sys::double2>()
        );
    }
}
