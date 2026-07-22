use super::runtime::{cuda_error, map_driver_kind};
use super::vector::DeviceBuffer;
use crate::algebra::prelude::*;
use crate::error::{CudaErrorKind, KError};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

/// Functions loaded from the PTX shipped with kryst. Keeping the module alive
/// through each `CudaFunction` also makes teardown follow CUDA's required
/// function-before-context order.
pub(crate) struct CudaKernels {
    axpby: CudaFunction,
    cg_update: CudaFunction,
    gather: CudaFunction,
    multi_dot: CudaFunction,
}

impl std::fmt::Debug for CudaKernels {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaKernels").finish_non_exhaustive()
    }
}

impl CudaKernels {
    pub(crate) fn load(context: &Arc<CudaContext>) -> Result<Self, KError> {
        #[cfg(not(feature = "complex"))]
        let (ptx, axpby_name, cg_update_name, gather_name, multi_dot_name) = (
            include_str!("ptx/vector_real.ptx"),
            "kryst_axpby_f64",
            "kryst_cg_update_f64",
            "kryst_gather_f64",
            "kryst_multi_dot_f64",
        );
        #[cfg(feature = "complex")]
        let (ptx, axpby_name, cg_update_name, gather_name, multi_dot_name) = (
            include_str!("ptx/vector_complex.ptx"),
            "kryst_axpby_c64",
            "kryst_cg_update_c64",
            "kryst_gather_c64",
            "kryst_multi_dot_c64",
        );

        let module = context
            .load_module(Ptx::from_src(ptx))
            .map_err(|error| map_driver_kind(CudaErrorKind::Kernel, "load kryst PTX", error))?;
        let axpby = module.load_function(axpby_name).map_err(|error| {
            map_driver_kind(CudaErrorKind::Kernel, "load kryst AXPBY kernel", error)
        })?;
        let cg_update = module.load_function(cg_update_name).map_err(|error| {
            map_driver_kind(CudaErrorKind::Kernel, "load kryst CG update kernel", error)
        })?;
        let gather = module.load_function(gather_name).map_err(|error| {
            map_driver_kind(CudaErrorKind::Kernel, "load kryst gather kernel", error)
        })?;
        let multi_dot = module.load_function(multi_dot_name).map_err(|error| {
            map_driver_kind(CudaErrorKind::Kernel, "load kryst multi-dot kernel", error)
        })?;
        Ok(Self {
            axpby,
            cg_update,
            gather,
            multi_dot,
        })
    }

    /// Compute `<basis[i], rhs>` for every pointer in `basis_ptrs`, followed
    /// by `<rhs, rhs>`. One fixed 256-thread block owns each dot product, so
    /// the reduction tree and its write are deterministic for a fixed launch.
    pub(crate) fn multi_dot(
        &self,
        stream: &CudaStream,
        basis_ptrs: &CudaSlice<u64>,
        basis_count: usize,
        rhs: &DeviceBuffer,
        output: &mut DeviceBuffer,
    ) -> Result<(), KError> {
        let output_count = basis_count.checked_add(1).ok_or_else(|| {
            cuda_error(
                CudaErrorKind::Kernel,
                "configure kryst multi-dot kernel",
                "basis count overflow",
            )
        })?;
        if basis_ptrs.len() < basis_count || output.len() < output_count {
            return Err(KError::InvalidInput(
                "CUDA multi-dot pointer table or output buffer is too small".into(),
            ));
        }
        let blocks = u32::try_from(output_count).map_err(|_| {
            cuda_error(
                CudaErrorKind::Kernel,
                "configure kryst multi-dot kernel",
                "basis count exceeds CUDA's one-dimensional grid limit",
            )
        })?;
        let basis_count = basis_count as u64;
        let n = rhs.len() as u64;
        let mut launch = stream.launch_builder(&self.multi_dot);
        launch
            .arg(basis_ptrs)
            .arg(rhs)
            .arg(output)
            .arg(&basis_count)
            .arg(&n);
        // SAFETY: basis_ptrs contains stable pointers to basis allocations of
        // length rhs.len(); output has basis_count+1 elements. GmresWorkspace
        // owns all allocations for the duration of the launch.
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|error| map_driver_kind(CudaErrorKind::Kernel, "launch multi-dot", error))?;
        Ok(())
    }

    pub(crate) fn gather(
        &self,
        stream: &CudaStream,
        source: &DeviceBuffer,
        indices: &CudaSlice<u64>,
        packed: &mut DeviceBuffer,
    ) -> Result<(), KError> {
        let Some(config) = launch_config(indices.len())? else {
            return Ok(());
        };
        let n = indices.len() as u64;
        let mut launch = stream.launch_builder(&self.gather);
        launch.arg(source).arg(indices).arg(packed).arg(&n);
        // SAFETY: the PTX reads exactly `indices.len()` u64 indices and writes
        // the same number of DeviceScalar elements. The caller validates every
        // index against `source` and the output length before launch.
        unsafe { launch.launch(config) }
            .map_err(|error| map_driver_kind(CudaErrorKind::Kernel, "launch gather", error))?;
        Ok(())
    }

    pub(crate) fn axpby(
        &self,
        stream: &CudaStream,
        alpha: S,
        x: &DeviceBuffer,
        beta: S,
        y: &mut DeviceBuffer,
    ) -> Result<(), KError> {
        let Some(config) = launch_config(x.len())? else {
            return Ok(());
        };
        let n = x.len() as u64;
        #[cfg(feature = "complex")]
        let (alpha_re, alpha_im, beta_re, beta_im) =
            (alpha.real(), alpha.imag(), beta.real(), beta.imag());
        let mut launch = stream.launch_builder(&self.axpby);
        launch.arg(x).arg(y);
        #[cfg(not(feature = "complex"))]
        launch.arg(&alpha).arg(&beta);
        #[cfg(feature = "complex")]
        launch
            .arg(&alpha_re)
            .arg(&alpha_im)
            .arg(&beta_re)
            .arg(&beta_im);
        launch.arg(&n);
        // SAFETY: the selected PTX signature matches the cfg-specific scalar
        // layout and every pointer references `n` DeviceScalar elements.
        unsafe { launch.launch(config) }
            .map_err(|error| map_driver_kind(CudaErrorKind::Kernel, "launch AXPBY", error))?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn cg_update(
        &self,
        stream: &CudaStream,
        alpha: S,
        p: &DeviceBuffer,
        ap: &DeviceBuffer,
        x: &mut DeviceBuffer,
        r: &mut DeviceBuffer,
    ) -> Result<(), KError> {
        let Some(config) = launch_config(p.len())? else {
            return Ok(());
        };
        let n = p.len() as u64;
        #[cfg(feature = "complex")]
        let (alpha_re, alpha_im) = (alpha.real(), alpha.imag());
        let mut launch = stream.launch_builder(&self.cg_update);
        launch.arg(p).arg(ap).arg(x).arg(r);
        #[cfg(not(feature = "complex"))]
        launch.arg(&alpha);
        #[cfg(feature = "complex")]
        launch.arg(&alpha_re).arg(&alpha_im);
        launch.arg(&n);
        // SAFETY: the selected PTX signature matches the cfg-specific scalar
        // layout and all four pointers reference `n` DeviceScalar elements.
        unsafe { launch.launch(config) }
            .map_err(|error| map_driver_kind(CudaErrorKind::Kernel, "launch CG update", error))?;
        Ok(())
    }
}

fn launch_config(len: usize) -> Result<Option<LaunchConfig>, KError> {
    if len == 0 {
        return Ok(None);
    }
    const THREADS: usize = 256;
    let blocks = len.div_ceil(THREADS);
    let blocks = u32::try_from(blocks).map_err(|_| {
        cuda_error(
            CudaErrorKind::Kernel,
            "configure kryst CUDA kernel",
            "vector is too large for a one-dimensional CUDA launch",
        )
    })?;
    Ok(Some(LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (THREADS as u32, 1, 1),
        shared_mem_bytes: 0,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_in_ptx_exports_expected_real_kernels() {
        let ptx = include_str!("ptx/vector_real.ptx");
        assert!(ptx.contains(".entry kryst_axpby_f64"));
        assert!(ptx.contains(".entry kryst_cg_update_f64"));
        assert!(ptx.contains(".entry kryst_gather_f64"));
        assert!(ptx.contains(".entry kryst_multi_dot_f64"));
    }

    #[test]
    fn checked_in_ptx_exports_expected_complex_kernels() {
        let ptx = include_str!("ptx/vector_complex.ptx");
        assert!(ptx.contains(".entry kryst_axpby_c64"));
        assert!(ptx.contains(".entry kryst_cg_update_c64"));
        assert!(ptx.contains(".entry kryst_gather_c64"));
        assert!(ptx.contains(".entry kryst_multi_dot_c64"));
    }

    #[test]
    fn launch_configuration_covers_vector() {
        let config = launch_config(257).unwrap().unwrap();
        assert_eq!(config.grid_dim.0, 2);
        assert_eq!(config.block_dim.0, 256);
        assert!(launch_config(0).unwrap().is_none());
    }
}
