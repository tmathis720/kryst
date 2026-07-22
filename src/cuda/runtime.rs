use crate::algebra::prelude::*;
use crate::cuda::kernels::CudaKernels;
use crate::error::{CudaErrorKind, KError};
use cudarc::cublas;
use cudarc::cusparse;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CudaSpmvAlgorithm {
    #[default]
    Auto,
    Fast,
    Deterministic,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CudaMpiTransport {
    #[default]
    Auto,
    DeviceDirect,
    Staged,
}

#[derive(Clone, Debug)]
pub struct CudaOptions {
    pub device_ordinal: usize,
    pub spmv_algorithm: CudaSpmvAlgorithm,
    /// Select algorithms with stable execution order where CUDA exposes one.
    pub deterministic: bool,
    pub synchronize_debug: bool,
    pub collect_diagnostics: bool,
    pub mpi_transport: CudaMpiTransport,
    /// Permit multiple local MPI ranks to select devices modulo the visible
    /// device count. Disabled by default to catch accidental oversubscription.
    pub allow_device_oversubscription: bool,
}

impl Default for CudaOptions {
    fn default() -> Self {
        Self {
            device_ordinal: 0,
            spmv_algorithm: CudaSpmvAlgorithm::Auto,
            deterministic: false,
            synchronize_debug: false,
            collect_diagnostics: true,
            mpi_transport: CudaMpiTransport::Auto,
            allow_device_oversubscription: false,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CudaDiagnosticsSnapshot {
    pub allocations: u64,
    pub allocated_bytes: u64,
    pub host_to_device_bytes: u64,
    pub device_to_host_bytes: u64,
    pub device_to_device_bytes: u64,
    pub library_calls: u64,
    pub kernel_launches: u64,
    pub synchronizations: u64,
    pub setup_calls: u64,
    pub setup_time_ns: u64,
    pub solve_calls: u64,
    pub solve_time_ns: u64,
}

#[derive(Debug)]
pub(crate) struct CudaDiagnostics {
    enabled: bool,
    allocations: AtomicU64,
    allocated_bytes: AtomicU64,
    host_to_device_bytes: AtomicU64,
    device_to_host_bytes: AtomicU64,
    device_to_device_bytes: AtomicU64,
    library_calls: AtomicU64,
    kernel_launches: AtomicU64,
    synchronizations: AtomicU64,
    setup_calls: AtomicU64,
    setup_time_ns: AtomicU64,
    solve_calls: AtomicU64,
    solve_time_ns: AtomicU64,
}

impl CudaDiagnostics {
    fn new(enabled: bool) -> Self {
        Self {
            enabled,
            allocations: AtomicU64::new(0),
            allocated_bytes: AtomicU64::new(0),
            host_to_device_bytes: AtomicU64::new(0),
            device_to_host_bytes: AtomicU64::new(0),
            device_to_device_bytes: AtomicU64::new(0),
            library_calls: AtomicU64::new(0),
            kernel_launches: AtomicU64::new(0),
            synchronizations: AtomicU64::new(0),
            setup_calls: AtomicU64::new(0),
            setup_time_ns: AtomicU64::new(0),
            solve_calls: AtomicU64::new(0),
            solve_time_ns: AtomicU64::new(0),
        }
    }

    pub(crate) fn allocation(&self, bytes: usize) {
        if !self.enabled {
            return;
        }
        self.allocations.fetch_add(1, Ordering::Relaxed);
        self.allocated_bytes
            .fetch_add(bytes as u64, Ordering::Relaxed);
    }

    pub(crate) fn htod(&self, bytes: usize) {
        if !self.enabled {
            return;
        }
        self.host_to_device_bytes
            .fetch_add(bytes as u64, Ordering::Relaxed);
    }

    pub(crate) fn dtoh(&self, bytes: usize) {
        if !self.enabled {
            return;
        }
        self.device_to_host_bytes
            .fetch_add(bytes as u64, Ordering::Relaxed);
    }

    pub(crate) fn dtod(&self, bytes: usize) {
        if !self.enabled {
            return;
        }
        self.device_to_device_bytes
            .fetch_add(bytes as u64, Ordering::Relaxed);
    }

    pub(crate) fn library_call(&self) {
        if !self.enabled {
            return;
        }
        self.library_calls.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn kernel_launch(&self) {
        if !self.enabled {
            return;
        }
        self.kernel_launches.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn synchronization(&self) {
        if !self.enabled {
            return;
        }
        self.synchronizations.fetch_add(1, Ordering::Relaxed);
    }

    fn timing(&self, kind: CudaTimingKind, elapsed_ns: u64) {
        if !self.enabled {
            return;
        }
        let (calls, time) = match kind {
            CudaTimingKind::Setup => (&self.setup_calls, &self.setup_time_ns),
            CudaTimingKind::Solve => (&self.solve_calls, &self.solve_time_ns),
        };
        calls.fetch_add(1, Ordering::Relaxed);
        time.fetch_add(elapsed_ns, Ordering::Relaxed);
    }

    fn snapshot(&self) -> CudaDiagnosticsSnapshot {
        CudaDiagnosticsSnapshot {
            allocations: self.allocations.load(Ordering::Relaxed),
            allocated_bytes: self.allocated_bytes.load(Ordering::Relaxed),
            host_to_device_bytes: self.host_to_device_bytes.load(Ordering::Relaxed),
            device_to_host_bytes: self.device_to_host_bytes.load(Ordering::Relaxed),
            device_to_device_bytes: self.device_to_device_bytes.load(Ordering::Relaxed),
            library_calls: self.library_calls.load(Ordering::Relaxed),
            kernel_launches: self.kernel_launches.load(Ordering::Relaxed),
            synchronizations: self.synchronizations.load(Ordering::Relaxed),
            setup_calls: self.setup_calls.load(Ordering::Relaxed),
            setup_time_ns: self.setup_time_ns.load(Ordering::Relaxed),
            solve_calls: self.solve_calls.load(Ordering::Relaxed),
            solve_time_ns: self.solve_time_ns.load(Ordering::Relaxed),
        }
    }
}

#[derive(Clone, Copy)]
enum CudaTimingKind {
    Setup,
    Solve,
}

pub(crate) struct LibraryHandles {
    pub cublas: cublas::sys::cublasHandle_t,
    pub cusparse: cusparse::sys::cusparseHandle_t,
    context: Arc<CudaContext>,
}

// Access to both handles is serialized by CudaRuntime::handles. CUDA library
// handles are tied to the retained context, which is rebound before destruction.
unsafe impl Send for LibraryHandles {}

impl Drop for LibraryHandles {
    fn drop(&mut self) {
        let _ = self.context.bind_to_thread();
        unsafe {
            let _ = cublas::result::destroy_handle(self.cublas);
            let _ = cusparse::result::destroy(self.cusparse);
        }
    }
}

pub struct CudaRuntime {
    options: CudaOptions,
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    handles: Mutex<LibraryHandles>,
    kernels: CudaKernels,
    diagnostics: CudaDiagnostics,
    device_name: String,
    compute_capability: (i32, i32),
}

pub(crate) struct CudaTimingGuard {
    runtime: Arc<CudaRuntime>,
    kind: CudaTimingKind,
    started: Instant,
}

impl CudaTimingGuard {
    pub(crate) fn setup(runtime: Arc<CudaRuntime>) -> Self {
        Self {
            runtime,
            kind: CudaTimingKind::Setup,
            started: Instant::now(),
        }
    }

    pub(crate) fn solve(runtime: Arc<CudaRuntime>) -> Self {
        Self {
            runtime,
            kind: CudaTimingKind::Solve,
            started: Instant::now(),
        }
    }
}

impl Drop for CudaTimingGuard {
    fn drop(&mut self) {
        let elapsed = self.started.elapsed().as_nanos().min(u64::MAX as u128) as u64;
        self.runtime.diagnostics.timing(self.kind, elapsed);
    }
}

impl std::fmt::Debug for CudaRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaRuntime")
            .field("device_ordinal", &self.device_ordinal())
            .field("device_name", &self.device_name)
            .field("compute_capability", &self.compute_capability)
            .field("options", &self.options)
            .field("diagnostics", &self.diagnostics())
            .finish_non_exhaustive()
    }
}

impl CudaRuntime {
    pub fn new(device_ordinal: usize) -> Result<Arc<Self>, KError> {
        Self::with_options(CudaOptions {
            device_ordinal,
            ..CudaOptions::default()
        })
    }

    /// Select a device from the process-local MPI rank reported by common MPI
    /// and scheduler environment variables. If no local-rank variable exists,
    /// `options.device_ordinal` remains the explicit fallback.
    pub fn for_local_rank(mut options: CudaOptions) -> Result<Arc<Self>, KError> {
        let Some(local_rank) = detected_local_rank() else {
            return Self::with_options(options);
        };
        let device_count = Self::device_count()?;
        if device_count == 0 {
            return Err(cuda_error(
                CudaErrorKind::Unavailable,
                "select CUDA device for local MPI rank",
                "no CUDA devices are visible",
            ));
        }
        if local_rank >= device_count && !options.allow_device_oversubscription {
            return Err(cuda_error(
                CudaErrorKind::DeviceMismatch,
                "select CUDA device for local MPI rank",
                format!(
                    "local rank {local_rank} exceeds {device_count} visible device(s); set allow_device_oversubscription to opt in"
                ),
            ));
        }
        options.device_ordinal = local_rank % device_count;
        Self::with_options(options)
    }

    pub fn with_options(options: CudaOptions) -> Result<Arc<Self>, KError> {
        let driver_present = unsafe { cudarc::driver::sys::is_culib_present() };
        if !driver_present {
            return Err(cuda_error(
                CudaErrorKind::Unavailable,
                "load CUDA driver",
                "CUDA driver library was not found",
            ));
        }
        if !unsafe { cublas::sys::is_culib_present() } {
            return Err(cuda_error(
                CudaErrorKind::Unavailable,
                "load cuBLAS",
                "cuBLAS shared library was not found",
            ));
        }
        if !unsafe { cusparse::sys::is_culib_present() } {
            return Err(cuda_error(
                CudaErrorKind::Unavailable,
                "load cuSPARSE",
                "cuSPARSE shared library was not found",
            ));
        }

        let context = CudaContext::new(options.device_ordinal)
            .map_err(|e| map_driver("create CUDA context", e))?;
        let stream = context
            .new_stream()
            .map_err(|e| map_driver("create CUDA stream", e))?;
        let device_name = context
            .name()
            .map_err(|e| map_driver("query CUDA device name", e))?;
        let compute_capability = context
            .compute_capability()
            .map_err(|e| map_driver("query CUDA compute capability", e))?;
        let kernels = CudaKernels::load(&context)?;

        let cublas = cublas::result::create_handle().map_err(|e| {
            cuda_error(
                CudaErrorKind::Library,
                "create cuBLAS handle",
                format!("{e:?}"),
            )
        })?;
        let cusparse = match cusparse::result::create() {
            Ok(handle) => handle,
            Err(e) => {
                unsafe {
                    let _ = cublas::result::destroy_handle(cublas);
                }
                return Err(cuda_error(
                    CudaErrorKind::Library,
                    "create cuSPARSE handle",
                    format!("{e:?}"),
                ));
            }
        };

        let configure_handles = (|| unsafe {
            cublas::result::set_stream(cublas, stream.cu_stream().cast()).map_err(|e| {
                cuda_error(
                    CudaErrorKind::Library,
                    "set cuBLAS stream",
                    format!("{e:?}"),
                )
            })?;
            if options.deterministic {
                blas_status(
                    cublas::sys::cublasSetAtomicsMode(
                        cublas,
                        cublas::sys::cublasAtomicsMode_t::CUBLAS_ATOMICS_NOT_ALLOWED,
                    ),
                    "set deterministic cuBLAS atomics mode",
                )?;
                blas_status(
                    cublas::sys::cublasSetMathMode(
                        cublas,
                        cublas::sys::cublasMath_t::CUBLAS_PEDANTIC_MATH,
                    ),
                    "set deterministic cuBLAS math mode",
                )?;
            }
            status_to_result(
                cusparse::sys::cusparseSetStream(cusparse, stream.cu_stream().cast()),
                "set cuSPARSE stream",
            )?;
            Ok::<(), KError>(())
        })();
        if let Err(error) = configure_handles {
            unsafe {
                let _ = cublas::result::destroy_handle(cublas);
                let _ = cusparse::result::destroy(cusparse);
            }
            return Err(error);
        }

        let diagnostics = CudaDiagnostics::new(options.collect_diagnostics);

        Ok(Arc::new(Self {
            options,
            context: context.clone(),
            stream,
            handles: Mutex::new(LibraryHandles {
                cublas,
                cusparse,
                context,
            }),
            kernels,
            diagnostics,
            device_name,
            compute_capability,
        }))
    }

    pub fn device_count() -> Result<usize, KError> {
        if !unsafe { cudarc::driver::sys::is_culib_present() } {
            return Err(cuda_error(
                CudaErrorKind::Unavailable,
                "query CUDA devices",
                "CUDA driver library was not found",
            ));
        }
        CudaContext::device_count()
            .map(|n| n.max(0) as usize)
            .map_err(|e| map_driver("query CUDA devices", e))
    }

    pub fn device_ordinal(&self) -> usize {
        self.context.ordinal()
    }

    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    pub fn compute_capability(&self) -> (i32, i32) {
        self.compute_capability
    }

    pub fn options(&self) -> &CudaOptions {
        &self.options
    }

    pub fn diagnostics(&self) -> CudaDiagnosticsSnapshot {
        self.diagnostics.snapshot()
    }

    pub fn synchronize(&self) -> Result<(), KError> {
        self.stream.synchronize().map_err(|e| {
            map_driver_kind(CudaErrorKind::Synchronization, "synchronize stream", e)
        })?;
        self.diagnostics.synchronization();
        Ok(())
    }

    pub(crate) fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    pub(crate) fn context(&self) -> &Arc<CudaContext> {
        &self.context
    }

    pub(crate) fn diagnostics_ref(&self) -> &CudaDiagnostics {
        &self.diagnostics
    }

    pub(crate) fn handles(&self) -> Result<MutexGuard<'_, LibraryHandles>, KError> {
        self.context
            .bind_to_thread()
            .map_err(|e| map_driver("bind CUDA context", e))?;
        self.handles.lock().map_err(|_| {
            cuda_error(
                CudaErrorKind::Library,
                "lock CUDA library handles",
                "CUDA library handle mutex was poisoned",
            )
        })
    }

    pub(crate) fn maybe_sync_debug(&self) -> Result<(), KError> {
        if self.options.synchronize_debug {
            self.synchronize()?;
        }
        Ok(())
    }

    pub(crate) fn copy(
        &self,
        x: &crate::cuda::vector::DeviceBuffer,
        y: &mut crate::cuda::vector::DeviceBuffer,
    ) -> Result<(), KError> {
        check_lengths("CUDA copy", x.len(), y.len())?;
        let stream = self.stream();
        let (xp, _xr) = x.device_ptr(stream);
        let (yp, _yw) = y.device_ptr_mut(stream);
        let handles = self.handles()?;
        let status = unsafe { blas_copy(handles.cublas, x.len(), xp, yp) };
        drop(handles);
        blas_status(status, "cuBLAS copy")?;
        self.diagnostics.library_call();
        self.diagnostics.dtod(x.num_bytes());
        self.maybe_sync_debug()
    }

    pub(crate) fn scale(
        &self,
        alpha: S,
        x: &mut crate::cuda::vector::DeviceBuffer,
    ) -> Result<(), KError> {
        let n = x.len();
        let stream = self.stream();
        let (xp, _xw) = x.device_ptr_mut(stream);
        let handles = self.handles()?;
        let status = unsafe { blas_scale(handles.cublas, n, alpha, xp) };
        drop(handles);
        blas_status(status, "cuBLAS scale")?;
        self.diagnostics.library_call();
        self.maybe_sync_debug()
    }

    pub(crate) fn axpy(
        &self,
        alpha: S,
        x: &crate::cuda::vector::DeviceBuffer,
        y: &mut crate::cuda::vector::DeviceBuffer,
    ) -> Result<(), KError> {
        check_lengths("CUDA axpy", x.len(), y.len())?;
        let stream = self.stream();
        let (xp, _xr) = x.device_ptr(stream);
        let (yp, _yw) = y.device_ptr_mut(stream);
        let handles = self.handles()?;
        let status = unsafe { blas_axpy(handles.cublas, x.len(), alpha, xp, yp) };
        drop(handles);
        blas_status(status, "cuBLAS axpy")?;
        self.diagnostics.library_call();
        self.maybe_sync_debug()
    }

    pub(crate) fn axpby(
        &self,
        alpha: S,
        x: &crate::cuda::vector::DeviceBuffer,
        beta: S,
        y: &mut crate::cuda::vector::DeviceBuffer,
    ) -> Result<(), KError> {
        check_lengths("CUDA axpby", x.len(), y.len())?;
        self.kernels.axpby(self.stream(), alpha, x, beta, y)?;
        self.diagnostics.kernel_launch();
        self.maybe_sync_debug()
    }

    pub(crate) fn cg_update(
        &self,
        alpha: S,
        p: &crate::cuda::vector::DeviceBuffer,
        ap: &crate::cuda::vector::DeviceBuffer,
        x: &mut crate::cuda::vector::DeviceBuffer,
        r: &mut crate::cuda::vector::DeviceBuffer,
    ) -> Result<(), KError> {
        check_lengths("CUDA CG update p/ap", p.len(), ap.len())?;
        check_lengths("CUDA CG update p/x", p.len(), x.len())?;
        check_lengths("CUDA CG update p/r", p.len(), r.len())?;
        self.kernels.cg_update(self.stream(), alpha, p, ap, x, r)?;
        self.diagnostics.kernel_launch();
        self.maybe_sync_debug()
    }

    pub(crate) fn gather(
        &self,
        source: &crate::cuda::vector::DeviceBuffer,
        indices: &CudaSlice<u64>,
        packed: &mut crate::cuda::vector::DeviceBuffer,
    ) -> Result<(), KError> {
        check_lengths("CUDA gather indices/output", indices.len(), packed.len())?;
        self.kernels
            .gather(self.stream(), source, indices, packed)?;
        self.diagnostics.kernel_launch();
        self.maybe_sync_debug()
    }

    pub(crate) fn upload_vector_pointer_table(
        &self,
        vectors: &[crate::cuda::CudaVector],
    ) -> Result<CudaSlice<u64>, KError> {
        let stream = self.stream();
        let mut pointers = Vec::with_capacity(vectors.len());
        for vector in vectors {
            if vector.device_ordinal() != self.device_ordinal() {
                return Err(cuda_error(
                    CudaErrorKind::DeviceMismatch,
                    "build CUDA vector pointer table",
                    "basis vector and runtime use different devices",
                ));
            }
            let (pointer, record) = vector.buffer().device_ptr(stream);
            pointers.push(pointer);
            drop(record);
        }
        let table = stream
            .clone_htod(&pointers)
            .map_err(|error| map_driver("upload CUDA vector pointer table", error))?;
        let bytes = pointers.len().saturating_mul(std::mem::size_of::<u64>());
        self.diagnostics.allocation(bytes);
        self.diagnostics.htod(bytes);
        Ok(table)
    }

    pub(crate) fn arnoldi_multi_dot(
        &self,
        basis_ptrs: &CudaSlice<u64>,
        basis_count: usize,
        rhs: &crate::cuda::vector::DeviceBuffer,
        output: &mut crate::cuda::vector::DeviceBuffer,
        host_output: &mut [crate::cuda::vector::DeviceScalar],
    ) -> Result<(), KError> {
        let needed = basis_count
            .checked_add(1)
            .ok_or_else(|| KError::InvalidInput("CUDA Arnoldi basis count overflow".into()))?;
        if host_output.len() < needed || output.len() < needed {
            return Err(KError::InvalidInput(format!(
                "CUDA Arnoldi payload requires {needed} entries"
            )));
        }
        self.kernels
            .multi_dot(self.stream(), basis_ptrs, basis_count, rhs, output)?;
        self.diagnostics.kernel_launch();
        let output_view = output.slice(..needed);
        self.stream()
            .memcpy_dtoh(&output_view, host_output)
            .map_err(|error| map_driver("download CUDA Arnoldi payload", error))?;
        self.diagnostics
            .dtoh(needed.saturating_mul(std::mem::size_of::<crate::cuda::vector::DeviceScalar>()));
        self.synchronize()
    }

    pub(crate) fn dot(
        &self,
        x: &crate::cuda::vector::DeviceBuffer,
        y: &crate::cuda::vector::DeviceBuffer,
    ) -> Result<S, KError> {
        check_lengths("CUDA dot", x.len(), y.len())?;
        let stream = self.stream();
        let (xp, _xr) = x.device_ptr(stream);
        let (yp, _yr) = y.device_ptr(stream);
        let handles = self.handles()?;
        let result = unsafe { blas_dot(handles.cublas, x.len(), xp, yp) };
        drop(handles);
        let value = result?;
        self.diagnostics.library_call();
        self.diagnostics.synchronization();
        Ok(value)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dot2(
        &self,
        x0: &crate::cuda::vector::DeviceBuffer,
        y0: &crate::cuda::vector::DeviceBuffer,
        x1: &crate::cuda::vector::DeviceBuffer,
        y1: &crate::cuda::vector::DeviceBuffer,
        output: &mut crate::cuda::vector::DeviceBuffer,
    ) -> Result<[S; 2], KError> {
        check_lengths("CUDA dot2 pair 0", x0.len(), y0.len())?;
        check_lengths("CUDA dot2 pair 1", x1.len(), y1.len())?;
        if output.len() != 2 {
            return Err(KError::InvalidInput(format!(
                "CUDA dot2 output must contain two scalars, got {}",
                output.len()
            )));
        }
        let stream = self.stream();
        let output_bytes = output.num_bytes();
        {
            let (x0p, _x0r) = x0.device_ptr(stream);
            let (y0p, _y0r) = y0.device_ptr(stream);
            let (x1p, _x1r) = x1.device_ptr(stream);
            let (y1p, _y1r) = y1.device_ptr(stream);
            let (output_ptr, _ow) = output.device_ptr_mut(stream);
            let second_ptr =
                output_ptr + std::mem::size_of::<crate::cuda::vector::DeviceScalar>() as u64;
            let handles = self.handles()?;
            let device_mode = unsafe {
                cublas::sys::cublasSetPointerMode_v2(
                    handles.cublas,
                    cublas::sys::cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE,
                )
            };
            blas_status(device_mode, "set cuBLAS device pointer mode")?;
            let first =
                unsafe { blas_dot_to_device(handles.cublas, x0.len(), x0p, y0p, output_ptr) };
            let second =
                unsafe { blas_dot_to_device(handles.cublas, x1.len(), x1p, y1p, second_ptr) };
            let restore = unsafe {
                cublas::sys::cublasSetPointerMode_v2(
                    handles.cublas,
                    cublas::sys::cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST,
                )
            };
            drop(handles);
            // Restore the shared handle even when either dot launch failed.
            blas_status(restore, "restore cuBLAS host pointer mode")?;
            blas_status(first, "cuBLAS batched dot 0")?;
            blas_status(second, "cuBLAS batched dot 1")?;
        }
        self.diagnostics.library_call();
        self.diagnostics.library_call();

        let mut host = [crate::cuda::vector::DeviceScalar::default(); 2];
        stream
            .memcpy_dtoh(output, &mut host)
            .map_err(|error| map_driver("download CUDA dot2 payload", error))?;
        self.diagnostics.dtoh(output_bytes);
        self.synchronize()?;
        Ok(device_pair_to_scalars(host))
    }

    pub(crate) fn norm2(&self, x: &crate::cuda::vector::DeviceBuffer) -> Result<R, KError> {
        let stream = self.stream();
        let (xp, _xr) = x.device_ptr(stream);
        let handles = self.handles()?;
        let result = unsafe { blas_norm2(handles.cublas, x.len(), xp) };
        drop(handles);
        let value = result?;
        self.diagnostics.library_call();
        self.diagnostics.synchronization();
        Ok(value)
    }
}

fn detected_local_rank() -> Option<usize> {
    [
        "OMPI_COMM_WORLD_LOCAL_RANK",
        "MPI_LOCALRANKID",
        "MV2_COMM_WORLD_LOCAL_RANK",
        "SLURM_LOCALID",
    ]
    .iter()
    .find_map(|name| std::env::var(name).ok()?.parse::<usize>().ok())
}

pub(crate) fn cuda_error(
    kind: CudaErrorKind,
    operation: &'static str,
    message: impl Into<String>,
) -> KError {
    KError::Cuda {
        kind,
        operation,
        message: message.into(),
    }
}

pub(crate) fn map_driver(
    operation: &'static str,
    error: cudarc::driver::result::DriverError,
) -> KError {
    map_driver_kind(CudaErrorKind::Driver, operation, error)
}

pub(crate) fn map_driver_kind(
    kind: CudaErrorKind,
    operation: &'static str,
    error: cudarc::driver::result::DriverError,
) -> KError {
    cuda_error(kind, operation, format!("{error:?}"))
}

pub(crate) fn status_to_result(
    status: cusparse::sys::cusparseStatus_t,
    operation: &'static str,
) -> Result<(), KError> {
    use cusparse::result::CusparseError;
    status.result().map_err(|CusparseError(code)| {
        cuda_error(CudaErrorKind::Library, operation, format!("{code:?}"))
    })
}

pub(crate) fn blas_status(
    status: cublas::sys::cublasStatus_t,
    operation: &'static str,
) -> Result<(), KError> {
    status
        .result()
        .map_err(|e| cuda_error(CudaErrorKind::Library, operation, format!("{e:?}")))
}

fn check_lengths(operation: &'static str, x: usize, y: usize) -> Result<(), KError> {
    if x == y {
        Ok(())
    } else {
        Err(KError::InvalidInput(format!(
            "{operation} length mismatch: {x} vs {y}"
        )))
    }
}

#[cfg(not(feature = "complex"))]
unsafe fn blas_copy(
    handle: cublas::sys::cublasHandle_t,
    n: usize,
    x: u64,
    y: u64,
) -> cublas::sys::cublasStatus_t {
    unsafe {
        cublas::sys::cublasDcopy_v2_64(handle, n as i64, x as *const f64, 1, y as *mut f64, 1)
    }
}

#[cfg(feature = "complex")]
unsafe fn blas_copy(
    handle: cublas::sys::cublasHandle_t,
    n: usize,
    x: u64,
    y: u64,
) -> cublas::sys::cublasStatus_t {
    unsafe {
        cublas::sys::cublasZcopy_v2_64(
            handle,
            n as i64,
            x as *const cublas::sys::cuDoubleComplex,
            1,
            y as *mut cublas::sys::cuDoubleComplex,
            1,
        )
    }
}

#[cfg(not(feature = "complex"))]
unsafe fn blas_scale(
    handle: cublas::sys::cublasHandle_t,
    n: usize,
    alpha: S,
    x: u64,
) -> cublas::sys::cublasStatus_t {
    unsafe { cublas::sys::cublasDscal_v2_64(handle, n as i64, &alpha, x as *mut f64, 1) }
}

#[cfg(feature = "complex")]
unsafe fn blas_scale(
    handle: cublas::sys::cublasHandle_t,
    n: usize,
    alpha: S,
    x: u64,
) -> cublas::sys::cublasStatus_t {
    let alpha = cublas::sys::double2 {
        x: alpha.real(),
        y: alpha.imag(),
    };
    unsafe {
        cublas::sys::cublasZscal_v2_64(
            handle,
            n as i64,
            &alpha,
            x as *mut cublas::sys::cuDoubleComplex,
            1,
        )
    }
}

#[cfg(not(feature = "complex"))]
unsafe fn blas_axpy(
    handle: cublas::sys::cublasHandle_t,
    n: usize,
    alpha: S,
    x: u64,
    y: u64,
) -> cublas::sys::cublasStatus_t {
    unsafe {
        cublas::sys::cublasDaxpy_v2_64(
            handle,
            n as i64,
            &alpha,
            x as *const f64,
            1,
            y as *mut f64,
            1,
        )
    }
}

#[cfg(feature = "complex")]
unsafe fn blas_axpy(
    handle: cublas::sys::cublasHandle_t,
    n: usize,
    alpha: S,
    x: u64,
    y: u64,
) -> cublas::sys::cublasStatus_t {
    let alpha = cublas::sys::double2 {
        x: alpha.real(),
        y: alpha.imag(),
    };
    unsafe {
        cublas::sys::cublasZaxpy_v2_64(
            handle,
            n as i64,
            &alpha,
            x as *const cublas::sys::cuDoubleComplex,
            1,
            y as *mut cublas::sys::cuDoubleComplex,
            1,
        )
    }
}

#[cfg(not(feature = "complex"))]
unsafe fn blas_dot(
    handle: cublas::sys::cublasHandle_t,
    n: usize,
    x: u64,
    y: u64,
) -> Result<S, KError> {
    let mut result = 0.0;
    blas_status(
        unsafe {
            cublas::sys::cublasDdot_v2_64(
                handle,
                n as i64,
                x as *const f64,
                1,
                y as *const f64,
                1,
                &mut result,
            )
        },
        "cuBLAS dot",
    )?;
    Ok(result)
}

#[cfg(not(feature = "complex"))]
unsafe fn blas_dot_to_device(
    handle: cublas::sys::cublasHandle_t,
    n: usize,
    x: u64,
    y: u64,
    result: u64,
) -> cublas::sys::cublasStatus_t {
    unsafe {
        cublas::sys::cublasDdot_v2_64(
            handle,
            n as i64,
            x as *const f64,
            1,
            y as *const f64,
            1,
            result as *mut f64,
        )
    }
}

#[cfg(feature = "complex")]
unsafe fn blas_dot_to_device(
    handle: cublas::sys::cublasHandle_t,
    n: usize,
    x: u64,
    y: u64,
    result: u64,
) -> cublas::sys::cublasStatus_t {
    unsafe {
        cublas::sys::cublasZdotc_v2_64(
            handle,
            n as i64,
            x as *const cublas::sys::cuDoubleComplex,
            1,
            y as *const cublas::sys::cuDoubleComplex,
            1,
            result as *mut cublas::sys::cuDoubleComplex,
        )
    }
}

#[cfg(not(feature = "complex"))]
fn device_pair_to_scalars(values: [crate::cuda::vector::DeviceScalar; 2]) -> [S; 2] {
    values
}

#[cfg(feature = "complex")]
fn device_pair_to_scalars(values: [crate::cuda::vector::DeviceScalar; 2]) -> [S; 2] {
    [
        S::from_parts(values[0].re, values[0].im),
        S::from_parts(values[1].re, values[1].im),
    ]
}

#[cfg(feature = "complex")]
unsafe fn blas_dot(
    handle: cublas::sys::cublasHandle_t,
    n: usize,
    x: u64,
    y: u64,
) -> Result<S, KError> {
    let mut result = cublas::sys::double2 { x: 0.0, y: 0.0 };
    blas_status(
        unsafe {
            cublas::sys::cublasZdotc_v2_64(
                handle,
                n as i64,
                x as *const cublas::sys::cuDoubleComplex,
                1,
                y as *const cublas::sys::cuDoubleComplex,
                1,
                &mut result,
            )
        },
        "cuBLAS dotc",
    )?;
    Ok(S::from_parts(result.x, result.y))
}

#[cfg(not(feature = "complex"))]
unsafe fn blas_norm2(handle: cublas::sys::cublasHandle_t, n: usize, x: u64) -> Result<R, KError> {
    let mut result = 0.0;
    blas_status(
        unsafe {
            cublas::sys::cublasDnrm2_v2_64(handle, n as i64, x as *const f64, 1, &mut result)
        },
        "cuBLAS nrm2",
    )?;
    Ok(result)
}

#[cfg(feature = "complex")]
unsafe fn blas_norm2(handle: cublas::sys::cublasHandle_t, n: usize, x: u64) -> Result<R, KError> {
    let mut result = 0.0;
    blas_status(
        unsafe {
            cublas::sys::cublasDznrm2_v2_64(
                handle,
                n as i64,
                x as *const cublas::sys::cuDoubleComplex,
                1,
                &mut result,
            )
        },
        "cuBLAS complex nrm2",
    )?;
    Ok(result)
}
