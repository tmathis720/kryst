#![cfg(feature = "cuda")]

use kryst::context::ksp_context::SolverType;
use kryst::context::pc_context::PcType;
use kryst::cuda::{
    CudaCgVariant, CudaGmresVariant, CudaKspContext, CudaOptions, CudaRuntime, CudaSpmvAlgorithm,
};
use kryst::error::{CudaErrorKind, KError};

#[test]
fn cuda_is_optional_and_runtime_failures_are_structured() {
    // An impossible ordinal ensures this path is an error even on a GPU host.
    // CPU-only hosts fail earlier while dynamically loading the CUDA libraries.
    let error = CudaRuntime::new(usize::MAX).expect_err("invalid CUDA ordinal must fail");
    match error {
        KError::Cuda {
            kind: CudaErrorKind::Unavailable | CudaErrorKind::Driver | CudaErrorKind::Library,
            operation,
            message,
        } => {
            assert!(!operation.is_empty());
            assert!(!message.is_empty());
        }
        other => panic!("unexpected CUDA initialization error: {other:?}"),
    }
}

#[test]
fn cuda_options_have_a_fast_non_debug_default() {
    let options = CudaOptions::default();
    assert_eq!(options.device_ordinal, 0);
    assert_eq!(options.spmv_algorithm, CudaSpmvAlgorithm::Auto);
    assert!(!options.deterministic);
    assert!(!options.synchronize_debug);
    assert!(options.collect_diagnostics);
    assert!(!options.allow_device_oversubscription);
    assert_eq!(options.mpi_transport, kryst::cuda::CudaMpiTransport::Auto);
}

#[test]
fn cuda_capability_checks_reject_unsupported_combinations_early() {
    assert_eq!(CudaCgVariant::default(), CudaCgVariant::Classical);
    assert_eq!(CudaGmresVariant::default(), CudaGmresVariant::Classical);
    for solver in [
        SolverType::Cg,
        SolverType::Pcg,
        SolverType::Gmres,
        SolverType::Fgmres,
        SolverType::BiCgStab,
        SolverType::Cgs,
        SolverType::Cgnr,
        SolverType::Cr,
        SolverType::Lsqr,
        SolverType::Lsmr,
        SolverType::Gcr,
        SolverType::PipeGcr,
        SolverType::Qmr,
        SolverType::Tfqmr,
        SolverType::Tcqmr,
        SolverType::Richardson,
        SolverType::Chebyshev,
    ] {
        assert!(CudaKspContext::supports_solver_type(solver));
    }
    assert!(!CudaKspContext::supports_solver_type(SolverType::Minres));

    assert!(CudaKspContext::supports_pc_type(PcType::None));
    assert!(CudaKspContext::supports_pc_type(PcType::Jacobi));
    assert!(CudaKspContext::supports_pc_type(PcType::BlockJacobi));
    assert!(CudaKspContext::supports_pc_type(PcType::Chebyshev));
    assert!(CudaKspContext::supports_pc_type(PcType::Ilu0));
    assert!(CudaKspContext::supports_pc_type(PcType::Amg));
    assert!(!CudaKspContext::supports_pc_type(PcType::Ilu));
}
