#![cfg(feature = "cuda")]

use kryst::algebra::prelude::*;
use kryst::context::ksp_context::SolverType;
use kryst::context::pc_context::PcType;
use kryst::cuda::{
    CudaAmg, CudaAmgOptions, CudaCgVariant, CudaCsrOp, CudaDenseOp, CudaDistCsrOp,
    CudaGmresVariant, CudaKspContext, CudaLinOp, CudaOperation, CudaOptions, CudaRuntime,
    CudaVector,
};
use kryst::matrix::sparse::CsrMatrix;
use kryst::parallel::{NoComm, UniverseComm};
use std::sync::Arc;

fn assert_close(actual: &[S], expected: &[S], tolerance: R) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (*actual - *expected).abs() <= tolerance,
            "entry {index}: actual={actual:?}, expected={expected:?}"
        );
    }
}

fn scalar_bits(value: S) -> (u64, u64) {
    #[cfg(not(feature = "complex"))]
    {
        (value.to_bits(), 0)
    }
    #[cfg(feature = "complex")]
    {
        (value.real().to_bits(), value.imag().to_bits())
    }
}

#[test]
#[ignore = "requires an NVIDIA GPU with CUDA 12+ libraries"]
fn dense_gemv_and_dimension_validation_match_reference() {
    let runtime = CudaRuntime::new(0).expect("CUDA runtime");
    let operator = CudaDenseOp::from_row_major(
        runtime.clone(),
        2,
        3,
        &[
            S::from_real(1.0),
            S::from_real(2.0),
            S::from_real(3.0),
            S::from_real(4.0),
            S::from_real(5.0),
            S::from_real(6.0),
        ],
    )
    .unwrap();
    let input = CudaVector::from_host(
        runtime.clone(),
        &[S::one(), S::from_real(2.0), S::from_real(3.0)],
    )
    .unwrap();
    let mut output = CudaVector::zeros(runtime.clone(), 2).unwrap();
    operator
        .apply(CudaOperation::NonTranspose, &input, &mut output)
        .unwrap();
    assert_close(
        &output.to_host().unwrap(),
        &[S::from_real(14.0), S::from_real(32.0)],
        1e-12,
    );

    let transpose_input =
        CudaVector::from_host(runtime.clone(), &[S::one(), S::from_real(2.0)]).unwrap();
    let mut transpose_output = CudaVector::zeros(runtime.clone(), 3).unwrap();
    operator
        .apply(
            CudaOperation::ConjugateTranspose,
            &transpose_input,
            &mut transpose_output,
        )
        .unwrap();
    assert_close(
        &transpose_output.to_host().unwrap(),
        &[S::from_real(9.0), S::from_real(12.0), S::from_real(15.0)],
        1e-12,
    );

    let wrong_input = CudaVector::zeros(runtime, 2).unwrap();
    assert!(matches!(
        operator.apply(CudaOperation::NonTranspose, &wrong_input, &mut output),
        Err(kryst::KError::InvalidInput(_))
    ));
}

#[test]
#[ignore = "requires an NVIDIA GPU with CUDA 12+ libraries"]
fn deterministic_pipelined_pcg_is_bitwise_repeatable() {
    let runtime = CudaRuntime::with_options(CudaOptions {
        deterministic: true,
        ..CudaOptions::default()
    })
    .expect("deterministic CUDA runtime");
    let matrix = CsrMatrix::from_csr(
        4,
        4,
        vec![0, 2, 5, 8, 10],
        vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3],
        vec![
            S::from_real(2.0),
            S::from_real(-1.0),
            S::from_real(-1.0),
            S::from_real(2.0),
            S::from_real(-1.0),
            S::from_real(-1.0),
            S::from_real(2.0),
            S::from_real(-1.0),
            S::from_real(-1.0),
            S::from_real(2.0),
        ],
    );
    let operator = Arc::new(CudaCsrOp::from_host(runtime.clone(), &matrix).unwrap());
    let rhs = CudaVector::from_host(runtime.clone(), &[S::one(), S::zero(), S::zero(), S::one()])
        .unwrap();
    let mut solution = CudaVector::zeros(runtime.clone(), 4).unwrap();
    let mut ksp = CudaKspContext::new(runtime.clone());
    ksp.set_type(SolverType::Pcg).unwrap();
    ksp.set_cg_variant(CudaCgVariant::Pipelined);
    ksp.set_pc_type(PcType::Jacobi).unwrap();
    ksp.set_tolerances(1e-13, 1e-15, 1e8, 20).unwrap();
    ksp.set_operators(operator, None).unwrap();
    ksp.setup().unwrap();

    let first_stats = ksp.solve(&rhs, &mut solution).unwrap();
    assert!(first_stats.reason.is_converged(), "{first_stats:?}");
    let first: Vec<_> = solution
        .to_host()
        .unwrap()
        .into_iter()
        .map(scalar_bits)
        .collect();
    solution.fill_zero().unwrap();
    let before = runtime.diagnostics();
    let second_stats = ksp.solve(&rhs, &mut solution).unwrap();
    let after = runtime.diagnostics();
    assert!(second_stats.reason.is_converged(), "{second_stats:?}");
    let second: Vec<_> = solution
        .to_host()
        .unwrap()
        .into_iter()
        .map(scalar_bits)
        .collect();
    assert_eq!(first, second);
    assert_eq!(after.allocations, before.allocations);
}

#[test]
#[ignore = "requires an NVIDIA GPU with CUDA 12+ libraries"]
fn identity_cg_stays_resident_and_reuses_workspace() {
    let runtime = CudaRuntime::new(0).expect("CUDA runtime");
    let matrix = CsrMatrix::from_csr(
        3,
        3,
        vec![0, 1, 2, 3],
        vec![0, 1, 2],
        vec![S::one(), S::one(), S::one()],
    );
    let operator = Arc::new(CudaCsrOp::from_host(runtime.clone(), &matrix).unwrap());
    let b = CudaVector::from_host(
        runtime.clone(),
        &[S::from_real(1.0), S::from_real(2.0), S::from_real(3.0)],
    )
    .unwrap();
    let mut x = CudaVector::zeros(runtime.clone(), 3).unwrap();
    let mut ksp = CudaKspContext::new(runtime.clone());
    ksp.set_type(SolverType::Cg).unwrap();
    ksp.set_pc_type(PcType::None).unwrap();
    ksp.set_tolerances(1e-12, 1e-14, 1e8, 20).unwrap();
    ksp.set_operators(operator.clone(), None).unwrap();
    ksp.setup().unwrap();

    let before = runtime.diagnostics();
    let stats = ksp.solve(&b, &mut x).unwrap();
    assert!(stats.reason.is_converged());
    let after = runtime.diagnostics();
    assert_eq!(after.allocations, before.allocations);
    assert_eq!(after.host_to_device_bytes, before.host_to_device_bytes);
    assert_eq!(after.device_to_host_bytes, before.device_to_host_bytes);
    assert!(after.kernel_launches > before.kernel_launches);
    assert_eq!(after.solve_calls, before.solve_calls + 1);
    assert_eq!(after.setup_calls, before.setup_calls + 1);
    assert!(after.solve_time_ns >= before.solve_time_ns);
    assert!(after.setup_time_ns >= before.setup_time_ns);

    assert_close(
        &x.to_host().unwrap(),
        &[S::from_real(1.0), S::from_real(2.0), S::from_real(3.0)],
        1e-11,
    );

    x.fill_zero().unwrap();
    let before_repeat = runtime.diagnostics();
    let _ = ksp.solve(&b, &mut x).unwrap();
    let after_repeat = runtime.diagnostics();
    assert_eq!(after_repeat.allocations, before_repeat.allocations);
}

#[test]
#[ignore = "requires an NVIDIA GPU with CUDA 12+ libraries"]
fn pipelined_pcg_uses_a_single_device_scalar_payload_per_iteration() {
    let runtime = CudaRuntime::new(0).expect("CUDA runtime");
    let matrix = CsrMatrix::from_csr(
        3,
        3,
        vec![0, 2, 5, 7],
        vec![0, 1, 0, 1, 2, 1, 2],
        vec![
            S::from_real(2.0),
            S::from_real(-1.0),
            S::from_real(-1.0),
            S::from_real(2.0),
            S::from_real(-1.0),
            S::from_real(-1.0),
            S::from_real(2.0),
        ],
    );
    let operator = Arc::new(CudaCsrOp::from_host(runtime.clone(), &matrix).unwrap());
    let b = CudaVector::from_host(
        runtime.clone(),
        &[S::from_real(1.0), S::zero(), S::from_real(1.0)],
    )
    .unwrap();
    let mut x = CudaVector::zeros(runtime.clone(), 3).unwrap();
    let mut ksp = CudaKspContext::new(runtime.clone());
    ksp.set_type(SolverType::Pcg).unwrap();
    ksp.set_cg_variant(CudaCgVariant::Pipelined);
    ksp.set_pc_type(PcType::Jacobi).unwrap();
    ksp.set_tolerances(1e-12, 1e-14, 1e8, 12).unwrap();
    ksp.set_operators(operator.clone(), None).unwrap();
    ksp.setup().unwrap();
    let before = runtime.diagnostics();
    let stats = ksp.solve(&b, &mut x).unwrap();
    let after = runtime.diagnostics();
    assert!(stats.reason.is_converged(), "{stats:?}");
    assert_eq!(stats.reduction_model.unwrap().variant, "cuda-pipelined-cg");
    assert_eq!(after.allocations, before.allocations);
    assert_eq!(after.host_to_device_bytes, before.host_to_device_bytes);
    assert_close(&x.to_host().unwrap(), &[S::one(); 3], 1e-10);
}

#[test]
#[ignore = "requires an NVIDIA GPU with CUDA 12+ libraries"]
fn chebyshev_matches_host_stationary_semantics() {
    let runtime = CudaRuntime::new(0).expect("CUDA runtime");
    let matrix = CsrMatrix::from_csr(
        3,
        3,
        vec![0, 1, 2, 3],
        vec![0, 1, 2],
        vec![S::from_real(2.0), S::from_real(4.0), S::from_real(8.0)],
    );
    let operator = Arc::new(CudaCsrOp::from_host(runtime.clone(), &matrix).unwrap());
    let rhs = CudaVector::from_host(
        runtime.clone(),
        &[S::from_real(2.0), S::from_real(8.0), S::from_real(24.0)],
    )
    .unwrap();
    let mut solution = CudaVector::zeros(runtime.clone(), 3).unwrap();
    let mut ksp = CudaKspContext::new(runtime);
    ksp.set_type(SolverType::Chebyshev).unwrap();
    ksp.set_chebyshev_omega(1.0).unwrap();
    ksp.set_pc_type(PcType::Jacobi).unwrap();
    ksp.set_tolerances(1e-12, 1e-14, 1e8, 4).unwrap();
    ksp.set_operators(operator, None).unwrap();
    let stats = ksp.solve(&rhs, &mut solution).unwrap();
    assert!(stats.reason.is_converged(), "{stats:?}");
    assert_eq!(
        stats.reduction_model.unwrap().variant,
        "cuda-fused-chebyshev"
    );
    assert_close(
        &solution.to_host().unwrap(),
        &[S::one(), S::from_real(2.0), S::from_real(3.0)],
        1e-11,
    );
}

#[test]
#[ignore = "requires an NVIDIA GPU with CUDA 12+ libraries"]
fn nonsymmetric_bicgstab_matches_known_solution() {
    let runtime = CudaRuntime::new(0).expect("CUDA runtime");
    let matrix = CsrMatrix::from_csr(
        3,
        3,
        vec![0, 2, 5, 7],
        vec![0, 1, 0, 1, 2, 1, 2],
        vec![
            S::from_real(4.0),
            S::from_real(1.0),
            S::from_real(2.0),
            S::from_real(3.0),
            S::from_real(1.0),
            S::from_real(1.0),
            S::from_real(2.0),
        ],
    );
    let operator = Arc::new(CudaCsrOp::from_host(runtime.clone(), &matrix).unwrap());
    let rhs = CudaVector::from_host(
        runtime.clone(),
        &[S::from_real(6.0), S::from_real(11.0), S::from_real(8.0)],
    )
    .unwrap();
    let mut solution = CudaVector::zeros(runtime.clone(), 3).unwrap();
    let mut ksp = CudaKspContext::new(runtime.clone());
    ksp.set_type(SolverType::BiCgStab).unwrap();
    ksp.set_pc_type(PcType::Jacobi).unwrap();
    ksp.set_tolerances(1e-12, 1e-14, 1e8, 20).unwrap();
    ksp.set_operators(operator, None).unwrap();
    ksp.setup().unwrap();
    let before = runtime.diagnostics();
    let stats = ksp.solve(&rhs, &mut solution).unwrap();
    let after = runtime.diagnostics();
    assert!(stats.reason.is_converged(), "{stats:?}");
    assert_eq!(stats.reduction_model.unwrap().variant, "cuda-bicgstab");
    assert_eq!(after.allocations, before.allocations);
    assert_eq!(after.host_to_device_bytes, before.host_to_device_bytes);
    assert_close(
        &solution.to_host().unwrap(),
        &[S::one(), S::from_real(2.0), S::from_real(3.0)],
        1e-10,
    );
}

#[test]
#[ignore = "requires an NVIDIA GPU with CUDA 12+ libraries"]
fn right_preconditioned_cgs_matches_known_solution() {
    let runtime = CudaRuntime::new(0).expect("CUDA runtime");
    let matrix = CsrMatrix::from_csr(
        3,
        3,
        vec![0, 2, 5, 7],
        vec![0, 1, 0, 1, 2, 1, 2],
        vec![
            S::from_real(4.0),
            S::from_real(1.0),
            S::from_real(2.0),
            S::from_real(3.0),
            S::from_real(1.0),
            S::from_real(1.0),
            S::from_real(2.0),
        ],
    );
    let operator = Arc::new(CudaCsrOp::from_host(runtime.clone(), &matrix).unwrap());
    let rhs = CudaVector::from_host(
        runtime.clone(),
        &[S::from_real(6.0), S::from_real(11.0), S::from_real(8.0)],
    )
    .unwrap();
    let mut solution = CudaVector::zeros(runtime.clone(), 3).unwrap();
    let mut ksp = CudaKspContext::new(runtime.clone());
    ksp.set_type(SolverType::Cgs).unwrap();
    ksp.set_pc_type(PcType::Jacobi).unwrap();
    ksp.set_tolerances(1e-11, 1e-13, 1e8, 20).unwrap();
    ksp.set_operators(operator, None).unwrap();
    ksp.setup().unwrap();
    let before = runtime.diagnostics();
    let stats = ksp.solve(&rhs, &mut solution).unwrap();
    let after = runtime.diagnostics();
    assert!(stats.reason.is_converged(), "{stats:?}");
    assert_eq!(stats.reduction_model.unwrap().variant, "cuda-cgs");
    assert_eq!(after.allocations, before.allocations);
    assert_eq!(after.host_to_device_bytes, before.host_to_device_bytes);
    assert_close(
        &solution.to_host().unwrap(),
        &[S::one(), S::from_real(2.0), S::from_real(3.0)],
        1e-9,
    );
}

#[test]
#[ignore = "requires an NVIDIA GPU with CUDA 12+ libraries"]
fn rectangular_cgnr_cr_lsqr_and_lsmr_match_host_semantics() {
    let runtime = CudaRuntime::new(0).expect("CUDA runtime");
    // [1 0; 0 1; 1 1] * [2, -1] = [2, -1, 1].
    let matrix = CsrMatrix::from_csr(
        3,
        2,
        vec![0, 1, 2, 4],
        vec![0, 1, 0, 1],
        vec![S::one(), S::one(), S::one(), S::one()],
    );
    let operator = Arc::new(CudaCsrOp::from_host(runtime.clone(), &matrix).unwrap());
    let rhs = CudaVector::from_host(
        runtime.clone(),
        &[S::from_real(2.0), S::from_real(-1.0), S::one()],
    )
    .unwrap();

    for (solver, variant) in [
        (SolverType::Cgnr, "cuda-cgnr"),
        (SolverType::Cr, "cuda-cr-via-cgnr"),
        (SolverType::Lsqr, "cuda-lsqr"),
        (SolverType::Lsmr, "cuda-lsmr"),
    ] {
        let mut solution = CudaVector::zeros(runtime.clone(), 2).unwrap();
        let mut ksp = CudaKspContext::new(runtime.clone());
        ksp.set_type(solver).unwrap();
        ksp.set_pc_type(PcType::None).unwrap();
        ksp.set_tolerances(1e-12, 1e-14, 1e8, 20).unwrap();
        ksp.set_operators(operator.clone(), None).unwrap();
        ksp.setup().unwrap();
        let before = runtime.diagnostics();
        let stats = ksp.solve(&rhs, &mut solution).unwrap();
        let after = runtime.diagnostics();
        assert!(stats.reason.is_converged(), "{solver:?}: {stats:?}");
        assert_eq!(stats.reduction_model.unwrap().variant, variant);
        assert_eq!(after.allocations, before.allocations);
        assert_eq!(after.host_to_device_bytes, before.host_to_device_bytes);
        assert_close(
            &solution.to_host().unwrap(),
            &[S::from_real(2.0), S::from_real(-1.0)],
            1e-10,
        );
    }

    let mut host_solution = [S::zero(); 2];
    let mut host_ksp = CudaKspContext::new(runtime);
    host_ksp.set_type(SolverType::Cgnr).unwrap();
    host_ksp.set_pc_type(PcType::None).unwrap();
    host_ksp.set_tolerances(1e-12, 1e-14, 1e8, 20).unwrap();
    host_ksp.set_operators(operator, None).unwrap();
    let stats = host_ksp
        .solve_host(
            &[S::from_real(2.0), S::from_real(-1.0), S::one()],
            &mut host_solution,
        )
        .unwrap();
    assert!(stats.reason.is_converged(), "{stats:?}");
    assert_close(
        &host_solution,
        &[S::from_real(2.0), S::from_real(-1.0)],
        1e-10,
    );
}

#[test]
#[ignore = "requires an NVIDIA GPU with CUDA 12+ libraries"]
fn single_rank_distributed_operator_stays_device_resident() {
    let runtime = CudaRuntime::new(0).expect("CUDA runtime");
    let local = CsrMatrix::from_csr(
        3,
        3,
        vec![0, 2, 5, 7],
        vec![0, 1, 0, 1, 2, 1, 2],
        vec![
            S::from_real(2.0),
            S::from_real(-1.0),
            S::from_real(-1.0),
            S::from_real(2.0),
            S::from_real(-1.0),
            S::from_real(-1.0),
            S::from_real(2.0),
        ],
    );
    let operator = CudaDistCsrOp::from_local_rows(
        runtime.clone(),
        3,
        0,
        &local,
        &[0, 3],
        UniverseComm::NoComm(NoComm),
    )
    .unwrap();
    assert_eq!(operator.halo_send_volume(), 0);
    assert_eq!(operator.halo_recv_volume(), 0);
    let x = CudaVector::from_host(
        runtime.clone(),
        &[S::from_real(1.0), S::from_real(2.0), S::from_real(4.0)],
    )
    .unwrap();
    let mut y = CudaVector::zeros(runtime.clone(), 3).unwrap();
    let before = runtime.diagnostics();
    operator
        .apply(CudaOperation::NonTranspose, &x, &mut y)
        .unwrap();
    let after = runtime.diagnostics();
    assert_eq!(after.allocations, before.allocations);
    assert_eq!(after.device_to_host_bytes, before.device_to_host_bytes);
    assert_close(
        &y.to_host().unwrap(),
        &[S::from_real(0.0), S::from_real(-1.0), S::from_real(6.0)],
        1e-12,
    );
}

#[test]
#[ignore = "requires an NVIDIA GPU with CUDA 12+ libraries"]
fn jacobi_richardson_uses_fused_device_updates() {
    let runtime = CudaRuntime::new(0).expect("CUDA runtime");
    let matrix = CsrMatrix::from_csr(
        3,
        3,
        vec![0, 1, 2, 3],
        vec![0, 1, 2],
        vec![S::from_real(2.0), S::from_real(4.0), S::from_real(8.0)],
    );
    let operator = Arc::new(CudaCsrOp::from_host(runtime.clone(), &matrix).unwrap());
    let b = CudaVector::from_host(
        runtime.clone(),
        &[S::from_real(2.0), S::from_real(8.0), S::from_real(24.0)],
    )
    .unwrap();
    let mut x = CudaVector::zeros(runtime.clone(), 3).unwrap();
    let mut ksp = CudaKspContext::new(runtime.clone());
    ksp.set_type(SolverType::Richardson).unwrap();
    ksp.set_pc_type(PcType::Jacobi).unwrap();
    ksp.set_richardson_omega(1.0).unwrap();
    ksp.set_tolerances(1e-12, 1e-14, 1e8, 5).unwrap();
    ksp.set_operators(operator, None).unwrap();
    ksp.setup().unwrap();
    let before = runtime.diagnostics();
    let stats = ksp.solve(&b, &mut x).unwrap();
    let after = runtime.diagnostics();
    assert!(stats.reason.is_converged());
    assert_eq!(stats.iterations, 1);
    assert_eq!(after.allocations, before.allocations);
    assert!(after.kernel_launches >= before.kernel_launches + 4);
    assert_close(
        &x.to_host().unwrap(),
        &[S::from_real(1.0), S::from_real(2.0), S::from_real(3.0)],
        1e-12,
    );
}

#[test]
#[ignore = "requires an NVIDIA GPU with CUDA 12+ libraries"]
fn block_jacobi_factorization_and_apply_remain_device_resident() {
    let runtime = CudaRuntime::new(0).expect("CUDA runtime");
    let matrix = CsrMatrix::from_csr(
        4,
        4,
        vec![0, 2, 4, 6, 8],
        vec![0, 1, 0, 1, 2, 3, 2, 3],
        vec![
            S::from_real(4.0),
            S::from_real(1.0),
            S::from_real(2.0),
            S::from_real(3.0),
            S::from_real(5.0),
            S::from_real(-1.0),
            S::from_real(1.0),
            S::from_real(2.0),
        ],
    );
    let operator = Arc::new(CudaCsrOp::from_host(runtime.clone(), &matrix).unwrap());
    let b = CudaVector::from_host(
        runtime.clone(),
        &[
            S::from_real(5.0),
            S::from_real(5.0),
            S::from_real(4.0),
            S::from_real(3.0),
        ],
    )
    .unwrap();
    let mut x = CudaVector::zeros(runtime.clone(), 4).unwrap();
    let mut ksp = CudaKspContext::new(runtime.clone());
    ksp.set_type(SolverType::Richardson).unwrap();
    ksp.set_pc_type(PcType::BlockJacobi).unwrap();
    ksp.set_block_jacobi_size(2).unwrap();
    ksp.set_tolerances(1e-12, 1e-14, 1e8, 4).unwrap();
    ksp.set_operators(operator, None).unwrap();
    ksp.setup().unwrap();
    let before = runtime.diagnostics();
    let stats = ksp.solve(&b, &mut x).unwrap();
    let after = runtime.diagnostics();
    assert!(stats.reason.is_converged());
    assert_eq!(stats.iterations, 1);
    assert_eq!(after.allocations, before.allocations);
    assert_close(&x.to_host().unwrap(), &[S::one(); 4], 1e-11);
}

#[test]
#[ignore = "requires an NVIDIA GPU with CUDA 12+ libraries"]
fn chebyshev_pc_reuses_device_recurrence_workspace() {
    let runtime = CudaRuntime::new(0).expect("CUDA runtime");
    let matrix = CsrMatrix::from_csr(
        2,
        2,
        vec![0, 1, 2],
        vec![0, 1],
        vec![S::from_real(2.0), S::from_real(4.0)],
    );
    let operator = Arc::new(CudaCsrOp::from_host(runtime.clone(), &matrix).unwrap());
    let rhs =
        CudaVector::from_host(runtime.clone(), &[S::from_real(2.0), S::from_real(8.0)]).unwrap();
    let mut solution = CudaVector::zeros(runtime.clone(), 2).unwrap();
    let mut ksp = CudaKspContext::new(runtime.clone());
    ksp.set_type(SolverType::Fgmres).unwrap();
    ksp.set_pc_type(PcType::Chebyshev).unwrap();
    ksp.set_chebyshev_pc(2, 1.0, 5.0).unwrap();
    ksp.set_restart(3).unwrap();
    ksp.set_tolerances(1e-12, 1e-14, 1e8, 10).unwrap();
    ksp.set_operators(operator, None).unwrap();
    ksp.setup().unwrap();
    let before = runtime.diagnostics();
    let stats = ksp.solve(&rhs, &mut solution).unwrap();
    let after = runtime.diagnostics();
    assert!(stats.reason.is_converged(), "{stats:?}");
    assert_eq!(after.allocations, before.allocations);
    assert_eq!(after.host_to_device_bytes, before.host_to_device_bytes);
    assert_close(
        &solution.to_host().unwrap(),
        &[S::one(), S::from_real(2.0)],
        1e-10,
    );
}

#[test]
#[ignore = "requires an NVIDIA GPU with CUDA 12+ libraries"]
fn host_factorized_ilu0_uses_device_triangular_solves() {
    let runtime = CudaRuntime::new(0).expect("CUDA runtime");
    let matrix = CsrMatrix::from_csr(
        3,
        3,
        vec![0, 2, 5, 7],
        vec![0, 1, 0, 1, 2, 1, 2],
        vec![
            S::from_real(4.0),
            S::one(),
            S::from_real(2.0),
            S::from_real(3.0),
            S::one(),
            S::one(),
            S::from_real(2.0),
        ],
    );
    let operator = Arc::new(CudaCsrOp::from_host(runtime.clone(), &matrix).unwrap());
    let rhs = CudaVector::from_host(
        runtime.clone(),
        &[S::from_real(5.0), S::from_real(6.0), S::from_real(3.0)],
    )
    .unwrap();
    let mut solution = CudaVector::zeros(runtime.clone(), 3).unwrap();
    let mut ksp = CudaKspContext::new(runtime.clone());
    ksp.set_type(SolverType::Richardson).unwrap();
    ksp.set_pc_type(PcType::Ilu0).unwrap();
    ksp.set_richardson_omega(1.0).unwrap();
    ksp.set_tolerances(1e-12, 1e-14, 1e8, 4).unwrap();
    ksp.set_operators(operator.clone(), None).unwrap();
    ksp.setup().unwrap();
    let before = runtime.diagnostics();
    let stats = ksp.solve(&rhs, &mut solution).unwrap();
    let after = runtime.diagnostics();
    assert!(stats.reason.is_converged(), "{stats:?}");
    assert_eq!(stats.iterations, 1);
    assert_eq!(after.allocations, before.allocations);
    assert_eq!(after.host_to_device_bytes, before.host_to_device_bytes);
    assert_close(&solution.to_host().unwrap(), &[S::one(); 3], 1e-10);

    // A numeric-only matrix update must reuse the analyzed triangular solve
    // descriptors and their allocations. Setup performs the expected value
    // uploads; the subsequent iteration remains allocation/transfer free.
    operator
        .update_values(&[
            S::from_real(8.0),
            S::from_real(2.0),
            S::from_real(4.0),
            S::from_real(6.0),
            S::from_real(2.0),
            S::from_real(2.0),
            S::from_real(4.0),
        ])
        .unwrap();
    ksp.setup().unwrap();
    solution.fill_zero().unwrap();
    let before_repeated_solve = runtime.diagnostics();
    let repeated_stats = ksp.solve(&rhs, &mut solution).unwrap();
    let after_repeated_solve = runtime.diagnostics();
    assert!(repeated_stats.reason.is_converged(), "{repeated_stats:?}");
    assert_eq!(repeated_stats.iterations, 1);
    assert_eq!(
        after_repeated_solve.allocations,
        before_repeated_solve.allocations
    );
    assert_eq!(
        after_repeated_solve.host_to_device_bytes,
        before_repeated_solve.host_to_device_bytes
    );
    assert_close(&solution.to_host().unwrap(), &[S::from_real(0.5); 3], 1e-10);
}

#[test]
#[ignore = "requires an NVIDIA GPU with CUDA 12+ libraries"]
fn uploaded_amg_v_cycle_reuses_all_level_workspace() {
    let runtime = CudaRuntime::new(0).expect("CUDA runtime");
    let n = 8usize;
    let mut rows = Vec::with_capacity(n + 1);
    let mut columns = Vec::new();
    let mut values = Vec::new();
    rows.push(0);
    for row in 0..n {
        if row > 0 {
            columns.push(row - 1);
            values.push(S::from_real(-1.0));
        }
        columns.push(row);
        values.push(S::from_real(2.0));
        if row + 1 < n {
            columns.push(row + 1);
            values.push(S::from_real(-1.0));
        }
        rows.push(columns.len());
    }
    let matrix = CsrMatrix::from_csr(n, n, rows, columns, values);
    let operator = Arc::new(CudaCsrOp::from_host(runtime.clone(), &matrix).unwrap());
    let amg = CudaAmg::from_csr_with_options(
        operator.as_ref(),
        CudaAmgOptions {
            coarse_size: 2,
            coarse_iterations: 24,
            ..CudaAmgOptions::default()
        },
    )
    .unwrap();
    assert!(amg.level_count() >= 3);
    let rhs_host = [
        S::one(),
        S::zero(),
        S::zero(),
        S::zero(),
        S::zero(),
        S::zero(),
        S::zero(),
        S::one(),
    ];
    let rhs = CudaVector::from_host(runtime.clone(), &rhs_host).unwrap();
    let mut solution = CudaVector::zeros(runtime.clone(), n).unwrap();
    let mut ksp = CudaKspContext::new(runtime.clone());
    ksp.set_type(SolverType::Fgmres).unwrap();
    ksp.set_restart(8).unwrap();
    ksp.set_tolerances(1e-10, 1e-13, 1e8, 50).unwrap();
    ksp.set_operators(operator, None).unwrap();
    ksp.set_preconditioner(Arc::new(amg)).unwrap();
    ksp.setup().unwrap();
    let before = runtime.diagnostics();
    let stats = ksp.solve(&rhs, &mut solution).unwrap();
    let after = runtime.diagnostics();
    assert!(stats.reason.is_converged(), "{stats:?}");
    assert_eq!(after.allocations, before.allocations);
    assert_eq!(after.host_to_device_bytes, before.host_to_device_bytes);
    assert_close(&solution.to_host().unwrap(), &[S::one(); 8], 1e-8);
}

#[test]
#[ignore = "requires an NVIDIA GPU with CUDA 12+ libraries"]
fn fgmres_jacobi_solves_nonsymmetric_csr_and_transpose_applies() {
    let runtime = CudaRuntime::new(0).expect("CUDA runtime");
    let matrix = CsrMatrix::from_csr(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![
            S::from_real(4.0),
            S::from_real(1.0),
            S::from_real(2.0),
            S::from_real(3.0),
        ],
    );
    let operator = Arc::new(CudaCsrOp::from_host(runtime.clone(), &matrix).unwrap());
    let b =
        CudaVector::from_host(runtime.clone(), &[S::from_real(1.0), S::from_real(2.0)]).unwrap();
    let mut x = CudaVector::zeros(runtime.clone(), 2).unwrap();
    let mut ksp = CudaKspContext::new(runtime.clone());
    ksp.set_type(SolverType::Fgmres).unwrap();
    ksp.set_pc_type(PcType::Jacobi).unwrap();
    ksp.set_tolerances(1e-12, 1e-14, 1e8, 20).unwrap();
    ksp.set_restart(4).unwrap();
    ksp.set_operators(operator.clone(), None).unwrap();
    let stats = ksp.solve(&b, &mut x).unwrap();
    assert!(stats.reason.is_converged());
    assert_close(
        &x.to_host().unwrap(),
        &[S::from_real(0.1), S::from_real(0.6)],
        1e-10,
    );

    let input = CudaVector::from_host(runtime.clone(), &[S::one(), S::one()]).unwrap();
    let mut output = CudaVector::zeros(runtime, 2).unwrap();
    operator
        .apply(CudaOperation::Transpose, &input, &mut output)
        .unwrap();
    assert_close(
        &output.to_host().unwrap(),
        &[S::from_real(6.0), S::from_real(4.0)],
        1e-12,
    );

    let structure_id = operator.structure_id();
    let values_id = operator.values_id();
    operator
        .update_values(&[
            S::from_real(8.0),
            S::from_real(2.0),
            S::from_real(4.0),
            S::from_real(6.0),
        ])
        .unwrap();
    assert_eq!(operator.structure_id(), structure_id);
    assert_ne!(operator.values_id(), values_id);
    x.fill_zero().unwrap();
    let before_update_solve = operator.runtime().diagnostics();
    let update_stats = ksp.solve(&b, &mut x).unwrap();
    let after_update_solve = operator.runtime().diagnostics();
    assert!(update_stats.reason.is_converged());
    assert_eq!(
        after_update_solve.allocations,
        before_update_solve.allocations
    );
    assert_close(
        &x.to_host().unwrap(),
        &[S::from_real(0.05), S::from_real(0.3)],
        1e-10,
    );
}

#[test]
#[ignore = "requires an NVIDIA GPU with CUDA 12+ libraries"]
fn pipelined_gmres_family_uses_one_arnoldi_collective_and_reuses_workspace() {
    let matrix = CsrMatrix::from_csr(
        3,
        3,
        vec![0, 2, 5, 7],
        vec![0, 1, 0, 1, 2, 1, 2],
        vec![
            S::from_real(4.0),
            S::from_real(1.0),
            S::from_real(2.0),
            S::from_real(3.0),
            S::from_real(1.0),
            S::from_real(1.0),
            S::from_real(2.0),
        ],
    );

    for solver_type in [SolverType::Gmres, SolverType::Fgmres] {
        let runtime = CudaRuntime::new(0).expect("CUDA runtime");
        let operator = Arc::new(CudaCsrOp::from_host(runtime.clone(), &matrix).unwrap());
        let rhs = CudaVector::from_host(
            runtime.clone(),
            &[S::from_real(6.0), S::from_real(11.0), S::from_real(8.0)],
        )
        .unwrap();
        let mut solution = CudaVector::zeros(runtime.clone(), 3).unwrap();
        let mut ksp = CudaKspContext::new(runtime.clone());
        ksp.set_type(solver_type).unwrap();
        ksp.set_gmres_variant(CudaGmresVariant::Pipelined).unwrap();
        ksp.set_pc_type(PcType::Jacobi).unwrap();
        ksp.set_restart(3).unwrap();
        ksp.set_tolerances(1e-12, 1e-14, 1e8, 20).unwrap();
        ksp.set_operators(operator, None).unwrap();
        ksp.setup().unwrap();

        let before = runtime.diagnostics();
        let stats = ksp.solve(&rhs, &mut solution).unwrap();
        let after = runtime.diagnostics();

        assert!(stats.reason.is_converged(), "{solver_type:?}: {stats:?}");
        assert_eq!(
            stats.reduction_model.unwrap().variant,
            "cuda-pipelined-cgs-gmres"
        );
        assert_eq!(after.allocations, before.allocations);
        assert_eq!(after.host_to_device_bytes, before.host_to_device_bytes);
        assert_close(
            &solution.to_host().unwrap(),
            &[S::one(), S::from_real(2.0), S::from_real(3.0)],
            1e-10,
        );
    }
}

#[test]
#[ignore = "requires an NVIDIA GPU with CUDA 12+ libraries"]
fn gcr_reuses_the_device_resident_fgmres_workspace() {
    let runtime = CudaRuntime::new(0).expect("CUDA runtime");
    let matrix = CsrMatrix::from_csr(
        3,
        3,
        vec![0, 2, 5, 7],
        vec![0, 1, 0, 1, 2, 1, 2],
        vec![
            S::from_real(4.0),
            S::from_real(1.0),
            S::from_real(2.0),
            S::from_real(3.0),
            S::from_real(1.0),
            S::from_real(1.0),
            S::from_real(2.0),
        ],
    );
    let operator = Arc::new(CudaCsrOp::from_host(runtime.clone(), &matrix).unwrap());
    let rhs = CudaVector::from_host(
        runtime.clone(),
        &[S::from_real(6.0), S::from_real(11.0), S::from_real(8.0)],
    )
    .unwrap();
    let mut solution = CudaVector::zeros(runtime.clone(), 3).unwrap();
    let mut ksp = CudaKspContext::new(runtime.clone());
    ksp.set_type(SolverType::Gcr).unwrap();
    ksp.set_pc_type(PcType::Jacobi).unwrap();
    ksp.set_restart(4).unwrap();
    ksp.set_tolerances(1e-12, 1e-14, 1e8, 20).unwrap();
    ksp.set_operators(operator, None).unwrap();
    ksp.setup().unwrap();
    let before = runtime.diagnostics();
    let stats = ksp.solve(&rhs, &mut solution).unwrap();
    let after = runtime.diagnostics();
    assert!(stats.reason.is_converged(), "{stats:?}");
    assert_eq!(
        stats.reduction_model.unwrap().variant,
        "cuda-gcr-via-fgmres"
    );
    assert_eq!(after.allocations, before.allocations);
    assert_eq!(after.host_to_device_bytes, before.host_to_device_bytes);
    assert_close(
        &solution.to_host().unwrap(),
        &[S::one(), S::from_real(2.0), S::from_real(3.0)],
        1e-10,
    );
}

#[test]
#[ignore = "requires an NVIDIA GPU with CUDA 12+ libraries"]
fn pipegcr_uses_device_resident_gcr_bases_without_hot_loop_allocations() {
    let runtime = CudaRuntime::new(0).expect("CUDA runtime");
    let matrix = CsrMatrix::from_csr(
        3,
        3,
        vec![0, 2, 5, 7],
        vec![0, 1, 0, 1, 2, 1, 2],
        vec![
            S::from_real(4.0),
            S::from_real(1.0),
            S::from_real(2.0),
            S::from_real(3.0),
            S::from_real(1.0),
            S::from_real(1.0),
            S::from_real(2.0),
        ],
    );
    let operator = Arc::new(CudaCsrOp::from_host(runtime.clone(), &matrix).unwrap());
    let rhs = CudaVector::from_host(
        runtime.clone(),
        &[S::from_real(6.0), S::from_real(11.0), S::from_real(8.0)],
    )
    .unwrap();
    let mut solution = CudaVector::zeros(runtime.clone(), 3).unwrap();
    let mut ksp = CudaKspContext::new(runtime.clone());
    ksp.set_type(SolverType::PipeGcr).unwrap();
    ksp.set_pc_type(PcType::Jacobi).unwrap();
    ksp.set_restart(3).unwrap();
    ksp.set_tolerances(1e-12, 1e-14, 1e8, 20).unwrap();
    ksp.set_operators(operator, None).unwrap();
    ksp.setup().unwrap();
    let before = runtime.diagnostics();
    let stats = ksp.solve(&rhs, &mut solution).unwrap();
    let after = runtime.diagnostics();
    assert!(stats.reason.is_converged(), "{stats:?}");
    assert_eq!(stats.reduction_model.unwrap().variant, "cuda-pipegcr");
    assert_eq!(after.allocations, before.allocations);
    assert_eq!(after.host_to_device_bytes, before.host_to_device_bytes);
    assert_close(
        &solution.to_host().unwrap(),
        &[S::one(), S::from_real(2.0), S::from_real(3.0)],
        1e-10,
    );
}

#[test]
#[ignore = "requires an NVIDIA GPU with CUDA 12+ libraries"]
fn qmr_uses_transpose_capable_csr_without_host_vector_transfers() {
    let runtime = CudaRuntime::new(0).expect("CUDA runtime");
    let matrix = CsrMatrix::from_csr(
        3,
        3,
        vec![0, 2, 5, 7],
        vec![0, 1, 0, 1, 2, 1, 2],
        vec![
            S::from_real(4.0),
            S::from_real(1.0),
            S::from_real(2.0),
            S::from_real(3.0),
            S::from_real(1.0),
            S::from_real(1.0),
            S::from_real(2.0),
        ],
    );
    let operator = Arc::new(CudaCsrOp::from_host(runtime.clone(), &matrix).unwrap());
    let rhs = CudaVector::from_host(
        runtime.clone(),
        &[S::from_real(6.0), S::from_real(11.0), S::from_real(8.0)],
    )
    .unwrap();
    let mut solution = CudaVector::zeros(runtime.clone(), 3).unwrap();
    let mut ksp = CudaKspContext::new(runtime.clone());
    ksp.set_type(SolverType::Qmr).unwrap();
    ksp.set_pc_type(PcType::None).unwrap();
    ksp.set_tolerances(1e-11, 1e-13, 1e8, 100).unwrap();
    ksp.set_operators(operator, None).unwrap();
    ksp.setup().unwrap();
    let before = runtime.diagnostics();
    let stats = ksp.solve(&rhs, &mut solution).unwrap();
    let after = runtime.diagnostics();
    assert!(stats.reason.is_converged(), "{stats:?}");
    assert_eq!(stats.reduction_model.unwrap().variant, "cuda-qmr-compat");
    assert_eq!(after.allocations, before.allocations);
    assert_eq!(after.host_to_device_bytes, before.host_to_device_bytes);
    assert_close(
        &solution.to_host().unwrap(),
        &[S::one(), S::from_real(2.0), S::from_real(3.0)],
        1e-9,
    );
}

#[test]
#[ignore = "requires an NVIDIA GPU with CUDA 12+ libraries"]
fn tfqmr_family_converges_on_identity_without_hot_loop_allocations() {
    let runtime = CudaRuntime::new(0).expect("CUDA runtime");
    let matrix = CsrMatrix::from_csr(3, 3, vec![0, 1, 2, 3], vec![0, 1, 2], vec![S::one(); 3]);
    let operator = Arc::new(CudaCsrOp::from_host(runtime.clone(), &matrix).unwrap());
    let rhs = CudaVector::from_host(
        runtime.clone(),
        &[S::one(), S::from_real(2.0), S::from_real(3.0)],
    )
    .unwrap();

    for solver in [SolverType::Tfqmr, SolverType::Tcqmr] {
        let mut solution = CudaVector::zeros(runtime.clone(), 3).unwrap();
        let mut ksp = CudaKspContext::new(runtime.clone());
        ksp.set_type(solver).unwrap();
        ksp.set_pc_type(PcType::None).unwrap();
        ksp.set_tolerances(1e-12, 1e-14, 1e8, 20).unwrap();
        ksp.set_operators(operator.clone(), None).unwrap();
        ksp.setup().unwrap();
        let before = runtime.diagnostics();
        let stats = ksp.solve(&rhs, &mut solution).unwrap();
        let after = runtime.diagnostics();
        assert!(stats.reason.is_converged(), "{solver:?}: {stats:?}");
        assert_eq!(after.allocations, before.allocations);
        assert_eq!(after.host_to_device_bytes, before.host_to_device_bytes);
        assert_close(
            &solution.to_host().unwrap(),
            &[S::one(), S::from_real(2.0), S::from_real(3.0)],
            1e-11,
        );
    }
}

#[cfg(feature = "complex")]
#[test]
#[ignore = "requires an NVIDIA GPU with CUDA 12+ libraries"]
fn conjugate_transpose_csr_matches_reference() {
    let runtime = CudaRuntime::new(0).expect("CUDA runtime");
    let matrix = CsrMatrix::from_csr(
        2,
        2,
        vec![0, 2, 3],
        vec![0, 1, 1],
        vec![
            S::from_parts(1.0, 1.0),
            S::from_real(2.0),
            S::from_parts(3.0, -1.0),
        ],
    );
    let operator = CudaCsrOp::from_host(runtime.clone(), &matrix).unwrap();
    let input = CudaVector::from_host(runtime.clone(), &[S::one(), S::one()]).unwrap();
    let mut output = CudaVector::zeros(runtime, 2).unwrap();
    operator
        .apply(CudaOperation::ConjugateTranspose, &input, &mut output)
        .unwrap();
    assert_close(
        &output.to_host().unwrap(),
        &[S::from_parts(1.0, -1.0), S::from_parts(5.0, 1.0)],
        1e-12,
    );
}
