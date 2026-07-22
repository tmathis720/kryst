//! Synchronized CPU/CUDA crossover report for a 2-D Poisson problem.
//!
//! ```text
//! cargo run --release --example cuda_crossover --features cuda -- 16 32 64
//! ```
//!
//! `KRYST_CUDA_BENCH_REPEATS` controls repeated solves (default: 5). Output is
//! CSV so a GPU CI job can retain it as a benchmark artifact.

#[cfg(feature = "complex")]
fn main() {
    eprintln!("cuda_crossover currently measures the f64 CUDA v1 path");
}

#[cfg(not(feature = "complex"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use kryst::algebra::prelude::*;
    use kryst::context::ksp_context::{KspContext, SolverType};
    use kryst::context::pc_context::PcType;
    use kryst::cuda::{CudaCsrOp, CudaKspContext, CudaRuntime, CudaVector};
    use kryst::matrix::LinOp;
    use kryst::matrix::op::CsrOp;
    use std::sync::Arc;
    use std::time::Instant;

    let repeats = std::env::var("KRYST_CUDA_BENCH_REPEATS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(5)
        .max(1);
    let grids: Vec<usize> = {
        let parsed: Vec<_> = std::env::args()
            .skip(1)
            .filter_map(|value| value.parse::<usize>().ok())
            .filter(|&value| value > 0)
            .collect();
        if parsed.is_empty() {
            vec![16, 32, 64, 96]
        } else {
            parsed
        }
    };
    let runtime = CudaRuntime::new(0)?;

    println!(
        "grid,n,nnz,cpu_setup_ms,cpu_first_ms,cpu_repeat_ms,gpu_setup_ms,gpu_first_ms,gpu_repeat_ms,gpu_alloc_bytes,gpu_h2d_bytes,gpu_d2h_bytes,gpu_faster"
    );
    for grid in grids {
        let matrix = poisson_2d(grid);
        let n = matrix.nrows();
        let nnz = matrix.values().len();
        let rhs = vec![S::one(); n];

        let cpu_operator = Arc::new(CsrOp::new(Arc::new(matrix.clone()))) as Arc<dyn LinOp<S = S>>;
        let mut cpu = KspContext::new();
        cpu.set_type(SolverType::Cg)?;
        cpu.set_pc_type(PcType::None, None)?;
        cpu.set_tolerances(1e-8, 1e-12, 1e8, n.max(100));
        cpu.set_operators(cpu_operator, None);
        let cpu_setup = elapsed(|| cpu.setup().map(|_| ()))?;
        let mut cpu_x = vec![S::zero(); n];
        let cpu_first = elapsed(|| {
            let stats = cpu.solve(&rhs, &mut cpu_x)?;
            assert!(
                stats.reason.is_converged(),
                "CPU reference failed: {stats:?}"
            );
            Ok::<(), kryst::error::KError>(())
        })?;
        let cpu_repeated = average(repeats, || {
            cpu_x.fill(S::zero());
            cpu.solve(&rhs, &mut cpu_x).map(|_| ())
        })?;

        let before = runtime.diagnostics();
        let setup_started = Instant::now();
        let gpu_operator = Arc::new(CudaCsrOp::from_host(runtime.clone(), &matrix)?);
        let gpu_rhs = CudaVector::from_host(runtime.clone(), &rhs)?;
        let mut gpu_x = CudaVector::zeros(runtime.clone(), n)?;
        let mut gpu = CudaKspContext::new(runtime.clone());
        gpu.set_type(SolverType::Cg)?;
        gpu.set_pc_type(PcType::None)?;
        gpu.set_tolerances(1e-8, 1e-12, 1e8, n.max(100))?;
        gpu.set_operators(gpu_operator, None)?;
        gpu.setup()?;
        runtime.synchronize()?;
        let gpu_setup = setup_started.elapsed();

        let gpu_first = elapsed(|| {
            let stats = gpu.solve(&gpu_rhs, &mut gpu_x)?;
            assert!(stats.reason.is_converged(), "CUDA solve failed: {stats:?}");
            Ok::<(), kryst::error::KError>(())
        })?;
        let gpu_repeated = average(repeats, || {
            gpu_x.fill_zero()?;
            gpu.solve(&gpu_rhs, &mut gpu_x).map(|_| ())
        })?;
        runtime.synchronize()?;
        let after = runtime.diagnostics();
        let allocated = after.allocated_bytes.saturating_sub(before.allocated_bytes);
        let h2d = after
            .host_to_device_bytes
            .saturating_sub(before.host_to_device_bytes);
        let d2h = after
            .device_to_host_bytes
            .saturating_sub(before.device_to_host_bytes);

        println!(
            "{grid},{n},{nnz},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{allocated},{h2d},{d2h},{}",
            millis(cpu_setup),
            millis(cpu_first),
            millis(cpu_repeated),
            millis(gpu_setup),
            millis(gpu_first),
            millis(gpu_repeated),
            gpu_repeated < cpu_repeated,
        );
    }
    Ok(())
}

#[cfg(not(feature = "complex"))]
fn elapsed<E>(mut operation: impl FnMut() -> Result<(), E>) -> Result<std::time::Duration, E> {
    let started = std::time::Instant::now();
    operation()?;
    Ok(started.elapsed())
}

#[cfg(not(feature = "complex"))]
fn average<E>(
    repeats: usize,
    mut operation: impl FnMut() -> Result<(), E>,
) -> Result<std::time::Duration, E> {
    let started = std::time::Instant::now();
    for _ in 0..repeats {
        operation()?;
    }
    Ok(started.elapsed() / repeats as u32)
}

#[cfg(not(feature = "complex"))]
fn millis(duration: std::time::Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

#[cfg(not(feature = "complex"))]
fn poisson_2d(grid: usize) -> kryst::matrix::sparse::CsrMatrix<f64> {
    use kryst::matrix::sparse::CsrMatrix;

    let n = grid * grid;
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::with_capacity(n * 5);
    let mut values = Vec::with_capacity(n * 5);
    row_ptr.push(0);
    for row in 0..grid {
        for column in 0..grid {
            let index = row * grid + column;
            if row > 0 {
                col_idx.push(index - grid);
                values.push(-1.0);
            }
            if column > 0 {
                col_idx.push(index - 1);
                values.push(-1.0);
            }
            col_idx.push(index);
            values.push(4.0);
            if column + 1 < grid {
                col_idx.push(index + 1);
                values.push(-1.0);
            }
            if row + 1 < grid {
                col_idx.push(index + grid);
                values.push(-1.0);
            }
            row_ptr.push(col_idx.len());
        }
    }
    CsrMatrix::from_csr(n, n, row_ptr, col_idx, values)
}
