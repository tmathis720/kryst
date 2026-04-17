//! to run:
//! cargo mpirun -n 4 --example complex_matrix_market_demo --features=complex,mpi,mpi_examples

#![cfg_attr(not(feature = "complex"), allow(dead_code))]

#[cfg(not(feature = "complex"))]
fn main() {
    eprintln!(
        "This example requires the `complex` feature. \\nre-run with `cargo run --features complex --example complex_matrix_market_demo`."
    );
}

#[cfg(feature = "complex")]
use kryst::error::KError;

#[cfg(feature = "complex")]
mod complex_demo {
    use std::env;
    use std::path::Path;
    use std::sync::Arc;
    use std::time::Instant;

    use super::KError;
    use kryst::algebra::blas::nrm2;
    use kryst::algebra::bridge::BridgeScratch;
    use kryst::algebra::prelude::*;
    use kryst::context::ksp_context::{ReorthPolicy, Workspace};
    use kryst::matrix::csr::CsrMatrix as ScalarCsrMatrix;
    use kryst::matrix::op::GenericCsrOp;
    use kryst::matrix::sparse::CsrMatrix as SparseCsrMatrix;
    use kryst::matrix::spmv::SpmvTuning;
    use kryst::ops::klinop::KLinOp;
    use kryst::ops::kpc::KPreconditioner;
    use kryst::parallel::{Comm, UniverseComm};
    use kryst::preconditioner::PcSide;
    use kryst::preconditioner::Preconditioner;
    use kryst::preconditioner::jacobi::Jacobi;
    use kryst::solver::LinearSolver;
    use kryst::solver::fgmres::{FgmresSolver, FgmresVariant};
    use kryst::utils::convergence::ConvergedReason;
    use kryst::utils::matrix_market::read_matrix_market;
    use kryst::utils::reduction::ReductOptions;

    #[cfg(feature = "mpi")]
    use kryst::parallel::MpiComm;
    #[cfg(not(feature = "mpi"))]
    use kryst::parallel::NoComm;

    pub fn run() -> Result<(), KError> {
        #[cfg(feature = "logging")]
        let _ = env_logger::try_init();

        #[cfg(feature = "mpi")]
        let comm = UniverseComm::Mpi(Arc::new(MpiComm::new()));
        #[cfg(not(feature = "mpi"))]
        let comm = UniverseComm::NoComm(NoComm);

        let rank = comm.rank();
        let size = comm.size();
        let config = BenchmarkConfig::from_env_args()?;
        #[cfg(feature = "mpi")]
        let is_parallel = matches!(comm, UniverseComm::Mpi(_)) && size > 1;
        #[cfg(not(feature = "mpi"))]
        let is_parallel = false;

        if rank == 0 {
            println!("Complex Matrix Market demo (FGMRES)");
            println!(
                "Parallel backend: {}",
                if is_parallel {
                    format!("MPI ({} ranks)", size)
                } else {
                    "serial".into()
                }
            );
            println!(
                "Benchmark runs: warmup={}, measured={}",
                config.warmup_runs, config.measured_runs
            );
            println!("===============================================================");
            println!();
        }

        let base = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/mtx");
        let cases = [
            ("qc324.mtx", "qc324 (complex, general)"),
            ("dwg961a.mtx", "dwg961a (complex, general)"),
        ];

        for (mat_name, descr) in cases {
            let mat_path = base.join(mat_name);
            let available = mat_path.exists();
            if !available {
                if rank == 0 {
                    println!("⚠ Missing file {mat_name} for {descr}, skipping.\n");
                }
                continue;
            }

            let problem = match load_problem_complex(&mat_path, &comm) {
                Ok(p) => p,
                Err(err) => {
                    if rank == 0 {
                        println!("❌ Failed to load {descr}: {err}\n");
                    }
                    continue;
                }
            };

            let rhs_norm = nrm2(&problem.rhs);
            if rank == 0 {
                println!("=== {descr} — {} ===", problem.backend_descr);
                println!(
                    "Run backend: {} ({})",
                    problem.backend.run_label(),
                    problem.backend.details()
                );
                println!(
                    "Benchmark/export label: {}",
                    problem.backend.benchmark_export_label()
                );
                println!("Global DOFs: {}", problem.global_n);
                println!("Local DOFs (rank {rank}): {}", problem.local_n);
                if problem.comm.size() > 1 && problem.local_n == problem.global_n {
                    println!("Note: replicated execution: MPI ranks are not sharing SpMV rows.");
                }
                println!("‖rhs‖₂ = {:.3e}", rhs_norm);
                println!("Residual semantics: rec/reported = solver recurrence/monitor residual.");
                println!(
                    "                    true(explicit) = ||b - A x||₂ recomputed after solve."
                );
                let include_dof_col = matches!(
                    problem.backend,
                    CsrBackend::Serial | CsrBackend::Distributed
                );
                if include_dof_col {
                    println!(
                        "{:<36} {:>9} {:>9} {:>9} {:>7} {:>6} {:>14} {:>14} {:>26} {:>12}",
                        "Method",
                        "Setup(s)",
                        "Med(s)",
                        "Min(s)",
                        "Iters",
                        "Reds",
                        "Rec/Reported",
                        "True(explicit)",
                        "Reason",
                        "DOF/s"
                    );
                } else {
                    println!(
                        "{:<36} {:>9} {:>9} {:>9} {:>7} {:>6} {:>14} {:>14} {:>26}",
                        "Method",
                        "Setup(s)",
                        "Med(s)",
                        "Min(s)",
                        "Iters",
                        "Reds",
                        "Rec/Reported",
                        "True(explicit)",
                        "Reason"
                    );
                }
                println!("{}", "-".repeat(if include_dof_col { 160 } else { 146 }));
            }

            let runs = [
                RunSpec::fgmres_jacobi_right(50),
                RunSpec::fgmres_none_right(50),
            ];

            for spec in runs {
                match run_once(&problem, &spec, &config) {
                    Ok(row) => {
                        if rank == 0 {
                            let include_dof_col = matches!(
                                problem.backend,
                                CsrBackend::Serial | CsrBackend::Distributed
                            );
                            let explicit_true = row
                                .explicit_true_residual
                                .map(|v| format!("{v:.2e}"))
                                .unwrap_or_else(|| "N/A".to_string());
                            if include_dof_col {
                                let dof = row
                                    .dof_per_sec
                                    .map(|v| format!("{v:.2e}"))
                                    .unwrap_or_else(|| "N/A".to_string());
                                println!(
                                    "{:<36} {:>9.3} {:>9.3} {:>9.3} {:>7} {:>6} {:>14.2e} {:>14} {:>26?} {:>12}",
                                    row.method,
                                    row.setup_secs,
                                    row.median_solve_secs,
                                    row.min_solve_secs,
                                    row.iterations,
                                    row.reductions,
                                    row.reported_residual,
                                    explicit_true,
                                    row.reason,
                                    dof
                                );
                            } else {
                                println!(
                                    "{:<36} {:>9.3} {:>9.3} {:>9.3} {:>7} {:>6} {:>14.2e} {:>14} {:>26?}",
                                    row.method,
                                    row.setup_secs,
                                    row.median_solve_secs,
                                    row.min_solve_secs,
                                    row.iterations,
                                    row.reductions,
                                    row.reported_residual,
                                    explicit_true,
                                    row.reason
                                );
                            }
                        }
                    }
                    Err(err) => {
                        if rank == 0 {
                            println!(
                                "{:<36} {:>9} {:>9} {:>9} {:>7} {:>6} {:>14} {:>14} {:>26}",
                                spec.name,
                                "FAIL",
                                "FAIL",
                                "FAIL",
                                "N/A",
                                "N/A",
                                "N/A",
                                "N/A",
                                "N/A"
                            );
                            println!("    → {err}");
                        }
                    }
                }
                problem.comm.barrier();
            }

            if rank == 0 {
                println!("{}", "=".repeat(96));
                println!();
            }
            problem.comm.barrier();
        }

        if rank == 0 {
            println!("Example complete.");
            println!(
                "Final backend summary: {}.",
                if is_parallel {
                    "MPI run (see per-case backend labels for replicated/distributed mode)"
                } else {
                    "serial CSR"
                }
            );
        }

        Ok(())
    }

    struct Problem {
        op: Arc<dyn KLinOp<Scalar = S>>,
        rhs: Vec<S>,
        csr_for_pc: Arc<SparseCsrMatrix<S>>,
        local_n: usize,
        global_n: usize,
        comm: UniverseComm,
        backend: CsrBackend,
        backend_descr: String,
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum CsrBackend {
        Serial,
        Replicated,
        Distributed,
    }

    impl CsrBackend {
        fn run_label(self) -> &'static str {
            match self {
                Self::Serial => "serial CSR",
                Self::Replicated => "replicated CSR",
                Self::Distributed => "distributed CSR",
            }
        }

        fn details(self) -> &'static str {
            match self {
                Self::Serial => "single rank operator",
                Self::Replicated => "all ranks hold full matrix, identical solve",
                Self::Distributed => "row-partitioned operator",
            }
        }

        fn benchmark_export_label(self) -> &'static str {
            match self {
                Self::Serial => "serial",
                Self::Replicated => "replicated",
                Self::Distributed => "distributed",
            }
        }
    }

    struct ResultRow {
        method: String,
        setup_secs: f64,
        median_solve_secs: f64,
        min_solve_secs: f64,
        iterations: usize,
        reductions: usize,
        reported_residual: R,
        explicit_true_residual: Option<R>,
        reason: ConvergedReason,
        dof_per_sec: Option<f64>,
    }

    struct RunSpec {
        name: &'static str,
        restart: usize,
        pc_side: PcSide,
        pc: PcKind,
    }

    enum PcKind {
        None,
        Jacobi,
    }

    #[derive(Clone, Copy, Debug)]
    struct BenchmarkConfig {
        warmup_runs: usize,
        measured_runs: usize,
    }

    impl Default for BenchmarkConfig {
        fn default() -> Self {
            Self {
                warmup_runs: 1,
                measured_runs: 5,
            }
        }
    }

    impl BenchmarkConfig {
        fn from_env_args() -> Result<Self, KError> {
            let mut cfg = Self::default();
            let mut args = env::args().skip(1);
            while let Some(arg) = args.next() {
                match arg.as_str() {
                    "--warmup-runs" => {
                        let Some(v) = args.next() else {
                            return Err(KError::InvalidInput(
                                "missing value for --warmup-runs".into(),
                            ));
                        };
                        cfg.warmup_runs = parse_positive_usize("--warmup-runs", &v)?;
                    }
                    "--measured-runs" => {
                        let Some(v) = args.next() else {
                            return Err(KError::InvalidInput(
                                "missing value for --measured-runs".into(),
                            ));
                        };
                        cfg.measured_runs = parse_positive_usize("--measured-runs", &v)?;
                    }
                    "--help" | "-h" => {
                        if cfg!(feature = "mpi") {
                            println!(
                                "Usage: cargo mpirun -n <ranks> --example complex_matrix_market_demo --features complex,mpi,mpi_examples -- [--warmup-runs N] [--measured-runs N]"
                            );
                        } else {
                            println!(
                                "Usage: cargo run --example complex_matrix_market_demo --features complex -- [--warmup-runs N] [--measured-runs N]"
                            );
                        }
                        std::process::exit(0);
                    }
                    _ => {}
                }
            }
            if cfg.measured_runs == 0 {
                return Err(KError::InvalidInput(
                    "--measured-runs must be at least 1".into(),
                ));
            }
            Ok(cfg)
        }
    }

    fn parse_positive_usize(flag: &str, value: &str) -> Result<usize, KError> {
        value.parse::<usize>().map_err(|_| {
            KError::InvalidInput(format!(
                "invalid value '{value}' for {flag}, expected non-negative integer"
            ))
        })
    }

    impl RunSpec {
        fn fgmres_jacobi_right(restart: usize) -> Self {
            Self {
                name: "FGMRES(restart) + Jacobi (R)",
                restart,
                pc_side: PcSide::Right,
                pc: PcKind::Jacobi,
            }
        }

        fn fgmres_none_right(restart: usize) -> Self {
            Self {
                name: "FGMRES(restart) + None (R)",
                restart,
                pc_side: PcSide::Right,
                pc: PcKind::None,
            }
        }
    }

    fn run_once(
        problem: &Problem,
        spec: &RunSpec,
        bench_cfg: &BenchmarkConfig,
    ) -> Result<ResultRow, KError> {
        let b = &problem.rhs;
        let mut jacobi_pc: Option<Jacobi> = None;
        let setup_start = Instant::now();
        if matches!(spec.pc, PcKind::Jacobi) {
            let mut pc = Jacobi::new();
            pc.setup(problem.csr_for_pc.as_ref())?;
            jacobi_pc = Some(pc);
        }
        let setup_secs = setup_start.elapsed().as_secs_f64();
        for _ in 0..bench_cfg.warmup_runs {
            let mut x = vec![S::zero(); problem.local_n];
            let mut solver = configured_solver(spec.restart);
            let mut workspace = Workspace::new(problem.local_n);
            solver.setup_workspace(&mut workspace);
            let _ = solver.solve_k(
                problem.op.as_ref(),
                jacobi_pc
                    .as_mut()
                    .map(|pc| pc as &mut dyn KPreconditioner<Scalar = S>),
                b,
                &mut x,
                spec.pc_side,
                &problem.comm,
                None,
                Some(&mut workspace),
            )?;
        }

        let mut solve_times = Vec::with_capacity(bench_cfg.measured_runs);
        let mut x_last = vec![S::zero(); problem.local_n];
        let mut final_stats = None;
        for _ in 0..bench_cfg.measured_runs {
            let mut x = vec![S::zero(); problem.local_n];
            problem.comm.barrier();
            let start = Instant::now();
            let mut solver = configured_solver(spec.restart);
            let mut workspace = Workspace::new(problem.local_n);
            solver.setup_workspace(&mut workspace);
            let stats = solver.solve_k(
                problem.op.as_ref(),
                jacobi_pc
                    .as_mut()
                    .map(|pc| pc as &mut dyn KPreconditioner<Scalar = S>),
                b,
                &mut x,
                spec.pc_side,
                &problem.comm,
                None,
                Some(&mut workspace),
            )?;
            problem.comm.barrier();
            let solve_secs = start.elapsed().as_secs_f64();
            solve_times.push(solve_secs);
            x_last = x;
            final_stats = Some(stats);
        }
        let stats = final_stats
            .ok_or_else(|| KError::InvalidInput("no measured solve run executed".into()))?;
        let min_solve_secs = solve_times.iter().copied().fold(f64::INFINITY, f64::min);
        let median_solve_secs = median(&mut solve_times);

        let reductions = stats.counters.num_global_reductions;
        let mut ax = vec![S::zero(); b.len()];
        let mut scratch = BridgeScratch::default();
        problem.op.matvec_s(&x_last, &mut ax, &mut scratch);
        for (ri, bi) in ax.iter_mut().zip(b.iter().copied()) {
            *ri = bi - *ri;
        }
        let explicit_true_residual = Some(
            problem
                .comm
                .reduction_engine(&ReductOptions::default())
                .norm2_s(&ax),
        );
        let dof_per_sec = if matches!(
            problem.backend,
            CsrBackend::Serial | CsrBackend::Distributed
        ) && median_solve_secs > 0.0
        {
            Some(problem.global_n as f64 / median_solve_secs)
        } else {
            None
        };

        Ok(ResultRow {
            method: format!("{} (m = {})", spec.name, spec.restart),
            setup_secs,
            median_solve_secs,
            min_solve_secs,
            iterations: stats.iterations,
            reductions,
            reported_residual: stats.final_residual,
            explicit_true_residual,
            reason: stats.reason,
            dof_per_sec,
        })
    }

    fn configured_solver(restart: usize) -> FgmresSolver {
        let mut solver = FgmresSolver::new(1e-8, 500, restart);
        solver.variant = FgmresVariant::Pipelined;
        solver.reorth = ReorthPolicy::IfNeeded;
        solver.atol = 1e-12;
        solver.dtol = 1e6;
        solver
    }

    fn median(samples: &mut [f64]) -> f64 {
        samples.sort_by(f64::total_cmp);
        let n = samples.len();
        if n % 2 == 1 {
            samples[n / 2]
        } else {
            (samples[n / 2 - 1] + samples[n / 2]) * 0.5
        }
    }

    fn load_problem_complex(mat_path: &Path, comm: &UniverseComm) -> Result<Problem, KError> {
        let mm = read_matrix_market(mat_path)?;
        let csr_sparse: SparseCsrMatrix<S> = mm.to_csr_matrix_scalar()?;
        let nrows = csr_sparse.nrows();
        let ncols = csr_sparse.ncols();
        let row_ptr = csr_sparse.row_ptr().to_vec();
        let col_idx = csr_sparse.col_idx().to_vec();
        let values = csr_sparse.values().to_vec();
        let csr_for_pc = Arc::new(csr_sparse);

        let csr_scalar = ScalarCsrMatrix::new(nrows, ncols, row_ptr, col_idx, values);
        let csr_arc = Arc::new(csr_scalar);

        let op = GenericCsrOp::new(csr_arc.clone(), &SpmvTuning::default()).with_comm(comm.clone());
        let op_arc: Arc<dyn KLinOp<Scalar = S>> = Arc::new(op);

        let rhs = match try_load_rhs_s(mat_path, csr_arc.nrows()) {
            Some(vec) => vec,
            None => {
                let ones = vec![S::one(); csr_arc.ncols()];
                let mut b = vec![S::zero(); csr_arc.nrows()];
                csr_arc.spmv(&ones, &mut b);
                b
            }
        };

        let local_n = nrows;
        let global_n = nrows;
        let backend = classify_backend(comm.size(), local_n, global_n);

        Ok(Problem {
            op: op_arc,
            rhs,
            csr_for_pc,
            local_n,
            global_n,
            comm: comm.clone(),
            backend,
            backend_descr: format!(
                "Generic CSR (complex, {})",
                backend.benchmark_export_label()
            ),
        })
    }

    fn classify_backend(size: usize, local_n: usize, global_n: usize) -> CsrBackend {
        if size <= 1 {
            CsrBackend::Serial
        } else if local_n == global_n {
            CsrBackend::Replicated
        } else {
            CsrBackend::Distributed
        }
    }

    fn try_load_rhs_s(mat_path: &Path, n: usize) -> Option<Vec<S>> {
        let mut rhs_path = mat_path.to_path_buf();
        if let Some(stem) = mat_path.file_stem().and_then(|s| s.to_str()) {
            rhs_path.set_file_name(format!("{stem}_rhs.mtx"));
            if rhs_path.exists() {
                if let Ok(mm) = read_matrix_market(&rhs_path) {
                    if let Ok(v) = mm.to_vector_scalar() {
                        if v.len() == n {
                            return Some(v);
                        }
                    }
                }
            }
        }
        None
    }
}

#[cfg(feature = "complex")]
fn main() -> Result<(), KError> {
    complex_demo::run()
}
