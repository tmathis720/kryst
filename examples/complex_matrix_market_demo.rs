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
    use std::fs::File;
    use std::io::Write;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::time::Instant;

    use super::KError;
    use kryst::algebra::bridge::BridgeScratch;
    use kryst::algebra::prelude::*;
    use kryst::context::ksp_context::{ReorthPolicy, Workspace};
    use kryst::matrix::DistCsrOp;
    use kryst::matrix::dist_csr::DistributedPlanDiagnostics;
    use kryst::matrix::sparse::CsrMatrix as SparseCsrMatrix;
    use kryst::ops::klinop::KLinOp;
    use kryst::ops::kpc::KPreconditioner;
    use kryst::parallel::{Comm, UniverseComm};
    use kryst::preconditioner::PcSide;
    use kryst::preconditioner::Preconditioner;
    use kryst::preconditioner::ilu_csr::{IluCsr, IluCsrConfig, IluKind};
    use kryst::preconditioner::jacobi::Jacobi;
    use kryst::solver::fgmres::{
        FgmresSolver, FgmresStagnationPolicy, FgmresVariant, OrthogMethod, ResidualCheckPolicy,
    };
    use kryst::solver::{LinearSolver, MonitorAction, MonitorCallback};
    use kryst::utils::convergence::ConvergedReason;
    use kryst::utils::matrix_market::read_matrix_market;

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
            println!("Run mode: {}", config.run_mode.label());
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
            println!(
                "Mode profile: {}",
                match config.run_mode {
                    RunMode::Correctness => {
                        "small matrices, explicit true residuals, optional replicated operator checks"
                    }
                    RunMode::Scalability => {
                        "distributed operator + local-block PC only, global norms only, lightweight reporting"
                    }
                    RunMode::Robustness => {
                        "robustness stress: fallback-enabled stagnation handling and restart-heavy behavior"
                    }
                }
            );
            println!(
                "Stagnation fallback: {} (min_inner_before_fallback={})",
                if config.run_mode == RunMode::Correctness || !config.allow_stagnation_fallback {
                    "disabled"
                } else {
                    "enabled"
                },
                config.min_inner_before_fallback
            );
            println!("FGMRES haptol: {:.3e}", config.fgmres_haptol);
            if config.run_mode == RunMode::Correctness {
                println!(
                "Replicated check marker: {}",
                    if config.mark_replicated_check {
                        "enabled (metadata-only)"
                    } else {
                        "disabled"
                    }
                );
            }
            if config.residual_history {
                println!(
                    "Residual history: enabled{}",
                    config
                        .residual_history_file
                        .as_ref()
                        .map(|p| format!(", output={}", p.display()))
                        .unwrap_or_default()
                );
            }
            println!("===============================================================");
            println!();
        }

        let base = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/mtx");
        let all_cases = [
            ("qc324.mtx", "qc324 (complex, general)"),
            ("dwg961a.mtx", "dwg961a (complex, general)"),
        ];
        let cases: Vec<(&str, &str)> = match config.run_mode {
            RunMode::Correctness => all_cases.to_vec(),
            RunMode::Scalability => all_cases.to_vec(),
            RunMode::Robustness => all_cases.to_vec(),
        };

        for (mat_name, descr) in cases {
            let mat_path = base.join(mat_name);
            let available = mat_path.exists();
            if !available {
                if rank == 0 {
                    println!("⚠ Missing file {mat_name} for {descr}, skipping.\n");
                }
                continue;
            }

            let problem = match load_problem_complex(
                &mat_path,
                &comm,
                config.run_mode,
                config.mark_replicated_check,
            ) {
                Ok(p) => p,
                Err(err) => {
                    if rank == 0 {
                        println!("❌ Failed to load {descr}: {err}\n");
                    }
                    continue;
                }
            };

            let rhs_norm2_local: f64 = problem.rhs.iter().map(|v| v.abs2()).sum();
            let rhs_norm = problem.comm.all_reduce_f64(rhs_norm2_local).sqrt();
            if rank == 0 {
                println!(
                    "=== [{} mode] {descr} — {} ===",
                    config.run_mode.label(),
                    problem.backend_descr
                );
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
                println!(
                    "Local row range (rank {rank}): [{}..{})",
                    problem.global_row_start,
                    problem.global_row_start + problem.local_n
                );
                if problem.comm.size() > 1 && problem.local_n == problem.global_n {
                    println!("Note: replicated execution: MPI ranks are not sharing SpMV rows.");
                }
                if config.run_mode == RunMode::Correctness {
                    println!(
                        "MPI scalability note: use distributed mode for scaling claims; correctness mode emphasizes solver validation."
                    );
                }
                println!("‖rhs‖₂ = {:.3e}", rhs_norm);
                println!("Residual semantics: rec/reported = solver recurrence/monitor residual.");
                if config.run_mode == RunMode::Correctness {
                    println!(
                        "                    true(explicit) = ||b - A x||₂ recomputed after solve."
                    );
                }
                println!(
                    "FGMRES side policy: requested left/symmetric are normalized to effective right preconditioning."
                );
                if config.run_mode == RunMode::Correctness && config.mark_replicated_check {
                    println!(
                        "Replicated check marker: ENABLED (metadata-only marker for cross-run comparison)."
                    );
                }
                let include_dof_col = matches!(
                    problem.backend,
                    CsrBackend::Serial | CsrBackend::Distributed
                );
                if config.run_mode == RunMode::Scalability {
                    println!(
                        "{:<7} {:<8} {:<6} {:<36} {:<34} {:>9} {:>9} {:>7} {:>5} {:>5} {:>5} {:>6} {:>14} {:>12}",
                        "Op",
                        "Exec",
                        "PCdom",
                        "Method",
                        "Effective policy",
                        "Med(s)",
                        "Min(s)",
                        "Iters",
                        "Rst",
                        "Inn",
                        "Pfb",
                        "Reds",
                        "Rec/Reported",
                        "DOF/s"
                    );
                    println!("{}", "-".repeat(184));
                } else if include_dof_col {
                    println!(
                        "{:<7} {:<8} {:<6} {:<36} {:<34} {:<34} {:>9} {:>9} {:>9} {:>7} {:>5} {:>5} {:>5} {:>6} {:>14} {:>14} {:>14} {:>14} {:>26} {:>12}",
                        "Op",
                        "Exec",
                        "PCdom",
                        "Method",
                        "Requested policy",
                        "Effective policy",
                        "Setup(s)",
                        "Med(s)",
                        "Min(s)",
                        "Iters",
                        "Rst",
                        "Inn",
                        "Pfb",
                        "Reds",
                        "Rec/Reported",
                        "True(explicit)",
                        "True(rel)",
                        "x_err(rel)",
                        "Reason",
                        "DOF/s"
                    );
                    println!("{}", "-".repeat(312));
                } else {
                    println!(
                        "{:<7} {:<8} {:<6} {:<36} {:<34} {:<34} {:>9} {:>9} {:>9} {:>7} {:>5} {:>5} {:>5} {:>6} {:>14} {:>14} {:>14} {:>14} {:>26}",
                        "Op",
                        "Exec",
                        "PCdom",
                        "Method",
                        "Requested policy",
                        "Effective policy",
                        "Setup(s)",
                        "Med(s)",
                        "Min(s)",
                        "Iters",
                        "Rst",
                        "Inn",
                        "Pfb",
                        "Reds",
                        "Rec/Reported",
                        "True(explicit)",
                        "True(rel)",
                        "x_err(rel)",
                        "Reason"
                    );
                    println!("{}", "-".repeat(296));
                }
                println!("Legend: Op=operator storage (csr-cx=complex CSR), Exec=execution backend (ser=serial, mpi-row=MPI row partition, mpi-repl=MPI replicated operator), PCdom=PC domain (full=global matrix ILU, own0=owned-block overlap=0 local ILU/ASM, n/a=no ILU domain).");
            }

            let runs = RunSpec::build_default_matrix(&config, &problem);

            for spec in runs {
                match run_once(&problem, &spec, &config) {
                    Ok(row) => {
                        if rank == 0 {
                            println!("{}", render_result_row(&row, config.run_mode, &problem));
                        }
                    }
                    Err(err) => {
                        if rank == 0 {
                            if config.run_mode == RunMode::Scalability {
                                println!(
                                    "{:<36} {:<34} {:>9} {:>9} {:>7} {:>6} {:>14} {:>12}",
                                    spec.method_label(),
                                    "N/A",
                                    "FAIL",
                                    "FAIL",
                                    "N/A",
                                    "N/A",
                                    "N/A",
                                    "N/A"
                                );
                            } else {
                                println!(
                                    "{:<36} {:<34} {:<34} {:>9} {:>9} {:>9} {:>7} {:>6} {:>14} {:>14} {:>14} {:>26}",
                                    spec.method_label(),
                                    spec.requested_policy_label(),
                                    "N/A",
                                    "FAIL",
                                    "FAIL",
                                    "FAIL",
                                    "N/A",
                                    "N/A",
                                    "N/A",
                                    "N/A",
                                    "N/A",
                                    "N/A"
                                );
                            }
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
        dist_plan_diagnostics: DistributedPlanDiagnostics,
        rhs: Vec<S>,
        rhs_source: RhsSource,
        csr_for_pc: Arc<SparseCsrMatrix<S>>,
        local_rows_nnz: usize,
        local_n: usize,
        global_n: usize,
        global_row_start: usize,
        comm: UniverseComm,
        backend: CsrBackend,
        backend_descr: String,
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum RhsSource {
        LoadedFromFile,
        GeneratedAOnes,
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
        operator_storage: &'static str,
        execution_backend: &'static str,
        pc_domain: &'static str,
        method: String,
        requested_policy: String,
        effective_policy: String,
        setup_secs: f64,
        median_solve_secs: f64,
        min_solve_secs: f64,
        iterations: usize,
        reductions: usize,
        restart_count: Option<usize>,
        inner_iterations_last_cycle: Option<usize>,
        pipeline_fallbacks: Option<usize>,
        reported_residual: R,
        explicit_true_residual: Option<R>,
        explicit_true_residual_rel: Option<R>,
        x_error_rel: Option<R>,
        reason: ConvergedReason,
        dof_per_sec: Option<f64>,
    }

    struct RunSpec {
        restart: usize,
        variant: FgmresVariant,
        residual_check_policy: ResidualCheckPolicy,
        orthog: OrthogMethod,
        reorth: ReorthPolicy,
        pc_side: PcSide,
        pc: PcKind,
    }

    #[derive(Clone, Debug)]
    struct CsrForPcDiagnostics {
        nnz_local_block: usize,
        nnz_local_rows: usize,
        nnz_ratio: f64,
        diag_min_abs: f64,
        diag_max_abs: f64,
        diag_tiny_or_missing_count: usize,
    }

    #[derive(Clone, Debug)]
    struct RankSpread {
        global_sum: f64,
        rank_min: f64,
        rank_max: f64,
        rank0_local: f64,
    }

    #[derive(Clone, Debug)]
    struct CsrForPcDiagnosticsGlobal {
        nnz_local_block: RankSpread,
        nnz_local_rows: RankSpread,
        nnz_ratio: RankSpread,
        diag_min_abs: RankSpread,
        diag_max_abs: RankSpread,
        diag_tiny_or_missing_count: RankSpread,
        pivot_perturbation_count: Option<RankSpread>,
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum PcKind {
        None,
        JacobiWeak,
        Ilu0Local,
        IlutLocal,
        MpiBlockJacobiIlu0Local,
        LocalIluk { k: usize },
        ReplicatedFullIlu0,
        ReplicatedFullIluk { k: usize },
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum PcDispatchBranch {
        None,
        JacobiWeak,
        Ilu0Local,
        IlutLocal,
    }

    #[derive(Clone, Debug)]
    struct BenchmarkConfig {
        run_mode: RunMode,
        warmup_runs: usize,
        measured_runs: usize,
        rtol: f64,
        atol: f64,
        maxits: usize,
        restarts: Vec<usize>,
        pcs: Vec<PcKind>,
        include_restart_200: bool,
        variants: Vec<FgmresVariant>,
        orthogs: Vec<OrthogMethod>,
        reorths: Vec<ReorthPolicy>,
        allow_stagnation_fallback: bool,
        min_inner_before_fallback: usize,
        mark_replicated_check: bool,
        residual_history: bool,
        residual_history_file: Option<PathBuf>,
        residual_history_force: bool,
        fgmres_haptol: f64,
        row_scale: bool,
        row_scale_tiny: f64,
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum RunMode {
        Correctness,
        Scalability,
        Robustness,
    }

    impl RunMode {
        fn label(self) -> &'static str {
            match self {
                Self::Correctness => "correctness",
                Self::Scalability => "scalability",
                Self::Robustness => "robustness",
            }
        }

        fn parse(token: &str) -> Result<Self, KError> {
            match token.trim().to_ascii_lowercase().as_str() {
                "correctness" => Ok(Self::Correctness),
                "scalability" => Ok(Self::Scalability),
                "robustness" => Ok(Self::Robustness),
                other => Err(KError::InvalidInput(format!(
                    "invalid run mode '{other}', expected correctness|scalability|robustness"
                ))),
            }
        }
    }

    impl Default for BenchmarkConfig {
        fn default() -> Self {
            Self {
                run_mode: RunMode::Correctness,
                warmup_runs: 1,
                measured_runs: 5,
                rtol: 1e-8,
                atol: 1e-12,
                maxits: 500,
                restarts: vec![50, 100, 150],
                pcs: Vec::new(),
                include_restart_200: false,
                variants: vec![FgmresVariant::Classical],
                orthogs: vec![OrthogMethod::ClassicalGS],
                reorths: vec![ReorthPolicy::IfNeeded],
                allow_stagnation_fallback: false,
                min_inner_before_fallback: 8,
                mark_replicated_check: false,
                residual_history: false,
                residual_history_file: None,
                residual_history_force: false,
                fgmres_haptol: 1e-30,
                row_scale: false,
                row_scale_tiny: 1e-15,
            }
        }
    }

    impl BenchmarkConfig {
        fn from_env_args() -> Result<Self, KError> {
            Self::from_args(env::args().skip(1))
        }

        fn from_args<I>(args: I) -> Result<Self, KError>
        where
            I: IntoIterator<Item = String>,
        {
            let mut cfg = Self::default();
            let mut args = args.into_iter().peekable();
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
                                "Usage: cargo mpirun -n <ranks> --example complex_matrix_market_demo --features complex,mpi,mpi_examples -- [--mode correctness|scalability|robustness] [--mark-replicated-check] [--warmup-runs N] [--measured-runs N] [--rtol F] [--atol F] [--maxits N] [--restarts csv] [--pcs csv] [--include-restart-200] [--allow-stagnation-fallback] [--min-inner-before-fallback N] [--fgmres-variant csv] [--fgmres-orthog csv] [--fgmres-reorth csv] [--fgmres-haptol F] [--residual-history] [--residual-history-file <path>] [--residual-history-force] [--row-scale [tiny]]"
                            );
                        } else {
                            println!(
                                "Usage: cargo run --example complex_matrix_market_demo --features complex -- [--mode correctness|scalability|robustness] [--mark-replicated-check] [--warmup-runs N] [--measured-runs N] [--rtol F] [--atol F] [--maxits N] [--restarts csv] [--pcs csv] [--include-restart-200] [--allow-stagnation-fallback] [--min-inner-before-fallback N] [--fgmres-variant csv] [--fgmres-orthog csv] [--fgmres-reorth csv] [--fgmres-haptol F] [--residual-history] [--residual-history-file <path>] [--residual-history-force] [--row-scale [tiny]]"
                            );
                        }
                        std::process::exit(0);
                    }
                    "--mode" => {
                        let Some(v) = args.next() else {
                            return Err(KError::InvalidInput("missing value for --mode".into()));
                        };
                        cfg.run_mode = RunMode::parse(&v)?;
                    }
                    "--mark-replicated-check" => {
                        cfg.mark_replicated_check = true;
                    }
                    "--correctness-replicated-check" => {
                        return Err(KError::InvalidInput(
                            "--correctness-replicated-check is metadata-only and has been renamed to --mark-replicated-check".into(),
                        ));
                    }
                    "--restarts" => {
                        let Some(v) = args.next() else {
                            return Err(KError::InvalidInput(
                                "missing value for --restarts".into(),
                            ));
                        };
                        cfg.restarts = parse_usize_csv("--restarts", &v)?;
                    }
                    "--rtol" => {
                        let Some(v) = args.next() else {
                            return Err(KError::InvalidInput("missing value for --rtol".into()));
                        };
                        cfg.rtol = parse_positive_f64("--rtol", &v)?;
                    }
                    "--atol" => {
                        let Some(v) = args.next() else {
                            return Err(KError::InvalidInput("missing value for --atol".into()));
                        };
                        cfg.atol = parse_positive_f64("--atol", &v)?;
                    }
                    "--maxits" => {
                        let Some(v) = args.next() else {
                            return Err(KError::InvalidInput("missing value for --maxits".into()));
                        };
                        cfg.maxits = parse_positive_usize("--maxits", &v)?;
                    }
                    "--pcs" => {
                        let Some(v) = args.next() else {
                            return Err(KError::InvalidInput("missing value for --pcs".into()));
                        };
                        cfg.pcs = parse_pc_csv("--pcs", &v)?;
                    }
                    "--include-restart-200" => {
                        cfg.include_restart_200 = true;
                    }
                    "--allow-stagnation-fallback" => {
                        cfg.allow_stagnation_fallback = true;
                    }
                    "--min-inner-before-fallback" => {
                        let Some(v) = args.next() else {
                            return Err(KError::InvalidInput(
                                "missing value for --min-inner-before-fallback".into(),
                            ));
                        };
                        cfg.min_inner_before_fallback =
                            parse_positive_usize("--min-inner-before-fallback", &v)?;
                    }
                    "--fgmres-variant" => {
                        let Some(v) = args.next() else {
                            return Err(KError::InvalidInput(
                                "missing value for --fgmres-variant".into(),
                            ));
                        };
                        cfg.variants = parse_variant_csv("--fgmres-variant", &v)?;
                    }
                    "--fgmres-orthog" => {
                        let Some(v) = args.next() else {
                            return Err(KError::InvalidInput(
                                "missing value for --fgmres-orthog".into(),
                            ));
                        };
                        cfg.orthogs = parse_orthog_csv("--fgmres-orthog", &v)?;
                    }
                    "--residual-history" => {
                        cfg.residual_history = true;
                    }
                    "--residual-history-file" => {
                        let Some(v) = args.next() else {
                            return Err(KError::InvalidInput(
                                "missing value for --residual-history-file".into(),
                            ));
                        };
                        cfg.residual_history_file = Some(PathBuf::from(v));
                    }
                    "--residual-history-force" => {
                        cfg.residual_history_force = true;
                    }
                    "--row-scale" => {
                        cfg.row_scale = true;
                        if let Some(next) = args.peek() {
                            if !next.starts_with("--") {
                                let tiny = args.next().ok_or_else(|| {
                                    KError::InvalidInput(
                                        "failed to read --row-scale optional tiny".into(),
                                    )
                                })?;
                                cfg.row_scale_tiny = parse_positive_f64("--row-scale", &tiny)?;
                            }
                        }
                    }
                    "--fgmres-reorth" => {
                        let Some(v) = args.next() else {
                            return Err(KError::InvalidInput(
                                "missing value for --fgmres-reorth".into(),
                            ));
                        };
                        cfg.reorths = parse_reorth_csv("--fgmres-reorth", &v)?;
                    }
                    "--fgmres-haptol" => {
                        let Some(v) = args.next() else {
                            return Err(KError::InvalidInput(
                                "missing value for --fgmres-haptol".into(),
                            ));
                        };
                        cfg.fgmres_haptol = parse_positive_finite_f64("--fgmres-haptol", &v)?;
                    }
                    _ => {
                        return Err(KError::InvalidInput(format!("unknown argument: {arg}")));
                    }
                }
            }
            if cfg.measured_runs == 0 {
                return Err(KError::InvalidInput(
                    "--measured-runs must be at least 1".into(),
                ));
            }
            if cfg.include_restart_200 && !cfg.restarts.contains(&200) {
                cfg.restarts.push(200);
            }
            if cfg.run_mode == RunMode::Scalability {
                cfg.mark_replicated_check = false;
            }
            if cfg.pcs.iter().any(|pc| pc.is_ilut()) {
                eprintln!(
                    "⚠ --pcs includes ILUT for a complex run: current ILUT path is a degraded real projection and is not trusted for complex robustness benchmarking."
                );
            }
            if cfg.residual_history
                && cfg.run_mode != RunMode::Correctness
                && !cfg.residual_history_force
            {
                cfg.residual_history = false;
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

    fn parse_positive_f64(flag: &str, value: &str) -> Result<f64, KError> {
        let val = value.parse::<f64>().map_err(|_| {
            KError::InvalidInput(format!(
                "invalid value '{value}' for {flag}, expected non-negative float"
            ))
        })?;
        if val < 0.0 {
            return Err(KError::InvalidInput(format!(
                "invalid value '{value}' for {flag}, expected non-negative float"
            )));
        }
        Ok(val)
    }

    fn parse_positive_finite_f64(flag: &str, value: &str) -> Result<f64, KError> {
        let val = value.parse::<f64>().map_err(|_| {
            KError::InvalidInput(format!(
                "invalid value '{value}' for {flag}, expected positive finite float"
            ))
        })?;
        if !val.is_finite() || val <= 0.0 {
            return Err(KError::InvalidInput(format!(
                "invalid value '{value}' for {flag}, expected positive finite float"
            )));
        }
        Ok(val)
    }

    fn parse_usize_csv(flag: &str, value: &str) -> Result<Vec<usize>, KError> {
        let vals = value
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|v| parse_positive_usize(flag, v))
            .collect::<Result<Vec<_>, _>>()?;
        if vals.is_empty() {
            return Err(KError::InvalidInput(format!(
                "{flag} expects at least one integer value"
            )));
        }
        Ok(vals)
    }

    fn parse_variant(token: &str) -> Result<FgmresVariant, KError> {
        match token.trim().to_ascii_lowercase().as_str() {
            "classical" => Ok(FgmresVariant::Classical),
            "pipelined" => Ok(FgmresVariant::Pipelined),
            other => Err(KError::InvalidInput(format!(
                "invalid fgmres variant '{other}', expected classical|pipelined"
            ))),
        }
    }

    fn parse_orthog(token: &str) -> Result<OrthogMethod, KError> {
        match token.trim().to_ascii_lowercase().as_str() {
            "classical" | "cgs" | "cgs_refined" | "cgs-refined" | "refined" => {
                Ok(OrthogMethod::ClassicalGS)
            }
            "mgs" | "modified" => Ok(OrthogMethod::ModifiedGS),
            other => Err(KError::InvalidInput(format!(
                "invalid orthog '{other}', expected classical|cgs|cgs_refined|mgs|modified"
            ))),
        }
    }

    fn parse_reorth(token: &str) -> Result<ReorthPolicy, KError> {
        match token.trim().to_ascii_lowercase().as_str() {
            "never" => Ok(ReorthPolicy::Never),
            "ifneeded" | "if-needed" => Ok(ReorthPolicy::IfNeeded),
            "always" => Ok(ReorthPolicy::Always),
            other => Err(KError::InvalidInput(format!(
                "invalid reorth '{other}', expected never|ifneeded|always"
            ))),
        }
    }

    fn parse_variant_csv(flag: &str, value: &str) -> Result<Vec<FgmresVariant>, KError> {
        parse_csv(flag, value, parse_variant)
    }

    fn parse_orthog_csv(flag: &str, value: &str) -> Result<Vec<OrthogMethod>, KError> {
        parse_csv(flag, value, parse_orthog)
    }

    fn parse_reorth_csv(flag: &str, value: &str) -> Result<Vec<ReorthPolicy>, KError> {
        parse_csv(flag, value, parse_reorth)
    }

    fn parse_pc(token: &str) -> Result<PcKind, KError> {
        let token_norm = token.trim().to_ascii_lowercase();
        if let Some(k_str) = token_norm.strip_prefix("local-iluk:") {
            return Ok(PcKind::LocalIluk {
                k: parse_positive_usize("local-iluk:k", k_str)?,
            });
        }
        if let Some(k_str) = token_norm.strip_prefix("replicated-iluk:") {
            return Ok(PcKind::ReplicatedFullIluk {
                k: parse_positive_usize("replicated-iluk:k", k_str)?,
            });
        }
        match token_norm.as_str() {
            "none" | "off" => Ok(PcKind::None),
            "jacobi" | "jacobi-weak" | "weak-jacobi" => Ok(PcKind::JacobiWeak),
            "ilu0" | "ilu0-local" | "local-ilu0" => Ok(PcKind::Ilu0Local),
            "ilut" | "ilut-local" | "local-ilut" => Ok(PcKind::IlutLocal),
            "replicated-ilu0" => Ok(PcKind::ReplicatedFullIlu0),
            "mpi-block-jacobi-ilu0" | "block-jacobi-ilu0" | "mpi-block-ilu0" => {
                Ok(PcKind::MpiBlockJacobiIlu0Local)
            }
            other => Err(KError::InvalidInput(format!(
                "invalid pc '{other}', expected none|jacobi|local-ilu0|local-ilut|local-iluk:<k>|replicated-ilu0|replicated-iluk:<k>|mpi-block-jacobi-ilu0"
            ))),
        }
    }

    fn parse_pc_csv(flag: &str, value: &str) -> Result<Vec<PcKind>, KError> {
        parse_csv(flag, value, parse_pc)
    }

    fn parse_csv<T, F>(flag: &str, value: &str, mut parser: F) -> Result<Vec<T>, KError>
    where
        F: FnMut(&str) -> Result<T, KError>,
    {
        let vals = value
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(&mut parser)
            .collect::<Result<Vec<_>, _>>()?;
        if vals.is_empty() {
            return Err(KError::InvalidInput(format!(
                "{flag} expects at least one value"
            )));
        }
        Ok(vals)
    }

    impl RunSpec {
        fn build_default_matrix(cfg: &BenchmarkConfig, problem: &Problem) -> Vec<Self> {
            let pcs = if cfg.pcs.is_empty() {
                match cfg.run_mode {
                    RunMode::Correctness => {
                        if problem.comm.size() > 1 {
                            vec![
                                PcKind::MpiBlockJacobiIlu0Local,
                                PcKind::JacobiWeak,
                                PcKind::None,
                            ]
                        } else {
                            vec![PcKind::Ilu0Local, PcKind::JacobiWeak, PcKind::None]
                        }
                    }
                    RunMode::Scalability => {
                        if problem.comm.size() > 1 {
                            vec![PcKind::MpiBlockJacobiIlu0Local]
                        } else {
                            vec![PcKind::Ilu0Local]
                        }
                    }
                    RunMode::Robustness => {
                        if problem.comm.size() > 1 {
                            vec![PcKind::MpiBlockJacobiIlu0Local, PcKind::JacobiWeak]
                        } else {
                            vec![PcKind::Ilu0Local, PcKind::JacobiWeak]
                        }
                    }
                }
            } else {
                cfg.pcs.clone()
            };
            let mut runs = Vec::new();
            for &restart in &cfg.restarts {
                for &variant in &cfg.variants {
                    for &orthog in &cfg.orthogs {
                        for &reorth in &cfg.reorths {
                            for &pc in &pcs {
                                if pc == PcKind::MpiBlockJacobiIlu0Local && problem.comm.size() <= 1
                                {
                                    continue;
                                }
                                runs.push(Self {
                                    restart,
                                    variant,
                                    residual_check_policy: match cfg.run_mode {
                                        RunMode::Correctness => ResidualCheckPolicy::OnConvergence,
                                        RunMode::Scalability => ResidualCheckPolicy::RestartOnly,
                                        RunMode::Robustness => ResidualCheckPolicy::RestartOnly,
                                    },
                                    orthog,
                                    reorth,
                                    pc_side: PcSide::Right,
                                    pc,
                                });
                            }
                        }
                    }
                }
            }
            runs
        }

        fn method_label(&self) -> String {
            let effective_side = normalized_fgmres_side(self.pc_side);
            let side_desc = if effective_side == self.pc_side {
                format!("{}", pc_side_label(self.pc_side))
            } else {
                format!(
                    "{}→{} (normalized)",
                    pc_side_label(self.pc_side),
                    pc_side_label(effective_side)
                )
            };
            format!(
                "FGMRES+{} [m={}, v={}, reschk={}, orth={}, reorth={}, pc=requested {}, effective {}]",
                self.pc.label(),
                self.restart,
                variant_label(self.variant),
                residual_check_policy_label(self.residual_check_policy),
                orthog_label(self.orthog),
                reorth_label(self.reorth),
                pc_side_label(self.pc_side),
                side_desc,
            )
        }

        fn requested_policy_label(&self) -> String {
            format!(
                "variant={}, restart={}, residual-check={}",
                variant_label(self.variant),
                self.restart,
                residual_check_policy_label(self.residual_check_policy)
            )
        }
    }

    fn operator_storage_label(_problem: &Problem) -> &'static str {
        "csr-cx"
    }

    fn execution_backend_label(problem: &Problem) -> &'static str {
        match problem.backend {
            CsrBackend::Serial => "ser",
            CsrBackend::Replicated => "mpi-repl",
            CsrBackend::Distributed => "mpi-row",
        }
    }

    fn pc_domain_label(pc: PcKind) -> &'static str {
        match pc {
            PcKind::ReplicatedFullIlu0 | PcKind::ReplicatedFullIluk { .. } => "full",
            PcKind::Ilu0Local | PcKind::IlutLocal | PcKind::MpiBlockJacobiIlu0Local | PcKind::LocalIluk { .. } => "own0",
            PcKind::None | PcKind::JacobiWeak => "n/a",
        }
    }

    impl PcKind {
        fn label(&self) -> &'static str {
            match self {
                Self::None => "none (unpreconditioned reference)",
                Self::JacobiWeak => "jacobi (weak baseline)",
                Self::Ilu0Local => "block-jacobi-ilu0-overlap0",
                Self::IlutLocal => {
                    "local ILUT [degraded/provisional complex path: real-projection fallback; not trusted for complex robustness benchmarking]"
                }
                Self::MpiBlockJacobiIlu0Local => "block-jacobi-ilu0-overlap0 [mpi spelling alias]",
                Self::LocalIluk { .. } => {
                    "local ILU(k) (block-Jacobi ILU(k), zero overlap/local owned block)"
                }
                Self::ReplicatedFullIlu0 => {
                    "replicated full ILU(0) [correctness only, not scalable]"
                }
                Self::ReplicatedFullIluk { .. } => {
                    "replicated full ILU(k) [correctness only, not scalable]"
                }
            }
        }

        fn dispatch_branch(self) -> PcDispatchBranch {
            match self {
                Self::None => PcDispatchBranch::None,
                Self::JacobiWeak => PcDispatchBranch::JacobiWeak,
                Self::Ilu0Local
                | Self::MpiBlockJacobiIlu0Local
                | Self::LocalIluk { .. }
                | Self::ReplicatedFullIlu0
                | Self::ReplicatedFullIluk { .. } => PcDispatchBranch::Ilu0Local,
                Self::IlutLocal => PcDispatchBranch::IlutLocal,
            }
        }

        fn explicit_alias_of(self) -> Option<Self> {
            match self {
                Self::MpiBlockJacobiIlu0Local => Some(Self::Ilu0Local),
                Self::None
                | Self::JacobiWeak
                | Self::Ilu0Local
                | Self::IlutLocal
                | Self::LocalIluk { .. }
                | Self::ReplicatedFullIlu0
                | Self::ReplicatedFullIluk { .. } => None,
            }
        }

        fn semantic_experiment_key(self, mpi_mode: bool) -> String {
            match self {
                Self::MpiBlockJacobiIlu0Local | Self::Ilu0Local if mpi_mode => {
                    "block-jacobi-ilu0-overlap0".to_string()
                }
                Self::MpiBlockJacobiIlu0Local => "mpi-block-jacobi-ilu0-local".to_string(),
                Self::Ilu0Local => "local-ilu0".to_string(),
                Self::None => "none".to_string(),
                Self::JacobiWeak => "jacobi-weak".to_string(),
                Self::IlutLocal => "local-ilut".to_string(),
                Self::LocalIluk { k } => format!("local-iluk:{k}"),
                Self::ReplicatedFullIlu0 => "replicated-ilu0".to_string(),
                Self::ReplicatedFullIluk { k } => format!("replicated-iluk:{k}"),
            }
        }
        fn is_ilut(self) -> bool {
            matches!(self, Self::IlutLocal)
        }
    }

    fn variant_label(variant: FgmresVariant) -> &'static str {
        match variant {
            FgmresVariant::Classical => "classical",
            FgmresVariant::Pipelined => "pipelined",
        }
    }

    fn orthog_label(orthog: OrthogMethod) -> &'static str {
        match orthog {
            OrthogMethod::ClassicalGS => "classical-gs",
            OrthogMethod::ModifiedGS => "modified-gs",
        }
    }

    fn reorth_label(reorth: ReorthPolicy) -> &'static str {
        match reorth {
            ReorthPolicy::Never => "never",
            ReorthPolicy::IfNeeded => "if-needed",
            ReorthPolicy::Always => "always",
        }
    }

    fn residual_check_policy_label(policy: ResidualCheckPolicy) -> &'static str {
        match policy {
            ResidualCheckPolicy::RestartOnly => "restart-only",
            ResidualCheckPolicy::OnConvergence => "on-convergence",
            ResidualCheckPolicy::EveryIteration => "every-iteration",
            ResidualCheckPolicy::Debug => "debug",
        }
    }

    fn pc_side_label(pc_side: PcSide) -> &'static str {
        match pc_side {
            PcSide::Right => "right",
            PcSide::Left => "left",
            PcSide::Symmetric => "symmetric",
        }
    }

    fn normalized_fgmres_side(requested_side: PcSide) -> PcSide {
        match requested_side {
            PcSide::Right => PcSide::Right,
            PcSide::Left | PcSide::Symmetric => PcSide::Right,
        }
    }

    #[derive(Clone, Debug)]
    struct ResidualHistoryEntry {
        iter: usize,
        recurrence_residual: R,
        true_residual: Option<R>,
        checkpoint: bool,
    }

    #[derive(Default)]
    struct RunResidualHistory {
        entries: Vec<ResidualHistoryEntry>,
    }

    struct RowScaledOp {
        base: Arc<dyn KLinOp<Scalar = S>>,
        d: Vec<R>,
    }

    impl KLinOp for RowScaledOp {
        type Scalar = S;

        fn dims(&self) -> (usize, usize) {
            self.base.dims()
        }

        fn matvec_s(&self, x: &[S], y: &mut [S], scratch: &mut BridgeScratch) {
            self.base.matvec_s(x, y, scratch);
            for (yi, di) in y.iter_mut().zip(self.d.iter().copied()) {
                *yi *= S::from_real(di);
            }
        }

        fn supports_t_matvec_s(&self) -> bool {
            self.base.supports_t_matvec_s()
        }

        fn t_matvec_s(&self, x: &[S], y: &mut [S], scratch: &mut BridgeScratch) {
            self.base.t_matvec_s(x, y, scratch);
        }
    }

    fn compute_row_scaling(matrix: &SparseCsrMatrix<S>, tiny: f64) -> Vec<R> {
        let row_ptr = matrix.row_ptr();
        let vals = matrix.values();
        (0..matrix.nrows())
            .map(|r| {
                let mut row_inf = 0.0f64;
                for nz in row_ptr[r]..row_ptr[r + 1] {
                    row_inf = row_inf.max(vals[nz].abs());
                }
                if row_inf > tiny { 1.0 / row_inf } else { 1.0 }
            })
            .collect()
    }

    fn run_once(
        problem: &Problem,
        spec: &RunSpec,
        bench_cfg: &BenchmarkConfig,
    ) -> Result<ResultRow, KError> {
        let b_unscaled = &problem.rhs;
        let (row_scaling, op_scaled): (Option<Vec<R>>, Arc<dyn KLinOp<Scalar = S>>) =
            if bench_cfg.row_scale {
                let d = compute_row_scaling(problem.csr_for_pc.as_ref(), bench_cfg.row_scale_tiny);
                (
                    Some(d.clone()),
                    Arc::new(RowScaledOp {
                        base: problem.op.clone(),
                        d,
                    }),
                )
            } else {
                (None, problem.op.clone())
            };
        let b_scaled: Vec<S> = if let Some(d) = &row_scaling {
            b_unscaled
                .iter()
                .zip(d.iter())
                .map(|(bi, di)| *bi * S::from_real(*di))
                .collect()
        } else {
            b_unscaled.clone()
        };
        let b = &b_scaled;
        let effective_pc_side = normalized_fgmres_side(spec.pc_side);
        let csr_pc_diag =
            csr_for_pc_diagnostics(problem.csr_for_pc.as_ref(), problem.local_rows_nnz);
        enum PcHandle {
            Jacobi(Jacobi),
            Ilu0(IluCsr),
            MpiBlockJacobiIlu0(IluCsr),
            ReplicatedFull {
                ilu: IluCsr,
                global_n: usize,
                global_row_start: usize,
                local_n: usize,
                scratch_in: std::sync::Mutex<Vec<S>>,
                scratch_out: std::sync::Mutex<Vec<S>>,
            },
        }

        impl PcHandle {
            fn as_kpc_mut(&mut self) -> &mut dyn KPreconditioner<Scalar = S> {
                match self {
                    Self::Jacobi(pc) => pc,
                    Self::Ilu0(pc) => pc,
                    Self::MpiBlockJacobiIlu0(pc) => pc,
                    Self::ReplicatedFull { .. } => self,
                }
            }
        }
        impl KPreconditioner for PcHandle {
            type Scalar = S;
            fn dims(&self) -> (usize, usize) {
                match self {
                    Self::Jacobi(pc) => KPreconditioner::dims(pc),
                    Self::Ilu0(pc) => KPreconditioner::dims(pc),
                    Self::MpiBlockJacobiIlu0(pc) => KPreconditioner::dims(pc),
                    Self::ReplicatedFull { global_n, .. } => (*global_n, *global_n),
                }
            }
            fn apply_s(
                &self,
                side: PcSide,
                x: &[S],
                y: &mut [S],
                scratch: &mut BridgeScratch,
            ) -> Result<(), KError> {
                match self {
                    Self::Jacobi(pc) => pc.apply_s(side, x, y, scratch),
                    Self::Ilu0(pc) => pc.apply_s(side, x, y, scratch),
                    Self::MpiBlockJacobiIlu0(pc) => pc.apply_s(side, x, y, scratch),
                    Self::ReplicatedFull {
                        ilu,
                        global_n,
                        global_row_start,
                        local_n,
                        scratch_in,
                        scratch_out,
                    } => {
                        let mut in_full = scratch_in.lock().map_err(|_| {
                            KError::InvalidInput(
                                "failed to lock replicated ILU input scratch".into(),
                            )
                        })?;
                        let mut out_full = scratch_out.lock().map_err(|_| {
                            KError::InvalidInput(
                                "failed to lock replicated ILU output scratch".into(),
                            )
                        })?;
                        in_full.fill(S::zero());
                        in_full[*global_row_start..(*global_row_start + *local_n)]
                            .copy_from_slice(x);
                        let _ = global_n;
                        ilu.apply_s(side, &in_full, &mut out_full, scratch)?;
                        y.copy_from_slice(
                            &out_full[*global_row_start..(*global_row_start + *local_n)],
                        );
                        Ok(())
                    }
                }
            }
        }

        let mut pc: Option<PcHandle> = None;
        let setup_start = Instant::now();
        match spec.pc.dispatch_branch() {
            PcDispatchBranch::None => {}
            PcDispatchBranch::JacobiWeak => {
                let mut jacobi = Jacobi::new();
                jacobi.setup(problem.csr_for_pc.as_ref())?;
                pc = Some(PcHandle::Jacobi(jacobi));
            }
            PcDispatchBranch::Ilu0Local => {
                validate_local_ilu_owned_block(problem.csr_for_pc.as_ref())?;
                let mut cfg = IluCsrConfig::default();
                cfg.kind = match spec.pc {
                    PcKind::Ilu0Local
                    | PcKind::MpiBlockJacobiIlu0Local
                    | PcKind::ReplicatedFullIlu0 => IluKind::Ilu0,
                    PcKind::IlutLocal => IluKind::Ilut {
                        params: Default::default(),
                    },
                    PcKind::LocalIluk { k } | PcKind::ReplicatedFullIluk { k } => {
                        IluKind::Iluk { k }
                    }
                    PcKind::None | PcKind::JacobiWeak => IluKind::Ilu0,
                };
                let mut ilu = IluCsr::new_with_config(cfg);
                ilu.setup(problem.csr_for_pc.as_ref())?;
                if bench_cfg.run_mode == RunMode::Correctness {
                    let pivot_perturbation_count = None::<usize>;
                    let diag_global = reduce_csr_for_pc_diagnostics(
                        &problem.comm,
                        &csr_pc_diag,
                        pivot_perturbation_count,
                    );
                    if problem.comm.rank() == 0 {
                        print_csr_for_pc_diagnostics(&spec.method_label(), &diag_global);
                    }
                }
                pc = Some(match spec.pc {
                    PcKind::Ilu0Local | PcKind::IlutLocal | PcKind::LocalIluk { .. } => {
                        PcHandle::Ilu0(ilu)
                    }
                    PcKind::MpiBlockJacobiIlu0Local => PcHandle::MpiBlockJacobiIlu0(ilu),
                    PcKind::ReplicatedFullIlu0 | PcKind::ReplicatedFullIluk { .. } => {
                        PcHandle::ReplicatedFull {
                            ilu,
                            global_n: problem.global_n,
                            global_row_start: problem.global_row_start,
                            local_n: problem.local_n,
                            scratch_in: std::sync::Mutex::new(vec![S::zero(); problem.global_n]),
                            scratch_out: std::sync::Mutex::new(vec![S::zero(); problem.global_n]),
                        }
                    }
                    PcKind::None | PcKind::JacobiWeak => unreachable!(),
                });
            }
            PcDispatchBranch::IlutLocal => {
                validate_local_ilu_owned_block(problem.csr_for_pc.as_ref())?;
                let mut cfg = IluCsrConfig::default();
                cfg.kind = IluKind::Ilut {
                    params: Default::default(),
                };
                let mut ilu = IluCsr::new_with_config(cfg);
                ilu.setup(problem.csr_for_pc.as_ref())?;
                if bench_cfg.run_mode == RunMode::Correctness {
                    let pivot_perturbation_count = None::<usize>;
                    let diag_global = reduce_csr_for_pc_diagnostics(
                        &problem.comm,
                        &csr_pc_diag,
                        pivot_perturbation_count,
                    );
                    if problem.comm.rank() == 0 {
                        print_csr_for_pc_diagnostics(&spec.method_label(), &diag_global);
                    }
                }
                pc = Some(PcHandle::Ilu0(ilu));
            }
        }
        let setup_secs = setup_start.elapsed().as_secs_f64();
        for _ in 0..bench_cfg.warmup_runs {
            let mut x = vec![S::zero(); problem.local_n];
            let mut solver = configured_solver(spec, bench_cfg);
            apply_dist_plan_policy(&mut solver, problem);
            let mut workspace = Workspace::new(problem.local_n);
            solver.setup_workspace(&mut workspace);
            let _ = solver.solve_k(
                op_scaled.as_ref(),
                pc.as_mut().map(PcHandle::as_kpc_mut),
                b,
                &mut x,
                effective_pc_side,
                &problem.comm,
                None,
                Some(&mut workspace),
            )?;
        }

        let mut solve_times = Vec::with_capacity(bench_cfg.measured_runs);
        let mut x_last = vec![S::zero(); problem.local_n];
        let mut final_stats = None;
        let mut residual_history_last = RunResidualHistory::default();
        for _ in 0..bench_cfg.measured_runs {
            let mut x = vec![S::zero(); problem.local_n];
            problem.comm.barrier();
            let start = Instant::now();
            let mut solver = configured_solver(spec, bench_cfg);
            apply_dist_plan_policy(&mut solver, problem);
            let mut workspace = Workspace::new(problem.local_n);
            solver.setup_workspace(&mut workspace);
            let mut run_history = RunResidualHistory::default();
            let mut monitors: Vec<Box<MonitorCallback<R>>> = Vec::new();
            if bench_cfg.residual_history {
                monitors.push(Box::new(|it, res, _| {
                    let _ = (it, res);
                    MonitorAction::Continue
                }));
            }
            let history_ref = std::sync::Arc::new(std::sync::Mutex::new(Vec::<(usize, R)>::new()));
            if bench_cfg.residual_history {
                let history_ref_c = history_ref.clone();
                monitors.clear();
                monitors.push(Box::new(move |it, res, _| {
                    if let Ok(mut h) = history_ref_c.lock() {
                        h.push((it, res));
                    }
                    MonitorAction::Continue
                }));
            }
            let stats = solver.solve_k(
                op_scaled.as_ref(),
                pc.as_mut().map(PcHandle::as_kpc_mut),
                b,
                &mut x,
                effective_pc_side,
                &problem.comm,
                if monitors.is_empty() {
                    None
                } else {
                    Some(&monitors)
                },
                Some(&mut workspace),
            )?;
            if bench_cfg.residual_history {
                if let Ok(h) = history_ref.lock() {
                    run_history.entries = h
                        .iter()
                        .map(|(it, res)| ResidualHistoryEntry {
                            iter: *it,
                            recurrence_residual: *res,
                            true_residual: None,
                            checkpoint: false,
                        })
                        .collect();
                }
                residual_history_last = run_history;
            }
            problem.comm.barrier();
            let solve_secs = start.elapsed().as_secs_f64();
            solve_times.push(solve_secs);
            x_last = x;
            final_stats = Some(stats);
        }
        let stats = final_stats
            .ok_or_else(|| KError::InvalidInput("no measured solve run executed".into()))?;
        // Row scaling changes only equations (D_r A x = D_r b), so x is unchanged.
        // Keep an explicit "map-back" step so optional future column scaling can hook here.
        let x_unscaled = x_last;
        let min_solve_secs = solve_times.iter().copied().fold(f64::INFINITY, f64::min);
        let median_solve_secs = median(&mut solve_times);

        let reductions = stats.counters.num_global_reductions;
        let (explicit_true_residual, explicit_true_residual_rel) =
            if bench_cfg.run_mode == RunMode::Correctness {
                let mut ax = vec![S::zero(); b.len()];
                let mut scratch = BridgeScratch::default();
                problem.op.matvec_s(&x_unscaled, &mut ax, &mut scratch);
                for (ri, bi) in ax.iter_mut().zip(b_unscaled.iter().copied()) {
                    *ri = bi - *ri;
                }
                let r2_local = ax.iter().map(|v| v.abs2()).sum::<f64>();
                let true_res = problem.comm.all_reduce_f64(r2_local).sqrt();
                let rhs_norm2_local = b.iter().map(|v| v.abs2()).sum::<f64>();
                let rhs_norm = problem.comm.all_reduce_f64(rhs_norm2_local).sqrt();
                let rel_true = true_res / rhs_norm.max(f64::MIN_POSITIVE);
                (Some(true_res), Some(rel_true))
            } else {
                (None, None)
            };
        if bench_cfg.residual_history && bench_cfg.measured_runs > 0 && problem.comm.rank() == 0 {
            let mut checkpoint_count = 0usize;
            // mark restart boundaries based on effective restart interval and policy
            let restart = stats.effective_restart.unwrap_or(spec.restart).max(1);
            for e in &mut residual_history_last.entries {
                if e.iter > 0 && e.iter % restart == 0 {
                    e.checkpoint = true;
                    if matches!(
                        spec.residual_check_policy,
                        ResidualCheckPolicy::RestartOnly
                            | ResidualCheckPolicy::OnConvergence
                            | ResidualCheckPolicy::EveryIteration
                            | ResidualCheckPolicy::Debug
                    ) {
                        e.true_residual = explicit_true_residual;
                    }
                    checkpoint_count += 1;
                }
            }
            println!(
                "[history][rank0] {}: {} points, {} restart checkpoints",
                format!(
                    "{} [row-scale={}]",
                    spec.method_label(),
                    if bench_cfg.row_scale { "on" } else { "off" }
                ),
                residual_history_last.entries.len(),
                checkpoint_count
            );
            if let Some(path) = &bench_cfg.residual_history_file {
                dump_residual_history(path, &residual_history_last)?;
                println!("[history][rank0] wrote {}", path.display());
            }
        }

        let x_error_rel = if bench_cfg.run_mode == RunMode::Correctness
            && problem.rhs_source == RhsSource::GeneratedAOnes
        {
            let err2_local = x_unscaled
                .iter()
                .map(|xi| (*xi - S::one()).abs2())
                .sum::<f64>();
            let one2_local = x_unscaled.iter().map(|_| S::one().abs2()).sum::<f64>();
            let err = problem.comm.all_reduce_f64(err2_local).sqrt();
            let one_norm = problem.comm.all_reduce_f64(one2_local).sqrt();
            Some(err / one_norm.max(f64::MIN_POSITIVE))
        } else {
            None
        };
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
            operator_storage: operator_storage_label(problem),
            execution_backend: execution_backend_label(problem),
            pc_domain: pc_domain_label(spec.pc),
            method: format!(
                "{} [row-scale={}]",
                spec.method_label(),
                if bench_cfg.row_scale { "on" } else { "off" }
            ),
            requested_policy: spec.requested_policy_label(),
            effective_policy: format!(
                "variant={}, restart={}, residual-check={}",
                stats
                    .effective_variant
                    .as_deref()
                    .unwrap_or(variant_label(spec.variant)),
                stats.effective_restart.unwrap_or(spec.restart),
                stats
                    .effective_residual_check_policy
                    .as_deref()
                    .unwrap_or(residual_check_policy_label(spec.residual_check_policy))
            ) + if bench_cfg.run_mode == RunMode::Correctness
                && bench_cfg.mark_replicated_check
            {
                " [replicated-check=enabled]"
            } else {
                ""
            },
            setup_secs,
            median_solve_secs,
            min_solve_secs,
            iterations: stats.iterations,
            reductions,
            restart_count: stats.fgmres_counters.as_ref().map(|c| c.restart_count),
            inner_iterations_last_cycle: stats
                .fgmres_counters
                .as_ref()
                .map(|c| c.inner_iterations_last_cycle),
            pipeline_fallbacks: stats.fgmres_counters.as_ref().map(|c| c.pipeline_fallbacks),
            reported_residual: stats.final_residual,
            explicit_true_residual,
            explicit_true_residual_rel,
            x_error_rel,
            reason: stats.reason,
            dof_per_sec,
        })
    }

    fn validate_local_ilu_owned_block(matrix: &SparseCsrMatrix<S>) -> Result<(), KError> {
        if matrix.nrows() != matrix.ncols() {
            return Err(KError::InvalidInput(
                "local ILU requires square owned block".into(),
            ));
        }
        Ok(())
    }

    fn csr_for_pc_diagnostics(
        matrix: &SparseCsrMatrix<S>,
        nnz_local_rows: usize,
    ) -> CsrForPcDiagnostics {
        let mut diag_min_abs = f64::INFINITY;
        let mut diag_max_abs = 0.0f64;
        let mut diag_tiny_or_missing_count = 0usize;
        let tiny_diag_threshold = 1e-14f64;
        let row_ptr = matrix.row_ptr();
        let col_idx = matrix.col_idx();
        let values = matrix.values();
        for r in 0..matrix.nrows() {
            let mut diag = None;
            for nz in row_ptr[r]..row_ptr[r + 1] {
                if col_idx[nz] == r {
                    diag = Some(values[nz].abs());
                    break;
                }
            }
            match diag {
                Some(v) => {
                    diag_min_abs = diag_min_abs.min(v);
                    diag_max_abs = diag_max_abs.max(v);
                    if v <= tiny_diag_threshold {
                        diag_tiny_or_missing_count += 1;
                    }
                }
                None => diag_tiny_or_missing_count += 1,
            }
        }
        if !diag_min_abs.is_finite() {
            diag_min_abs = 0.0;
        }
        let nnz_local_block = values.len();
        let nnz_ratio = nnz_local_block as f64 / (nnz_local_rows.max(1) as f64);
        CsrForPcDiagnostics {
            nnz_local_block,
            nnz_local_rows,
            nnz_ratio,
            diag_min_abs,
            diag_max_abs,
            diag_tiny_or_missing_count,
        }
    }

    fn rank_spread_from_locals(locals: &[f64], rank0_local: f64) -> RankSpread {
        let global_sum = locals.iter().sum::<f64>();
        let rank_min = locals.iter().copied().fold(f64::INFINITY, f64::min);
        let rank_max = locals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        RankSpread {
            global_sum,
            rank_min: if rank_min.is_finite() { rank_min } else { 0.0 },
            rank_max: if rank_max.is_finite() { rank_max } else { 0.0 },
            rank0_local,
        }
    }

    fn reduce_rank_spread<C: Comm<Vec = Vec<f64>>>(comm: &C, local: f64) -> RankSpread {
        if comm.size() <= 1 {
            return RankSpread {
                global_sum: local,
                rank_min: local,
                rank_max: local,
                rank0_local: local,
            };
        }
        let global_sum = comm.all_reduce_f64(local);
        let mut gathered = Vec::new();
        comm.gather(&[local], &mut gathered, 0);
        let (rank_min, rank_max) = if comm.rank() == 0 && !gathered.is_empty() {
            (
                gathered.iter().copied().fold(f64::INFINITY, f64::min),
                gathered.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            )
        } else {
            (0.0, 0.0)
        };
        let rank0_local = if comm.rank() == 0 { local } else { 0.0 };
        RankSpread {
            global_sum,
            rank_min,
            rank_max,
            rank0_local,
        }
    }

    fn reduce_csr_for_pc_diagnostics<C: Comm<Vec = Vec<f64>>>(
        comm: &C,
        local: &CsrForPcDiagnostics,
        pivot_perturbation_count: Option<usize>,
    ) -> CsrForPcDiagnosticsGlobal {
        CsrForPcDiagnosticsGlobal {
            nnz_local_block: reduce_rank_spread(comm, local.nnz_local_block as f64),
            nnz_local_rows: reduce_rank_spread(comm, local.nnz_local_rows as f64),
            nnz_ratio: reduce_rank_spread(comm, local.nnz_ratio),
            diag_min_abs: reduce_rank_spread(comm, local.diag_min_abs),
            diag_max_abs: reduce_rank_spread(comm, local.diag_max_abs),
            diag_tiny_or_missing_count: reduce_rank_spread(
                comm,
                local.diag_tiny_or_missing_count as f64,
            ),
            pivot_perturbation_count: pivot_perturbation_count
                .map(|v| reduce_rank_spread(comm, v as f64)),
        }
    }

    fn format_csr_for_pc_diagnostics(method_label: &str, diag: &CsrForPcDiagnosticsGlobal) -> String {
        let pivot_text = diag
            .pivot_perturbation_count
            .as_ref()
            .map(|v| {
                format!(
                    "global_sum={:.0}, rank_min={:.0}, rank_max={:.0}, rank0_local={:.0}",
                    v.global_sum, v.rank_min, v.rank_max, v.rank0_local
                )
            })
            .unwrap_or_else(|| "N/A".to_string());
        format!(
            "[diag][rank0] {method_label}: nnz(A_ii):global_sum={:.0},rank_min={:.0},rank_max={:.0},rank0_local={:.0}; nnz(local rows):global_sum={:.0},rank_min={:.0},rank_max={:.0},rank0_local={:.0}; nnz_ratio:global_sum={:.3e},rank_min={:.3e},rank_max={:.3e},rank0_local={:.3e}; |diag|min:global_sum={:.3e},rank_min={:.3e},rank_max={:.3e},rank0_local={:.3e}; |diag|max:global_sum={:.3e},rank_min={:.3e},rank_max={:.3e},rank0_local={:.3e}; tiny/missing diag:global_sum={:.0},rank_min={:.0},rank_max={:.0},rank0_local={:.0}; pivot perturbations={}",
            diag.nnz_local_block.global_sum, diag.nnz_local_block.rank_min, diag.nnz_local_block.rank_max, diag.nnz_local_block.rank0_local,
            diag.nnz_local_rows.global_sum, diag.nnz_local_rows.rank_min, diag.nnz_local_rows.rank_max, diag.nnz_local_rows.rank0_local,
            diag.nnz_ratio.global_sum, diag.nnz_ratio.rank_min, diag.nnz_ratio.rank_max, diag.nnz_ratio.rank0_local,
            diag.diag_min_abs.global_sum, diag.diag_min_abs.rank_min, diag.diag_min_abs.rank_max, diag.diag_min_abs.rank0_local,
            diag.diag_max_abs.global_sum, diag.diag_max_abs.rank_min, diag.diag_max_abs.rank_max, diag.diag_max_abs.rank0_local,
            diag.diag_tiny_or_missing_count.global_sum, diag.diag_tiny_or_missing_count.rank_min, diag.diag_tiny_or_missing_count.rank_max, diag.diag_tiny_or_missing_count.rank0_local,
            pivot_text
        )
    }

    fn print_csr_for_pc_diagnostics(method_label: &str, diag: &CsrForPcDiagnosticsGlobal) {
        println!("{}", format_csr_for_pc_diagnostics(method_label, diag));
    }

    fn configured_solver(spec: &RunSpec, bench_cfg: &BenchmarkConfig) -> FgmresSolver {
        let mut solver = FgmresSolver::new(bench_cfg.rtol, bench_cfg.maxits, spec.restart);
        solver.variant = spec.variant;
        solver.residual_check_policy = spec.residual_check_policy;
        solver.orthog = spec.orthog;
        solver.reorth = spec.reorth;
        solver.atol = bench_cfg.atol;
        solver.dtol = 1e6;
        solver.haptol = bench_cfg.fgmres_haptol;
        solver.min_inner_before_fallback = bench_cfg.min_inner_before_fallback.max(1);
        solver.stagnation_policy =
            if bench_cfg.run_mode == RunMode::Correctness || !bench_cfg.allow_stagnation_fallback {
                FgmresStagnationPolicy::Disabled
            } else {
                match spec.variant {
                    FgmresVariant::Classical => FgmresStagnationPolicy::RestartClassicalToo,
                    FgmresVariant::Pipelined => FgmresStagnationPolicy::PipelineFallbackOnly,
                }
            };
        solver
    }

    fn apply_dist_plan_policy(solver: &mut FgmresSolver, problem: &Problem) {
        solver.apply_distcsr_policy(&problem.dist_plan_diagnostics, problem.comm.size());
    }

    fn dump_residual_history(path: &Path, history: &RunResidualHistory) -> Result<(), KError> {
        let mut f = File::create(path)
            .map_err(|e| KError::InvalidInput(format!("failed to create history file: {e}")))?;
        if path.extension().and_then(|e| e.to_str()) == Some("json") {
            writeln!(f, "[").map_err(|e| KError::InvalidInput(format!("write failed: {e}")))?;
            for (i, e) in history.entries.iter().enumerate() {
                writeln!(f, "  {{\"iter\":{},\"recurrence_residual\":{},\"true_residual\":{},\"checkpoint\":{}}}{}", e.iter, e.recurrence_residual, e.true_residual.map(|v| v.to_string()).unwrap_or_else(|| "null".to_string()), e.checkpoint, if i + 1 == history.entries.len() {""} else {","})
                    .map_err(|e| KError::InvalidInput(format!("write failed: {e}")))?;
            }
            writeln!(f, "]").map_err(|e| KError::InvalidInput(format!("write failed: {e}")))?;
        } else {
            writeln!(f, "iter,recurrence_residual,true_residual,checkpoint")
                .map_err(|e| KError::InvalidInput(format!("write failed: {e}")))?;
            for e in &history.entries {
                writeln!(
                    f,
                    "{},{},{},{}",
                    e.iter,
                    e.recurrence_residual,
                    e.true_residual.map(|v| v.to_string()).unwrap_or_default(),
                    e.checkpoint
                )
                .map_err(|e| KError::InvalidInput(format!("write failed: {e}")))?;
            }
        }
        Ok(())
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

    fn render_result_row(row: &ResultRow, mode: RunMode, problem: &Problem) -> String {
        let rst = row
            .restart_count
            .map(|v| v.to_string())
            .unwrap_or_else(|| "N/A".to_string());
        let inn = row
            .inner_iterations_last_cycle
            .map(|v| v.to_string())
            .unwrap_or_else(|| "N/A".to_string());
        let pfb = row
            .pipeline_fallbacks
            .map(|v| v.to_string())
            .unwrap_or_else(|| "N/A".to_string());
        let include_dof_col = matches!(
            problem.backend,
            CsrBackend::Serial | CsrBackend::Distributed
        );
        if mode == RunMode::Scalability {
            let dof = row
                .dof_per_sec
                .map(|v| format!("{v:.2e}"))
                .unwrap_or_else(|| "N/A".to_string());
            return format!(
                "{:<7} {:<8} {:<6} {:<36} {:<34} {:>9.3} {:>9.3} {:>7} {:>5} {:>5} {:>5} {:>6} {:>14.2e} {:>12}",
                row.operator_storage,
                row.execution_backend,
                row.pc_domain,
                row.method,
                row.effective_policy,
                row.median_solve_secs,
                row.min_solve_secs,
                row.iterations,
                rst,
                inn,
                pfb,
                row.reductions,
                row.reported_residual,
                dof
            );
        }
        let explicit_true = row
            .explicit_true_residual
            .map(|v| format!("{v:.2e}"))
            .unwrap_or_else(|| "N/A".to_string());
        let explicit_true_rel = row
            .explicit_true_residual_rel
            .map(|v| format!("{v:.2e}"))
            .unwrap_or_else(|| "N/A".to_string());
        let x_error_rel = row
            .x_error_rel
            .map(|v| format!("{v:.2e}"))
            .unwrap_or_else(|| "N/A".to_string());
        if include_dof_col {
            let dof = row
                .dof_per_sec
                .map(|v| format!("{v:.2e}"))
                .unwrap_or_else(|| "N/A".to_string());
            format!(
                "{:<7} {:<8} {:<6} {:<36} {:<34} {:<34} {:>9.3} {:>9.3} {:>9.3} {:>7} {:>5} {:>5} {:>5} {:>6} {:>14.2e} {:>14} {:>14} {:>14} {:>26?} {:>12}",
                row.operator_storage,
                row.execution_backend,
                row.pc_domain,
                row.method,
                row.requested_policy,
                row.effective_policy,
                row.setup_secs,
                row.median_solve_secs,
                row.min_solve_secs,
                row.iterations,
                rst,
                inn,
                pfb,
                row.reductions,
                row.reported_residual,
                explicit_true,
                explicit_true_rel,
                x_error_rel,
                row.reason,
                dof
            )
        } else {
            format!(
                "{:<7} {:<8} {:<6} {:<36} {:<34} {:<34} {:>9.3} {:>9.3} {:>9.3} {:>7} {:>5} {:>5} {:>5} {:>6} {:>14.2e} {:>14} {:>14} {:>14} {:>26?}",
                row.operator_storage,
                row.execution_backend,
                row.pc_domain,
                row.method,
                row.requested_policy,
                row.effective_policy,
                row.setup_secs,
                row.median_solve_secs,
                row.min_solve_secs,
                row.iterations,
                rst,
                inn,
                pfb,
                row.reductions,
                row.reported_residual,
                explicit_true,
                explicit_true_rel,
                x_error_rel,
                row.reason
            )
        }
    }

    fn load_problem_complex(
        mat_path: &Path,
        comm: &UniverseComm,
        _run_mode: RunMode,
        _mark_replicated_check: bool,
    ) -> Result<Problem, KError> {
        let mm = read_matrix_market(mat_path)?;
        let csr_sparse: SparseCsrMatrix<S> = mm.to_csr_matrix_scalar()?;
        let nrows = csr_sparse.nrows();
        let ncols = csr_sparse.ncols();

        let row_part = DistCsrOp::partition_rows_balanced(nrows, comm);
        let row_start = row_part[comm.rank()];
        let row_end = row_part[comm.rank() + 1];
        let local_csr = slice_csr_rows(&csr_sparse, row_start, row_end);
        let local_pc_block = slice_csr_rows_owned_cols(&csr_sparse, row_start, row_end);

        let op = DistCsrOp::from_local_rows(nrows, row_start, &local_csr, &row_part, comm.clone())?;
        let dist_plan_diagnostics = op.plan_diagnostics().clone();
        let op_arc: Arc<dyn KLinOp<Scalar = S>> = Arc::new(op);

        let (rhs_global, rhs_source) = match try_load_rhs_s(mat_path, nrows) {
            Some(vec) => (vec, RhsSource::LoadedFromFile),
            None => {
                let ones = vec![S::one(); ncols];
                let mut b = vec![S::zero(); nrows];
                csr_sparse.spmv(&ones, &mut b);
                (b, RhsSource::GeneratedAOnes)
            }
        };
        let rhs = rhs_global[row_start..row_end].to_vec();

        let local_n = row_end - row_start;
        let global_n = nrows;
        let backend = classify_backend(comm.size(), local_n, global_n);

        Ok(Problem {
            op: op_arc,
            dist_plan_diagnostics,
            rhs,
            rhs_source,
            csr_for_pc: Arc::new(local_pc_block),
            local_rows_nnz: local_csr.values().len(),
            local_n,
            global_n,
            global_row_start: row_start,
            comm: comm.clone(),
            backend,
            backend_descr: "Distributed CSR (complex)".to_string(),
        })
    }

    fn slice_csr_rows(matrix: &SparseCsrMatrix<S>, start: usize, end: usize) -> SparseCsrMatrix<S> {
        let row_ptr = matrix.row_ptr();
        let col_idx = matrix.col_idx();
        let values = matrix.values();
        let start_nnz = row_ptr[start];
        let end_nnz = row_ptr[end];

        let mut local_rp = Vec::with_capacity(end - start + 1);
        for r in start..=end {
            local_rp.push(row_ptr[r] - start_nnz);
        }
        let local_ci = col_idx[start_nnz..end_nnz].to_vec();
        let local_vals = values[start_nnz..end_nnz].to_vec();

        SparseCsrMatrix::from_csr(end - start, matrix.ncols(), local_rp, local_ci, local_vals)
    }

    fn slice_csr_rows_owned_cols(
        matrix: &SparseCsrMatrix<S>,
        row_start: usize,
        row_end: usize,
    ) -> SparseCsrMatrix<S> {
        let row_ptr = matrix.row_ptr();
        let col_idx = matrix.col_idx();
        let values = matrix.values();
        let local_n = row_end - row_start;

        let mut local_rp = Vec::with_capacity(local_n + 1);
        let mut local_ci = Vec::new();
        let mut local_vals = Vec::new();
        local_rp.push(0);

        for global_r in row_start..row_end {
            let mut row_nnz = 0usize;
            for nz in row_ptr[global_r]..row_ptr[global_r + 1] {
                let global_c = col_idx[nz];
                if (row_start..row_end).contains(&global_c) {
                    local_ci.push(global_c - row_start);
                    local_vals.push(values[nz]);
                    row_nnz += 1;
                }
            }
            local_rp.push(local_rp.last().copied().unwrap_or(0) + row_nnz);
        }

        SparseCsrMatrix::from_csr(local_n, local_n, local_rp, local_ci, local_vals)
    }

    fn partition_rows_balanced_for_size(n_global: usize, size: usize) -> Vec<usize> {
        let base = n_global / size;
        let rem = n_global % size;
        let mut out = Vec::with_capacity(size + 1);
        out.push(0);
        for rank in 0..size {
            let take = base + usize::from(rank < rem);
            out.push(out[rank] + take);
        }
        out
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

    #[test]
    fn qc324_owned_pc_block_is_local_square_on_4_way_partition() {
        let mat_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/mtx/qc324.mtx");
        if !mat_path.exists() {
            eprintln!("qc324.mtx unavailable; skipping local PC block-shape check.");
            return;
        }

        let mm = read_matrix_market(&mat_path).expect("read qc324");
        let csr_sparse: SparseCsrMatrix<S> = mm.to_csr_matrix_scalar().expect("qc324 to CSR");
        let n = csr_sparse.nrows();
        let part = partition_rows_balanced_for_size(n, 4);

        for rank in 0..4 {
            let row_start = part[rank];
            let row_end = part[rank + 1];
            let local_n = row_end - row_start;
            let pc_block = slice_csr_rows_owned_cols(&csr_sparse, row_start, row_end);
            eprintln!(
                "qc324 rank {rank}: rows=[{row_start}..{row_end}), pc_block=({} x {})",
                pc_block.nrows(),
                pc_block.ncols()
            );
            assert_eq!(pc_block.nrows(), local_n, "rank {rank} local PC rows");
            assert_eq!(pc_block.ncols(), local_n, "rank {rank} local PC cols");
            assert!(
                pc_block.col_idx().iter().all(|&c| c < local_n),
                "rank {rank} has out-of-range local column index"
            );
        }
    }

    #[test]
    fn local_ilu_rejects_rectangular_owned_block() {
        #[cfg(feature = "mpi")]
        let comm = UniverseComm::Mpi(Arc::new(MpiComm::new()));
        #[cfg(not(feature = "mpi"))]
        let comm = UniverseComm::NoComm(NoComm);

        let op_local =
            SparseCsrMatrix::from_csr(2, 2, vec![0, 1, 2], vec![0, 1], vec![S::one(), S::one()]);
        let row_part = vec![0, 2];
        let op = DistCsrOp::from_local_rows(2, 0, &op_local, &row_part, comm.clone())
            .expect("build local dist op");

        let rectangular_pc =
            SparseCsrMatrix::from_csr(2, 3, vec![0, 1, 2], vec![0, 1], vec![S::one(), S::one()]);
        let problem = Problem {
            op: Arc::new(op),
            dist_plan_diagnostics: DistributedPlanDiagnostics {
                overlap_mode: kryst::matrix::dist_csr::HaloOverlapMode::Disabled,
                kernel_strategy: kryst::matrix::dist_csr::DistLocalKernelStrategy::RowSplitScalar,
                local_spmv_kernel: None,
                row_locality_ratio: 1.0,
                border_ratio: 0.0,
                halo_recv_volume: 0,
                halo_send_volume: 0,
                expected_communication_fraction: 0.0,
                expected_computation_fraction: 1.0,
            },
            rhs: vec![S::one(), S::one()],
            rhs_source: RhsSource::GeneratedAOnes,
            csr_for_pc: Arc::new(rectangular_pc),
            local_rows_nnz: op_local.values().len(),
            local_n: 2,
            global_n: 2,
            global_row_start: 0,
            comm,
            backend: CsrBackend::Serial,
            backend_descr: "unit-test".to_string(),
        };
        let spec = RunSpec {
            restart: 10,
            variant: FgmresVariant::Classical,
            residual_check_policy: ResidualCheckPolicy::OnConvergence,
            orthog: OrthogMethod::ClassicalGS,
            reorth: ReorthPolicy::IfNeeded,
            pc_side: PcSide::Right,
            pc: PcKind::Ilu0Local,
        };
        let bench_cfg = BenchmarkConfig {
            warmup_runs: 0,
            measured_runs: 1,
            ..BenchmarkConfig::default()
        };

        match run_once(&problem, &spec, &bench_cfg) {
            Err(KError::InvalidInput(msg)) => {
                assert_eq!(msg, "local ILU requires square owned block");
            }
            Err(other) => panic!("unexpected error variant: {other:?}"),
            Ok(_) => panic!("expected rectangular ILU error"),
        }
    }

    #[test]
    fn run_once_reports_requested_and_effective_policy_after_dist_policy_mutation() {
        #[cfg(feature = "mpi")]
        let comm = UniverseComm::Mpi(Arc::new(MpiComm::new()));
        #[cfg(not(feature = "mpi"))]
        let comm = UniverseComm::NoComm(NoComm);

        let op_local =
            SparseCsrMatrix::from_csr(2, 2, vec![0, 1, 2], vec![0, 1], vec![S::one(), S::one()]);
        let row_part = vec![0, 2];
        let op = DistCsrOp::from_local_rows(2, 0, &op_local, &row_part, comm.clone())
            .expect("build local dist op");

        let problem = Problem {
            op: Arc::new(op),
            dist_plan_diagnostics: DistributedPlanDiagnostics {
                overlap_mode: kryst::matrix::dist_csr::HaloOverlapMode::Disabled,
                kernel_strategy: kryst::matrix::dist_csr::DistLocalKernelStrategy::RowSplitScalar,
                local_spmv_kernel: None,
                row_locality_ratio: 1.0,
                border_ratio: 0.0,
                halo_recv_volume: 0,
                halo_send_volume: 0,
                expected_communication_fraction: 0.1,
                expected_computation_fraction: 0.5,
            },
            rhs: vec![S::one(), S::one()],
            rhs_source: RhsSource::GeneratedAOnes,
            csr_for_pc: Arc::new(op_local),
            local_rows_nnz: 2,
            local_n: 2,
            global_n: 2,
            global_row_start: 0,
            comm,
            backend: CsrBackend::Serial,
            backend_descr: "unit-test".to_string(),
        };
        let spec = RunSpec {
            restart: 5,
            variant: FgmresVariant::Classical,
            residual_check_policy: ResidualCheckPolicy::OnConvergence,
            orthog: OrthogMethod::ClassicalGS,
            reorth: ReorthPolicy::IfNeeded,
            pc_side: PcSide::Right,
            pc: PcKind::None,
        };
        let bench_cfg = BenchmarkConfig {
            warmup_runs: 0,
            measured_runs: 1,
            ..BenchmarkConfig::default()
        };

        let row = run_once(&problem, &spec, &bench_cfg).expect("run once");
        assert!(row.requested_policy.contains("restart=5"));
        assert!(
            row.requested_policy
                .contains("residual-check=on-convergence")
        );
        assert!(row.effective_policy.contains("restart=16"));
        assert!(
            row.effective_policy
                .contains("residual-check=every-iteration")
        );

        let rendered = render_result_row(&row, RunMode::Correctness, &problem);
        assert!(rendered.contains(&row.requested_policy));
        assert!(rendered.contains(&row.effective_policy));
        assert!(rendered.contains("csr-cx"));
        assert!(rendered.contains("ser"));
    }

    #[test]
    fn metadata_labels_snapshot() {
        assert_eq!(pc_domain_label(PcKind::ReplicatedFullIlu0), "full");
        assert_eq!(pc_domain_label(PcKind::Ilu0Local), "own0");
        assert_eq!(pc_domain_label(PcKind::None), "n/a");
    }

    #[test]
    fn mpi_mode_run_matrix_has_no_duplicate_semantic_pc_experiments() {
        let cfg = BenchmarkConfig::default();
        let mpi_mode = true;
        let pcs = if cfg.pcs.is_empty() {
            match cfg.run_mode {
                RunMode::Correctness => {
                    if mpi_mode {
                        vec![
                            PcKind::MpiBlockJacobiIlu0Local,
                            PcKind::JacobiWeak,
                            PcKind::None,
                        ]
                    } else {
                        vec![PcKind::Ilu0Local, PcKind::JacobiWeak, PcKind::None]
                    }
                }
                RunMode::Scalability => {
                    if mpi_mode {
                        vec![PcKind::MpiBlockJacobiIlu0Local]
                    } else {
                        vec![PcKind::Ilu0Local]
                    }
                }
                RunMode::Robustness => {
                    if mpi_mode {
                        vec![PcKind::MpiBlockJacobiIlu0Local, PcKind::JacobiWeak]
                    } else {
                        vec![PcKind::Ilu0Local, PcKind::JacobiWeak]
                    }
                }
            }
        } else {
            cfg.pcs.clone()
        };

        let mut seen = std::collections::BTreeSet::new();
        for pc in pcs {
            let key = pc.semantic_experiment_key(mpi_mode);
            assert!(
                seen.insert(key.clone()),
                "duplicate semantic PC experiment: {key}"
            );
        }
    }
    #[test]
    fn preconditioner_dispatch_variants_are_distinct_or_explicit_aliases() {
        let variants = [
            PcKind::None,
            PcKind::JacobiWeak,
            PcKind::Ilu0Local,
            PcKind::IlutLocal,
            PcKind::MpiBlockJacobiIlu0Local,
        ];

        for (idx, left) in variants.iter().copied().enumerate() {
            for right in variants.iter().copied().skip(idx + 1) {
                if left.dispatch_branch() == right.dispatch_branch() {
                    let explicit_alias = left.explicit_alias_of() == Some(right)
                        || right.explicit_alias_of() == Some(left);
                    assert!(
                        explicit_alias,
                        "dispatch collision without alias marker: {left:?} vs {right:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn ilut_complex_label_explicitly_marks_degraded_provisional_path() {
        let label = PcKind::IlutLocal.label();
        assert!(label.contains("degraded/provisional"));
        assert!(label.contains("real-projection"));
    }

    #[test]
    fn correctness_mode_disables_stagnation_fallback_and_keeps_restart_target() {
        let spec = RunSpec {
            restart: 50,
            variant: FgmresVariant::Classical,
            residual_check_policy: ResidualCheckPolicy::OnConvergence,
            orthog: OrthogMethod::ClassicalGS,
            reorth: ReorthPolicy::IfNeeded,
            pc_side: PcSide::Right,
            pc: PcKind::None,
        };
        let cfg = BenchmarkConfig {
            run_mode: RunMode::Correctness,
            allow_stagnation_fallback: true,
            min_inner_before_fallback: 12,
            ..BenchmarkConfig::default()
        };
        let solver = configured_solver(&spec, &cfg);
        assert_eq!(solver.restart, 50);
        assert_eq!(solver.stagnation_policy, FgmresStagnationPolicy::Disabled);
        assert_eq!(solver.min_inner_before_fallback, 12);
    }

    #[test]
    fn benchmark_config_uses_conservative_default_fgmres_haptol() {
        let cfg = BenchmarkConfig::default();
        assert_eq!(cfg.fgmres_haptol, 1e-30);
    }

    #[test]
    fn benchmark_config_parses_fgmres_haptol_override() {
        let cfg = BenchmarkConfig::from_args(vec![
            "--fgmres-haptol".to_string(),
            "1e-22".to_string(),
        ])
        .expect("parse args");
        assert_eq!(cfg.fgmres_haptol, 1e-22);
    }

    #[test]
    fn benchmark_config_rejects_non_positive_or_non_finite_fgmres_haptol() {
        for bad in ["0", "-1", "NaN", "inf"] {
            let err = BenchmarkConfig::from_args(vec![
                "--fgmres-haptol".to_string(),
                bad.to_string(),
            ])
            .expect_err("invalid --fgmres-haptol should be rejected");
            match err {
                KError::InvalidInput(msg) => assert!(msg.contains("--fgmres-haptol")),
                other => panic!("unexpected error variant: {other:?}"),
            }
        }
    }

    #[test]
    fn benchmark_config_accepts_mark_replicated_check_flag() {
        let cfg = BenchmarkConfig::from_args(vec!["--mark-replicated-check".to_string()])
            .expect("parse args");
        assert!(cfg.mark_replicated_check);
    }

    #[test]
    fn benchmark_config_rejects_deprecated_correctness_replicated_check_flag() {
        let err = BenchmarkConfig::from_args(vec!["--correctness-replicated-check".to_string()])
            .expect_err("deprecated flag should be rejected");
        match err {
            KError::InvalidInput(msg) => {
                assert!(msg.contains("renamed to --mark-replicated-check"))
            }
            other => panic!("unexpected error variant: {other:?}"),
        }
    }

    #[test]
    fn diagnostics_format_includes_global_and_rank_spread_fields() {
        let diag = CsrForPcDiagnosticsGlobal {
            nnz_local_block: rank_spread_from_locals(&[12.0, 30.0], 12.0),
            nnz_local_rows: rank_spread_from_locals(&[10.0, 20.0], 10.0),
            nnz_ratio: rank_spread_from_locals(&[1.2, 1.5], 1.2),
            diag_min_abs: rank_spread_from_locals(&[1e-8, 1e-10], 1e-8),
            diag_max_abs: rank_spread_from_locals(&[9.0, 11.0], 9.0),
            diag_tiny_or_missing_count: rank_spread_from_locals(&[1.0, 4.0], 1.0),
            pivot_perturbation_count: Some(rank_spread_from_locals(&[0.0, 3.0], 0.0)),
        };
        let text = format_csr_for_pc_diagnostics("ilu0-local", &diag);
        assert!(text.contains("global_sum="));
        assert!(text.contains("rank_min="));
        assert!(text.contains("rank_max="));
        assert!(text.contains("rank0_local="));
    }

    #[test]
    fn diagnostics_capture_nonzero_off_rank_contribution() {
        let spread = rank_spread_from_locals(&[5.0, 0.0, 7.0], 5.0);
        assert_eq!(spread.rank0_local, 5.0);
        assert_eq!(spread.global_sum, 12.0);
        assert!(spread.global_sum > spread.rank0_local);
        assert_eq!(spread.rank_max, 7.0);
    }
}

#[cfg(feature = "complex")]
fn main() -> Result<(), KError> {
    complex_demo::run()
}
