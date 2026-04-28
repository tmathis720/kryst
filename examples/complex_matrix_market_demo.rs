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
    use kryst::solver::LinearSolver;
    use kryst::solver::fgmres::{
        FgmresSolver, FgmresVariant, OrthogMethod, PipelinePolicy, ResidualCheckPolicy,
    };
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
                }
            );
            if config.run_mode == RunMode::Correctness {
                println!(
                    "Replicated operator checks: {}",
                    if config.correctness_replicated_check {
                        "enabled (metadata/checkpoint reporting)"
                    } else {
                        "disabled"
                    }
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
                config.correctness_replicated_check,
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
                if config.run_mode == RunMode::Correctness && config.correctness_replicated_check {
                    println!(
                        "Replicated check marker: ENABLED (log-only marker for cross-run comparison)."
                    );
                }
                let include_dof_col = matches!(
                    problem.backend,
                    CsrBackend::Serial | CsrBackend::Distributed
                );
                if config.run_mode == RunMode::Scalability {
                    println!(
                        "{:<36} {:<34} {:>9} {:>9} {:>7} {:>6} {:>14} {:>12}",
                        "Method",
                        "Effective policy",
                        "Med(s)",
                        "Min(s)",
                        "Iters",
                        "Reds",
                        "Rec/Reported",
                        "DOF/s"
                    );
                    println!("{}", "-".repeat(140));
                } else if include_dof_col {
                    println!(
                        "{:<36} {:<34} {:<34} {:>9} {:>9} {:>9} {:>7} {:>6} {:>14} {:>14} {:>26} {:>12}",
                        "Method",
                        "Requested policy",
                        "Effective policy",
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
                    println!("{}", "-".repeat(236));
                } else {
                    println!(
                        "{:<36} {:<34} {:<34} {:>9} {:>9} {:>9} {:>7} {:>6} {:>14} {:>14} {:>26}",
                        "Method",
                        "Requested policy",
                        "Effective policy",
                        "Setup(s)",
                        "Med(s)",
                        "Min(s)",
                        "Iters",
                        "Reds",
                        "Rec/Reported",
                        "True(explicit)",
                        "Reason"
                    );
                    println!("{}", "-".repeat(222));
                }
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
                                    "{:<36} {:<34} {:<34} {:>9} {:>9} {:>9} {:>7} {:>6} {:>14} {:>14} {:>26}",
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
        csr_for_pc: Arc<SparseCsrMatrix<S>>,
        local_n: usize,
        global_n: usize,
        global_row_start: usize,
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
        requested_policy: String,
        effective_policy: String,
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
        restart: usize,
        variant: FgmresVariant,
        residual_check_policy: ResidualCheckPolicy,
        orthog: OrthogMethod,
        reorth: ReorthPolicy,
        pc_side: PcSide,
        pc: PcKind,
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum PcKind {
        None,
        JacobiWeak,
        Ilu0Local,
        MpiBlockJacobiIlu0Local,
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum PcDispatchBranch {
        None,
        JacobiWeak,
        Ilu0Local,
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
        disable_stagnation_restart: bool,
        correctness_replicated_check: bool,
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum RunMode {
        Correctness,
        Scalability,
    }

    impl RunMode {
        fn label(self) -> &'static str {
            match self {
                Self::Correctness => "correctness",
                Self::Scalability => "scalability",
            }
        }

        fn parse(token: &str) -> Result<Self, KError> {
            match token.trim().to_ascii_lowercase().as_str() {
                "correctness" => Ok(Self::Correctness),
                "scalability" => Ok(Self::Scalability),
                other => Err(KError::InvalidInput(format!(
                    "invalid run mode '{other}', expected correctness|scalability"
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
                disable_stagnation_restart: false,
                correctness_replicated_check: false,
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
                                "Usage: cargo mpirun -n <ranks> --example complex_matrix_market_demo --features complex,mpi,mpi_examples -- [--mode correctness|scalability] [--correctness-replicated-check] [--warmup-runs N] [--measured-runs N] [--rtol F] [--atol F] [--maxits N] [--restarts csv] [--pcs csv] [--include-restart-200] [--disable-stagnation-restart] [--fgmres-variant csv] [--fgmres-orthog csv] [--fgmres-reorth csv]"
                            );
                        } else {
                            println!(
                                "Usage: cargo run --example complex_matrix_market_demo --features complex -- [--mode correctness|scalability] [--correctness-replicated-check] [--warmup-runs N] [--measured-runs N] [--rtol F] [--atol F] [--maxits N] [--restarts csv] [--pcs csv] [--include-restart-200] [--disable-stagnation-restart] [--fgmres-variant csv] [--fgmres-orthog csv] [--fgmres-reorth csv]"
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
                    "--correctness-replicated-check" => {
                        cfg.correctness_replicated_check = true;
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
                            return Err(KError::InvalidInput(
                                "missing value for --maxits".into(),
                            ));
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
                    "--disable-stagnation-restart" => {
                        cfg.disable_stagnation_restart = true;
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
                    "--fgmres-reorth" => {
                        let Some(v) = args.next() else {
                            return Err(KError::InvalidInput(
                                "missing value for --fgmres-reorth".into(),
                            ));
                        };
                        cfg.reorths = parse_reorth_csv("--fgmres-reorth", &v)?;
                    }
                    _ => {}
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
                cfg.correctness_replicated_check = false;
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
        match token.trim().to_ascii_lowercase().as_str() {
            "none" | "off" => Ok(PcKind::None),
            "jacobi" | "jacobi-weak" | "weak-jacobi" => Ok(PcKind::JacobiWeak),
            "ilu0" | "ilu0-local" | "local-ilu0" => Ok(PcKind::Ilu0Local),
            "mpi-block-jacobi-ilu0" | "block-jacobi-ilu0" | "mpi-block-ilu0" => {
                Ok(PcKind::MpiBlockJacobiIlu0Local)
            }
            other => Err(KError::InvalidInput(format!(
                "invalid pc '{other}', expected none|jacobi|ilu0|mpi-block-jacobi-ilu0"
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
                    RunMode::Correctness => vec![
                        PcKind::Ilu0Local,
                        PcKind::MpiBlockJacobiIlu0Local,
                        PcKind::JacobiWeak,
                        PcKind::None,
                    ],
                    RunMode::Scalability => vec![PcKind::Ilu0Local],
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
                                if pc == PcKind::MpiBlockJacobiIlu0Local && problem.comm.size() <= 1 {
                                    continue;
                                }
                                runs.push(Self {
                                    restart,
                                    variant,
                                    residual_check_policy: match cfg.run_mode {
                                        RunMode::Correctness => ResidualCheckPolicy::OnConvergence,
                                        RunMode::Scalability => ResidualCheckPolicy::RestartOnly,
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

    impl PcKind {
        fn label(&self) -> &'static str {
            match self {
                Self::None => "none (unpreconditioned reference)",
                Self::JacobiWeak => "jacobi (weak baseline)",
                Self::Ilu0Local => "local ILU(0) (strong baseline)",
                Self::MpiBlockJacobiIlu0Local => {
                    "MPI block-Jacobi + local ILU(0) [alias: local ILU(0) path]"
                }
            }
        }

        fn dispatch_branch(self) -> PcDispatchBranch {
            match self {
                Self::None => PcDispatchBranch::None,
                Self::JacobiWeak => PcDispatchBranch::JacobiWeak,
                Self::Ilu0Local | Self::MpiBlockJacobiIlu0Local => PcDispatchBranch::Ilu0Local,
            }
        }

        fn explicit_alias_of(self) -> Option<Self> {
            match self {
                Self::MpiBlockJacobiIlu0Local => Some(Self::Ilu0Local),
                Self::None | Self::JacobiWeak | Self::Ilu0Local => None,
            }
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

    fn run_once(
        problem: &Problem,
        spec: &RunSpec,
        bench_cfg: &BenchmarkConfig,
    ) -> Result<ResultRow, KError> {
        let b = &problem.rhs;
        let effective_pc_side = normalized_fgmres_side(spec.pc_side);
        enum PcHandle {
            Jacobi(Jacobi),
            Ilu0(IluCsr),
            MpiBlockJacobiIlu0(IluCsr),
        }

        impl PcHandle {
            fn as_kpc_mut(&mut self) -> &mut dyn KPreconditioner<Scalar = S> {
                match self {
                    Self::Jacobi(pc) => pc,
                    Self::Ilu0(pc) => pc,
                    Self::MpiBlockJacobiIlu0(pc) => pc,
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
                cfg.kind = IluKind::Ilu0;
                let mut ilu = IluCsr::new_with_config(cfg);
                ilu.setup(problem.csr_for_pc.as_ref())?;
                pc = Some(if spec.pc == PcKind::Ilu0Local {
                    PcHandle::Ilu0(ilu)
                } else {
                    PcHandle::MpiBlockJacobiIlu0(ilu)
                });
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
                problem.op.as_ref(),
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
        for _ in 0..bench_cfg.measured_runs {
            let mut x = vec![S::zero(); problem.local_n];
            problem.comm.barrier();
            let start = Instant::now();
            let mut solver = configured_solver(spec, bench_cfg);
            apply_dist_plan_policy(&mut solver, problem);
            let mut workspace = Workspace::new(problem.local_n);
            solver.setup_workspace(&mut workspace);
            let stats = solver.solve_k(
                problem.op.as_ref(),
                pc.as_mut().map(PcHandle::as_kpc_mut),
                b,
                &mut x,
                effective_pc_side,
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
        let explicit_true_residual = if bench_cfg.run_mode == RunMode::Correctness {
            let mut ax = vec![S::zero(); b.len()];
            let mut scratch = BridgeScratch::default();
            problem.op.matvec_s(&x_last, &mut ax, &mut scratch);
            for (ri, bi) in ax.iter_mut().zip(b.iter().copied()) {
                *ri = bi - *ri;
            }
            let r2_local = ax.iter().map(|v| v.abs2()).sum::<f64>();
            Some(problem.comm.all_reduce_f64(r2_local).sqrt())
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
            method: spec.method_label(),
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
                && bench_cfg.correctness_replicated_check
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
            reported_residual: stats.final_residual,
            explicit_true_residual,
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

    fn configured_solver(spec: &RunSpec, bench_cfg: &BenchmarkConfig) -> FgmresSolver {
        let mut solver = FgmresSolver::new(bench_cfg.rtol, bench_cfg.maxits, spec.restart);
        solver.variant = spec.variant;
        solver.residual_check_policy = spec.residual_check_policy;
        solver.orthog = spec.orthog;
        solver.reorth = spec.reorth;
        solver.atol = bench_cfg.atol;
        solver.dtol = 1e6;
        if bench_cfg.disable_stagnation_restart {
            solver.pipeline_policy = PipelinePolicy::Strict;
        }
        solver
    }

    fn apply_dist_plan_policy(solver: &mut FgmresSolver, problem: &Problem) {
        solver.apply_distcsr_policy(&problem.dist_plan_diagnostics, problem.comm.size());
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
                "{:<36} {:<34} {:>9.3} {:>9.3} {:>7} {:>6} {:>14.2e} {:>12}",
                row.method,
                row.effective_policy,
                row.median_solve_secs,
                row.min_solve_secs,
                row.iterations,
                row.reductions,
                row.reported_residual,
                dof
            );
        }
        let explicit_true = row
            .explicit_true_residual
            .map(|v| format!("{v:.2e}"))
            .unwrap_or_else(|| "N/A".to_string());
        if include_dof_col {
            let dof = row
                .dof_per_sec
                .map(|v| format!("{v:.2e}"))
                .unwrap_or_else(|| "N/A".to_string());
            format!(
                "{:<36} {:<34} {:<34} {:>9.3} {:>9.3} {:>9.3} {:>7} {:>6} {:>14.2e} {:>14} {:>26?} {:>12}",
                row.method,
                row.requested_policy,
                row.effective_policy,
                row.setup_secs,
                row.median_solve_secs,
                row.min_solve_secs,
                row.iterations,
                row.reductions,
                row.reported_residual,
                explicit_true,
                row.reason,
                dof
            )
        } else {
            format!(
                "{:<36} {:<34} {:<34} {:>9.3} {:>9.3} {:>9.3} {:>7} {:>6} {:>14.2e} {:>14} {:>26?}",
                row.method,
                row.requested_policy,
                row.effective_policy,
                row.setup_secs,
                row.median_solve_secs,
                row.min_solve_secs,
                row.iterations,
                row.reductions,
                row.reported_residual,
                explicit_true,
                row.reason
            )
        }
    }

    fn load_problem_complex(
        mat_path: &Path,
        comm: &UniverseComm,
        _run_mode: RunMode,
        _correctness_replicated_check: bool,
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

        let rhs_global = match try_load_rhs_s(mat_path, nrows) {
            Some(vec) => vec,
            None => {
                let ones = vec![S::one(); ncols];
                let mut b = vec![S::zero(); nrows];
                csr_sparse.spmv(&ones, &mut b);
                b
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
            csr_for_pc: Arc::new(local_pc_block),
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
            csr_for_pc: Arc::new(rectangular_pc),
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
            csr_for_pc: Arc::new(op_local),
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
    }

    #[test]
    fn preconditioner_dispatch_variants_are_distinct_or_explicit_aliases() {
        let variants = [
            PcKind::None,
            PcKind::JacobiWeak,
            PcKind::Ilu0Local,
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
}

#[cfg(feature = "complex")]
fn main() -> Result<(), KError> {
    complex_demo::run()
}
