#![cfg(feature = "mpi")]
//! Large-scale MPI example demonstrating Matrix Market I/O with configurable solvers and preconditioners.
//!
//! This example shows how to:
//! 1. Read a sparse matrix and RHS vector from Matrix Market files
//! 2. Set up distributed parallel computation using MPI
//! 3. Use a unified KspContext for runtime solver and preconditioner selection
//! 4. Run a two-stage flow: reference direct solve first, iterative experiment second
//!
//! Usage:
//!   cargo mpirun -n 4 --example mpi_amg_gmres_demo --features mpi_examples [options]
//!
//! PETSc-style Options:
//!   -ksp_type <solver>         Iterative solver type (gmres, fgmres, bicgstab, ...)
//!   -pc_type <precond>         Preconditioner type for iterative phase
//!   -ksp_rtol <tol>            Relative tolerance [default: 1e-8]
//!   -ksp_atol <tol>            Absolute tolerance [default: 1e-12]
//!   -ksp_dtol <tol>            Divergence tolerance [default: 1e5]
//!   -ksp_max_it <iters>        Maximum iterations [default: 2000]
//!   -ksp_gmres_restart <n>     GMRES restart parameter [default: 200]
//!   -ksp_pc_side <side>        Preconditioning side [default: left; fgmres requires right]
//!   -matrix <path>             Matrix file path [default: examples/e05r0300/e05r0300.mtx]
//!   -rhs <path>                RHS vector file path [default: examples/e05r0300/e05r0300_rhs1.mtx]
//!   --reference-mode <mode>    auto|local|distributed|skip [default: auto]
//!   --diagnostic-profile <p>   off|right-krylov [default: off]
//!   --mpi-pc-stage <mode>      auto|local-direct-ref|bj-lu|bj-ilut|asm-lu|asm-ilut|fgmres-right-subksp
//!   --allow-divergence         Return success even if iterative phase fails verification

#[cfg(feature = "complex")]
fn main() {
    eprintln!("mpi_amg_gmres_demo.rs is unavailable when built with --features complex");
}

use kryst::config::options::parse_all_options;
use kryst::context::ksp_context::KspContext;
use kryst::matrix::op::{CsrOp, LinOp, wrap_with_comm};
use kryst::matrix::preprocess::{
    SystemTransform, build_inverse, col_1_norm_scaling, matrix_diagnostics,
    nonsymmetric_max_transversal_permutation, permute_rows_cols, row_1_norm_scaling, scale_cols,
    scale_rows,
};
use kryst::parallel::{Comm, UniverseComm};
use kryst::solver::MonitorAction;
use kryst::solver::gmres::StagnationPolicy;
use kryst::utils::matrix_market::{MatrixMarketData, read_matrix_market, write_vector_market};
use std::collections::HashMap;
use std::env;
use std::str::FromStr;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[cfg(feature = "mpi")]
use kryst::parallel::MpiComm;
use kryst::preconditioner::dist::{DistCoarseSolverRoute, DistCoarseStrategy};
#[cfg(feature = "dense-direct")]
use kryst::solver::dense_lu;
use kryst::utils::convergence::SolveStats;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PhaseMode {
    Local,
    Distributed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReferenceMode {
    Auto,
    DistributedDirect,
    LocalDirectReplicated,
    Skip,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DiagnosticProfile {
    Off,
    RightKrylovResidualAudit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MpiPcStageMode {
    Auto,
    LocalDirectRef,
    BlockJacobiLocalLu,
    BlockJacobiLocalIlut,
    AsmLocalLu,
    AsmLocalIlut,
    FgmresRightSubdomainKsp,
}

#[derive(Debug, Clone, Copy)]
struct PreprocessConfig {
    row_scale: bool,
    col_scale: bool,
    matching_perm: bool,
}

impl Default for PreprocessConfig {
    fn default() -> Self {
        Self {
            row_scale: false,
            col_scale: false,
            matching_perm: false,
        }
    }
}

#[derive(Debug, Clone)]
struct MatrixInspection {
    nrows: usize,
    ncols: usize,
    zero_diagonal_rows: usize,
    diagonal_dominance_violations: usize,
    symmetry_ratio: f64,
    is_likely_symmetric: bool,
}

fn inspect_matrix(data: &MatrixMarketData) -> MatrixInspection {
    let mut entries: HashMap<(usize, usize), f64> = HashMap::with_capacity(data.values.len());
    let mut row_abs_sum = vec![0.0; data.rows];
    let mut diag_abs = vec![0.0; data.rows];
    let mut diag_present = vec![false; data.rows];

    for idx in 0..data.values.len() {
        let r = data.row_indices[idx];
        let c = data.col_indices[idx];
        let v = data.values[idx];
        entries.insert((r, c), v);
        row_abs_sum[r] += v.abs();
        if r == c {
            diag_abs[r] += v.abs();
            if v.abs() > 0.0 {
                diag_present[r] = true;
            }
        }
    }

    let zero_diagonal_rows = diag_present.iter().filter(|&&present| !present).count();
    let diagonal_dominance_violations = (0..data.rows)
        .filter(|&r| diag_abs[r] + 1e-15 < row_abs_sum[r] - diag_abs[r])
        .count();

    let mut symmetry_pairs = 0usize;
    let mut symmetric_pairs = 0usize;
    for (&(r, c), &v_rc) in &entries {
        if r >= c {
            continue;
        }
        if let Some(v_cr) = entries.get(&(c, r)) {
            symmetry_pairs += 1;
            let diff = (v_rc - *v_cr).abs();
            let scale = 1.0 + v_rc.abs().max(v_cr.abs());
            if diff <= 1e-9 * scale {
                symmetric_pairs += 1;
            }
        }
    }

    let symmetry_ratio = if symmetry_pairs == 0 {
        0.0
    } else {
        symmetric_pairs as f64 / symmetry_pairs as f64
    };

    MatrixInspection {
        nrows: data.rows,
        ncols: data.cols,
        zero_diagonal_rows,
        diagonal_dominance_violations,
        symmetry_ratio,
        is_likely_symmetric: symmetry_ratio > 0.98,
    }
}

fn compute_relative_residual(csr: &CsrOp<f64>, x: &[f64], rhs: &[f64], comm: &UniverseComm) -> f64 {
    let mut ax = vec![0.0; rhs.len()];
    csr.matvec(x, &mut ax);
    let local_residual_sq = rhs
        .iter()
        .zip(ax.iter())
        .map(|(&b, &axv)| {
            let r = b - axv;
            r * r
        })
        .sum::<f64>();
    let local_rhs_sq = rhs.iter().map(|v| v * v).sum::<f64>();

    let global_residual_sq = comm.all_reduce_f64(local_residual_sq);
    let global_rhs_sq = comm.all_reduce_f64(local_rhs_sq);

    global_residual_sq.sqrt() / global_rhs_sq.sqrt()
}

fn compute_relative_residual_local(csr: &CsrOp<f64>, x: &[f64], rhs: &[f64]) -> f64 {
    let mut ax = vec![0.0; rhs.len()];
    csr.matvec(x, &mut ax);
    let residual_sq = rhs
        .iter()
        .zip(ax.iter())
        .map(|(&b, &axv)| {
            let r = b - axv;
            r * r
        })
        .sum::<f64>();
    let rhs_sq = rhs.iter().map(|v| v * v).sum::<f64>();
    residual_sq.sqrt() / rhs_sq.sqrt()
}

#[allow(clippy::too_many_arguments)]
fn run_phase(
    phase_name: &str,
    solver: &str,
    pc: &str,
    pc_side: &str,
    rtol: f64,
    atol: f64,
    dtol: f64,
    max_iters: usize,
    restart: usize,
    rhs: &[f64],
    base_op: Arc<CsrOp<f64>>,
    comm: UniverseComm,
    mode: PhaseMode,
    full_gmres_sanity: bool,
    rank: usize,
) -> Result<
    (
        SolveStats<f64>,
        Vec<f64>,
        Vec<(usize, f64)>,
        Duration,
        Duration,
    ),
    Box<dyn std::error::Error>,
> {
    let mut ksp = KspContext::new();
    ksp.set_type_from_str(solver)?;
    ksp.set_pc_type_from_str(pc)?;
    ksp.set_tolerances(rtol, atol, dtol, max_iters);
    ksp.set_restart(restart);
    ksp.set_pc_side_from_str(pc_side)?;
    if full_gmres_sanity {
        ksp.set_gmres_stagnation_policy(StagnationPolicy::LogOnly);
    }

    let monitor_data = Arc::new(Mutex::new(Vec::<(usize, f64)>::new()));
    let monitor_data_clone = monitor_data.clone();
    let phase_name_owned = phase_name.to_string();
    let monitor = Box::new(move |iter: usize, residual: f64, _reductions: usize| {
        if let Ok(mut data) = monitor_data_clone.lock() {
            data.push((iter, residual));
            if rank == 0 && (iter == 0 || iter % 20 == 0 || residual < rtol) {
                println!(
                    "    [{}] iter {:4}: internal residual = {:.6e}",
                    phase_name_owned, iter, residual
                );
            }
        }
        MonitorAction::Continue
    });
    ksp.add_monitor(monitor);

    let start_setup = Instant::now();
    match mode {
        PhaseMode::Local => {
            ksp.set_operators(base_op.clone(), None);
        }
        PhaseMode::Distributed => {
            let op_arc = wrap_with_comm(base_op.clone(), comm.clone());
            ksp.set_operators_with_comm(op_arc, None, comm.clone());
        }
    }
    ksp.setup()?;
    if rank == 0
        && let Some((active_restart, active_max_iters, variant, reorth, stagnation)) =
            ksp.debug_gmres_runtime()
    {
        println!(
            "    [{}] live gmres config: restart={}, max_iters={}, variant={:?}, reorth={:?}, stagnation={:?}",
            phase_name, active_restart, active_max_iters, variant, reorth, stagnation
        );
    }
    let setup_time = start_setup.elapsed();

    if mode == PhaseMode::Distributed {
        comm.barrier();
    }
    let mut solution = vec![0.0; rhs.len()];
    let start_solve = Instant::now();
    let stats = ksp.solve(rhs, &mut solution)?;
    let solve_time = start_solve.elapsed();

    let history = monitor_data
        .lock()
        .map(|data| data.clone())
        .unwrap_or_default();
    Ok((stats, solution, history, setup_time, solve_time))
}

fn parse_reference_mode_arg(args: &[String]) -> Result<ReferenceMode, Box<dyn std::error::Error>> {
    let mut idx = 0usize;
    while idx < args.len() {
        if args[idx] == "--reference-mode" {
            let value = args
                .get(idx + 1)
                .ok_or("--reference-mode requires one of: auto, local, distributed, skip")?;
            return match value.as_str() {
                "auto" => Ok(ReferenceMode::Auto),
                "local" => Ok(ReferenceMode::LocalDirectReplicated),
                "distributed" => Ok(ReferenceMode::DistributedDirect),
                "skip" => Ok(ReferenceMode::Skip),
                _ => Err(format!(
                    "Invalid --reference-mode '{}'; expected one of: auto, local, distributed, skip",
                    value
                )
                .into()),
            };
        }
        idx += 1;
    }
    Ok(ReferenceMode::Auto)
}

fn parse_diagnostic_profile_arg(
    args: &[String],
) -> Result<DiagnosticProfile, Box<dyn std::error::Error>> {
    let mut idx = 0usize;
    while idx < args.len() {
        if args[idx] == "--diagnostic-profile" {
            let value = args
                .get(idx + 1)
                .ok_or("--diagnostic-profile requires one of: off, right-krylov")?;
            return match value.as_str() {
                "off" => Ok(DiagnosticProfile::Off),
                "right-krylov" => Ok(DiagnosticProfile::RightKrylovResidualAudit),
                _ => Err(format!(
                    "Invalid --diagnostic-profile '{}'; expected one of: off, right-krylov",
                    value
                )
                .into()),
            };
        }
        idx += 1;
    }
    Ok(DiagnosticProfile::Off)
}

fn parse_preprocess_config(args: &[String]) -> PreprocessConfig {
    let mut cfg = PreprocessConfig::default();
    for arg in args {
        match arg.as_str() {
            "--pre-row-scale" => cfg.row_scale = true,
            "--pre-col-scale" => cfg.col_scale = true,
            "--pre-match-permute" => cfg.matching_perm = true,
            _ => {}
        }
    }
    cfg
}

fn parse_experiment_preset(args: &[String]) -> Option<String> {
    args.windows(2)
        .find(|w| w[0] == "--experiment-preset")
        .map(|w| w[1].clone())
}

fn parse_mpi_pc_stage_arg(args: &[String]) -> Result<MpiPcStageMode, Box<dyn std::error::Error>> {
    let mut idx = 0usize;
    while idx < args.len() {
        if args[idx] == "--mpi-pc-stage" {
            let value = args.get(idx + 1).ok_or(
                "--mpi-pc-stage requires one of: auto, local-direct-ref, bj-lu, bj-ilut, asm-lu, asm-ilut, fgmres-right-subksp",
            )?;
            return match value.as_str() {
                "auto" => Ok(MpiPcStageMode::Auto),
                "local-direct-ref" => Ok(MpiPcStageMode::LocalDirectRef),
                "bj-lu" => Ok(MpiPcStageMode::BlockJacobiLocalLu),
                "bj-ilut" => Ok(MpiPcStageMode::BlockJacobiLocalIlut),
                "asm-lu" => Ok(MpiPcStageMode::AsmLocalLu),
                "asm-ilut" => Ok(MpiPcStageMode::AsmLocalIlut),
                "fgmres-right-subksp" => Ok(MpiPcStageMode::FgmresRightSubdomainKsp),
                _ => Err(format!(
                    "Invalid --mpi-pc-stage '{}'; expected auto, local-direct-ref, bj-lu, bj-ilut, asm-lu, asm-ilut, fgmres-right-subksp",
                    value
                )
                .into()),
            };
        }
        idx += 1;
    }
    Ok(MpiPcStageMode::Auto)
}

fn monitor_semantics_tag(solver: &str, pc_side: &str) -> &'static str {
    if solver.eq_ignore_ascii_case("fgmres") || pc_side.eq_ignore_ascii_case("right") {
        "true"
    } else {
        "preconditioned"
    }
}

#[derive(Debug)]
enum ReferenceResult {
    Solved {
        x: Vec<f64>,
        setup_time: Duration,
        solve_time: Duration,
        solver_label: &'static str,
    },
    Skipped(&'static str),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReferenceSolveStatus {
    Succeeded,
    Skipped,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WrittenOutputVerificationStatus {
    Verified,
    Marginal,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IterativeSanityGateStatus {
    Passed,
    ExperimentalFailed,
}

fn format_phase_status_summary(
    reference_status: ReferenceSolveStatus,
    output_status: WrittenOutputVerificationStatus,
    sanity_status: IterativeSanityGateStatus,
) -> Vec<String> {
    let reference_line = match reference_status {
        ReferenceSolveStatus::Succeeded => "reference solve status: succeeded",
        ReferenceSolveStatus::Skipped => "reference solve status: skipped",
    };
    let output_line = match output_status {
        WrittenOutputVerificationStatus::Verified => {
            "written-output verification status: written output verified successfully"
        }
        WrittenOutputVerificationStatus::Marginal => {
            "written-output verification status: marginal residual quality"
        }
        WrittenOutputVerificationStatus::Failed => "written-output verification status: failed",
    };
    let sanity_line = match sanity_status {
        IterativeSanityGateStatus::Passed => "iterative sanity-gate status: passed",
        IterativeSanityGateStatus::ExperimentalFailed => {
            "iterative sanity-gate status: iterative phase failed sanity gate"
        }
    };
    vec![
        reference_line.to_string(),
        output_line.to_string(),
        sanity_line.to_string(),
    ]
}

fn run_local_dense_reference(
    matrix_data: &MatrixMarketData,
    rhs: &[f64],
) -> Result<ReferenceResult, Box<dyn std::error::Error>> {
    #[cfg(not(feature = "dense-direct"))]
    {
        let _ = (matrix_data, rhs);
        return Ok(ReferenceResult::Skipped(
            "dense-direct feature is required for local replicated reference solve.",
        ));
    }

    #[cfg(feature = "dense-direct")]
    {
        let start_setup = Instant::now();
        let dense = matrix_data.to_dense_matrix_scalar()?;
        let setup_time = start_setup.elapsed();

        let mut x = vec![0.0; rhs.len()];
        let start_solve = Instant::now();
        dense_lu::solve(&dense, rhs, &mut x)?;
        let solve_time = start_solve.elapsed();

        Ok(ReferenceResult::Solved {
            x,
            setup_time,
            solve_time,
            solver_label: "dense direct reference solve",
        })
    }
}

fn collective_phase_result<T>(
    phase_name: &str,
    result: Result<T, Box<dyn std::error::Error>>,
    comm: &UniverseComm,
    rank: usize,
) -> Result<T, Box<dyn std::error::Error>> {
    let local_fail = if result.is_err() { 1.0 } else { 0.0 };
    let any_fail = comm.all_reduce_f64(local_fail) > 0.0;
    if any_fail {
        if rank == 0 {
            if let Err(err) = &result {
                eprintln!("Collective failure in {phase_name}: {err}");
            } else {
                eprintln!("Collective failure in {phase_name}: another rank reported an error.");
            }
        }
        comm.barrier();
        return Err(format!("Collective failure in {phase_name}").into());
    }
    result
}

#[allow(clippy::too_many_arguments)]
fn run_reference_phase(
    reference_mode: ReferenceMode,
    matrix_data: &MatrixMarketData,
    rhs: &[f64],
    csr_op: Arc<CsrOp<f64>>,
    comm: &UniverseComm,
    rank: usize,
    rtol: f64,
    atol: f64,
    dtol: f64,
    max_iters: usize,
    restart: usize,
) -> Result<ReferenceResult, Box<dyn std::error::Error>> {
    match reference_mode {
        ReferenceMode::DistributedDirect => {
            if rank == 0 {
                println!("Reference phase (distributed): preonly + lu");
            }
            let distributed_ref = collective_phase_result(
                "reference (distributed)",
                run_phase(
                    "reference",
                    "preonly",
                    "lu",
                    "left",
                    rtol,
                    atol,
                    dtol,
                    max_iters,
                    restart,
                    rhs,
                    csr_op,
                    comm.clone(),
                    PhaseMode::Distributed,
                    false,
                    rank,
                ),
                comm,
                rank,
            )?;
            let (_, x_ref, _, setup_ref, solve_ref) = distributed_ref;
            Ok(ReferenceResult::Solved {
                x: x_ref,
                setup_time: setup_ref,
                solve_time: solve_ref,
                solver_label: "preonly + lu",
            })
        }
        ReferenceMode::LocalDirectReplicated => collective_phase_result(
            "reference (local replicated)",
            run_local_dense_reference(matrix_data, rhs),
            comm,
            rank,
        ),
        ReferenceMode::Skip => Ok(ReferenceResult::Skipped(
            "no compatible direct backend selected/available.",
        )),
        ReferenceMode::Auto => unreachable!("reference_mode is resolved before this match"),
    }
}

#[cfg(not(feature = "complex"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "mpi")]
    let comm = UniverseComm::Mpi(std::sync::Arc::new(MpiComm::new()));
    #[cfg(not(feature = "mpi"))]
    let comm = UniverseComm::NoComm(kryst::parallel::NoComm);

    let rank = comm.rank();
    let size = comm.size();

    let args: Vec<String> = env::args().collect();
    let cli_reference_mode = parse_reference_mode_arg(&args)?;
    let diagnostic_profile = parse_diagnostic_profile_arg(&args)?;
    let mpi_pc_stage = parse_mpi_pc_stage_arg(&args)?;
    let preprocess_cfg = parse_preprocess_config(&args);
    let experiment_preset = parse_experiment_preset(&args);

    let mut parse_args = Vec::with_capacity(args.len());
    let mut idx = 0usize;
    while idx < args.len() {
        if args[idx] == "--reference-mode" {
            idx += 2;
            continue;
        }
        if args[idx] == "--diagnostic-profile" {
            idx += 2;
            continue;
        }
        if args[idx] == "--experiment-preset" {
            idx += 2;
            continue;
        }
        if args[idx] == "--mpi-pc-stage" {
            idx += 2;
            continue;
        }
        if args[idx] == "--pre-row-scale"
            || args[idx] == "--pre-col-scale"
            || args[idx] == "--pre-match-permute"
        {
            idx += 1;
            continue;
        }
        parse_args.push(args[idx].clone());
        idx += 1;
    }

    let (mut ksp_opts, mut pc_opts) = parse_all_options(&parse_args)?;
    let allow_divergence = args.iter().any(|a| a == "--allow-divergence");

    match mpi_pc_stage {
        MpiPcStageMode::Auto | MpiPcStageMode::LocalDirectRef => {}
        MpiPcStageMode::BlockJacobiLocalLu => {
            pc_opts.pc_global = Some("block_jacobi".into());
            pc_opts.pc_local = Some("ilu".into());
            pc_opts.ilu_level = Some(0);
        }
        MpiPcStageMode::BlockJacobiLocalIlut => {
            pc_opts.pc_global = Some("block_jacobi".into());
            pc_opts.pc_local = Some("ilut".into());
            pc_opts.ilut_drop_tol = Some(pc_opts.ilut_drop_tol.unwrap_or(1e-4));
            pc_opts.ilut_max_fill = Some(pc_opts.ilut_max_fill.unwrap_or(20));
        }
        MpiPcStageMode::AsmLocalLu => {
            pc_opts.pc_global = Some("asm".into());
            pc_opts.asm_block_solver = Some("ludense".into());
            pc_opts.asm_inner_pc = Some("jacobi".into());
        }
        MpiPcStageMode::AsmLocalIlut => {
            pc_opts.pc_global = Some("asm".into());
            pc_opts.asm_block_solver = Some("csr".into());
            pc_opts.asm_inner_pc = Some("ilut".into());
            pc_opts.ilut_drop_tol = Some(pc_opts.ilut_drop_tol.unwrap_or(1e-4));
            pc_opts.ilut_max_fill = Some(pc_opts.ilut_max_fill.unwrap_or(20));
        }
        MpiPcStageMode::FgmresRightSubdomainKsp => {
            ksp_opts.ksp_type = Some("fgmres".into());
            ksp_opts.pc_side = Some("right".into());
            pc_opts.pc_type = Some("ksp".into());
            pc_opts.pc_global = Some("asm".into());
            pc_opts.asm_block_solver = Some("csr".into());
            pc_opts.asm_inner_pc = Some("ilut".into());
            pc_opts.pc_ksp_ksp_type = Some("fgmres".into());
            pc_opts.pc_ksp_pc_side = Some("right".into());
            pc_opts.pc_ksp_pc_type = Some("asm".into());
            pc_opts.pc_ksp_maxits = Some(pc_opts.pc_ksp_maxits.unwrap_or(2));
            pc_opts.pc_ksp_rtol = Some(pc_opts.pc_ksp_rtol.unwrap_or(1e-2));
        }
    }

    let dist_strategy = pc_opts
        .amg_dist_apply_mode
        .as_deref()
        .and_then(|v| DistCoarseStrategy::from_str(v).ok())
        .unwrap_or(DistCoarseStrategy::RootGather);
    let selected_route = pc_opts
        .amg_dist_coarse_solver_route
        .as_deref()
        .and_then(|v| v.split(',').next())
        .and_then(|v| DistCoarseSolverRoute::from_str(v.trim()).ok())
        .unwrap_or(DistCoarseSolverRoute::Auto);
    let fallback_chain = match dist_strategy {
        DistCoarseStrategy::RootGather => vec!["Root", "Local", "SuperLuDist"],
        DistCoarseStrategy::LocalPrototype => vec!["Local", "Root", "SuperLuDist"],
        DistCoarseStrategy::SuperLuDist => vec!["SuperLuDist", "Root", "Local"],
        DistCoarseStrategy::None => vec!["Local", "Root"],
    };

    let rtol = ksp_opts.rtol.unwrap_or(1e-8);
    let atol = ksp_opts.atol.unwrap_or(1e-12);
    let dtol = ksp_opts.dtol.unwrap_or(1e5);

    let matrix_file = ksp_opts
        .matrix_file
        .as_deref()
        .unwrap_or("examples/e05r0300/e05r0300.mtx");
    let rhs_file = ksp_opts
        .rhs_file
        .as_deref()
        .unwrap_or("examples/e05r0300/e05r0300_rhs1.mtx");

    if rank == 0 {
        println!("Kryst MPI Unified KSP Context Demo");
        println!("===================================");
        println!("Running on {} MPI processes", size);
        println!("Configuration:");
        println!("  Relative tolerance: {:.1e}", rtol);
        println!("  Absolute tolerance: {:.1e}", atol);
        println!("  Divergence tolerance: {:.1e}", dtol);
        println!("  Dist coarse route (selected): {:?}", selected_route);
        println!(
            "  Dist coarse fallback chain: {}",
            fallback_chain.join(" -> ")
        );
        println!("  Matrix file: {}", matrix_file);
        println!("  RHS file: {}", rhs_file);
        println!("  Reference mode (CLI): {:?}", cli_reference_mode);
        println!("  Diagnostic profile: {:?}", diagnostic_profile);
        println!(
            "  Preprocess stages: matching_perm={}, row_1_norm_scale={}, col_1_norm_scale={}",
            preprocess_cfg.matching_perm, preprocess_cfg.row_scale, preprocess_cfg.col_scale
        );
        if let Some(preset) = &experiment_preset {
            println!("  Experiment preset: {}", preset);
        }
        println!();
    }

    let start_io = Instant::now();
    let (matrix_data, rhs_data) = if rank == 0 {
        println!("Reading matrix from {}...", matrix_file);
        let matrix_data = read_matrix_market(matrix_file)?;
        println!(
            "Matrix: {}x{} with {} non-zeros",
            matrix_data.rows,
            matrix_data.cols,
            matrix_data.values.len()
        );
        println!("Reading RHS from {}...", rhs_file);
        let rhs_data = read_matrix_market(rhs_file)?;
        println!("RHS: {}x{} vector", rhs_data.rows, rhs_data.cols);
        (matrix_data, rhs_data)
    } else {
        (
            read_matrix_market(matrix_file)?,
            read_matrix_market(rhs_file)?,
        )
    };

    let matrix = matrix_data.to_csr_matrix()?;
    let rhs_orig = rhs_data.to_vector()?;
    let original_csr_op = Arc::new(CsrOp::new(Arc::new(matrix.clone())));

    let before_diag = matrix_diagnostics(&matrix);
    let mut transform = SystemTransform::identity(matrix.nrows(), matrix.ncols());
    let mut transformed_matrix = matrix.clone();
    if preprocess_cfg.matching_perm {
        let (row_perm, col_perm) = nonsymmetric_max_transversal_permutation(&transformed_matrix);
        transformed_matrix = permute_rows_cols(&transformed_matrix, &row_perm, &col_perm);
        transform.row_perm = row_perm;
        transform.row_pinv = build_inverse(&transform.row_perm);
        transform.col_perm = col_perm;
        transform.col_pinv = build_inverse(&transform.col_perm);
    }
    if preprocess_cfg.row_scale {
        let row_scale = row_1_norm_scaling(&transformed_matrix, 1e-15);
        transformed_matrix = scale_rows(&transformed_matrix, &row_scale);
        transform.row_scale = Some(row_scale);
    }
    if preprocess_cfg.col_scale {
        let col_scale = col_1_norm_scaling(&transformed_matrix, 1e-15);
        transformed_matrix = scale_cols(&transformed_matrix, &col_scale);
        transform.col_scale = Some(col_scale);
    }
    let rhs_t = transform.apply_rhs(&rhs_orig);
    let after_diag = matrix_diagnostics(&transformed_matrix);

    let csr_arc = Arc::new(transformed_matrix);
    let csr_op = Arc::new(CsrOp::new(csr_arc.clone()));
    let rhs = rhs_t;

    let io_time = start_io.elapsed();
    let inspection = inspect_matrix(&matrix_data);

    let full_gmres_restart = inspection.nrows.min(256);
    let restart = ksp_opts.restart.unwrap_or(full_gmres_restart);
    let max_iters = ksp_opts.maxits.unwrap_or(restart);

    let user_solver = ksp_opts.ksp_type.is_some();
    let user_pc = pc_opts.pc_type.is_some();
    let user_pc_side = ksp_opts.pc_side.is_some();

    let mut iterative_solver = ksp_opts.ksp_type.as_deref().unwrap_or("gmres").to_string();
    let mut iterative_pc = pc_opts.pc_type.as_deref().unwrap_or("none").to_string();
    let mut iterative_pc_side = if user_pc_side {
        ksp_opts.pc_side.as_deref().unwrap_or("left").to_string()
    } else if iterative_solver.eq_ignore_ascii_case("fgmres") {
        "right".to_string()
    } else {
        "left".to_string()
    };

    if let Some(preset) = experiment_preset.as_deref() {
        match preset {
            "gmres+ilut" => {
                iterative_solver = "gmres".to_string();
                iterative_pc = "ilut".to_string();
                iterative_pc_side = "left".to_string();
            }
            "fgmres(right)+ilutp" => {
                iterative_solver = "fgmres".to_string();
                iterative_pc = "ilutp".to_string();
                iterative_pc_side = "right".to_string();
            }
            _ => {
                return Err(format!(
                    "Unknown --experiment-preset '{}'; expected gmres+ilut or fgmres(right)+ilutp",
                    preset
                )
                .into());
            }
        }
    }

    if iterative_solver == "fgmres" && iterative_pc_side != "right" {
        return Err("Invalid options: fgmres requires -ksp_pc_side right".into());
    }

    if rank == 0 {
        println!("I/O completed in {:.3}s", io_time.as_secs_f64());
        println!("Matrix inspection:");
        println!("  Shape: {}x{}", inspection.nrows, inspection.ncols);
        println!("  Zero diagonal rows: {}", inspection.zero_diagonal_rows);
        println!(
            "  Diagonal dominance violations: {} / {}",
            inspection.diagonal_dominance_violations, inspection.nrows
        );
        println!(
            "  Symmetry estimate: {:.2}% (likely symmetric: {})",
            100.0 * inspection.symmetry_ratio,
            inspection.is_likely_symmetric
        );
        println!(
            "  Diagnostics before preprocessing: diag_nz={}, diag_zero={}, row1(min/med/max)=({:.3e}, {:.3e}, {:.3e})",
            before_diag.diagonal_nonzeros,
            before_diag.diagonal_zeros,
            before_diag.row_norm_1.min,
            before_diag.row_norm_1.median,
            before_diag.row_norm_1.max
        );
        println!(
            "  Diagnostics after preprocessing:  diag_nz={}, diag_zero={}, row1(min/med/max)=({:.3e}, {:.3e}, {:.3e})",
            after_diag.diagonal_nonzeros,
            after_diag.diagonal_zeros,
            after_diag.row_norm_1.min,
            after_diag.row_norm_1.median,
            after_diag.row_norm_1.max
        );
        println!(
            "  Applied transforms: matching_perm={}, row_scale={}, col_scale={}",
            preprocess_cfg.matching_perm, preprocess_cfg.row_scale, preprocess_cfg.col_scale
        );
        println!("  Effective max iterations: {}", max_iters);
        println!("  Effective GMRES restart: {}", restart);
        if inspection.zero_diagonal_rows > 0 && !user_pc {
            println!("  Default iterative PC policy: using 'none' (zero diagonal rows detected).");
        }
        if !inspection.is_likely_symmetric && !user_solver {
            println!("  Default iterative solver policy: using 'gmres' (nonsymmetric estimate).");
        }
        if ksp_opts.restart.is_none() {
            println!(
                "  Default restart policy: using restart=min(nrows,256)={} for small-matrix sanity.",
                restart
            );
        }
        if ksp_opts.maxits.is_none() {
            println!(
                "  Default max-it policy: using max_it=restart={} for full-GMRES-style sanity.",
                max_iters
            );
        }
        println!();
    }

    #[cfg(feature = "logging")]
    {
        env_logger::init();
    }

    let reference_mode = match cli_reference_mode {
        ReferenceMode::Auto => {
            if inspection.nrows > 1000 {
                ReferenceMode::Skip
            } else if cfg!(feature = "superlu_dist") {
                ReferenceMode::DistributedDirect
            } else if cfg!(feature = "dense-direct") {
                ReferenceMode::LocalDirectReplicated
            } else {
                ReferenceMode::Skip
            }
        }
        fixed => fixed,
    };
    let reference_mode = if (preprocess_cfg.matching_perm
        || preprocess_cfg.row_scale
        || preprocess_cfg.col_scale)
        && reference_mode == ReferenceMode::LocalDirectReplicated
    {
        if rank == 0 {
            println!(
                "Preprocessing is enabled; disabling local replicated dense reference mode for consistency."
            );
        }
        ReferenceMode::Skip
    } else {
        reference_mode
    };
    let mut reference_rel_residual = None;
    let mut reference_solution: Option<Vec<f64>> = None;
    let mut reference_setup = Duration::from_secs(0);
    let mut reference_solve = Duration::from_secs(0);
    let reference_status;

    if rank == 0 {
        println!("Resolved reference mode: {:?}", reference_mode);
    }

    match run_reference_phase(
        reference_mode,
        &matrix_data,
        &rhs,
        csr_op.clone(),
        &comm,
        rank,
        rtol,
        atol,
        dtol,
        max_iters,
        restart,
    )? {
        ReferenceResult::Solved {
            x,
            setup_time,
            solve_time,
            solver_label,
        } => {
            reference_status = ReferenceSolveStatus::Succeeded;
            if rank == 0 && reference_mode == ReferenceMode::LocalDirectReplicated {
                println!("Reference phase (local replicated): {solver_label}");
            }
            let ref_rr = compute_relative_residual(csr_op.as_ref(), &x, &rhs, &comm);
            reference_rel_residual = Some(ref_rr);
            reference_solution = Some(x);
            reference_setup = setup_time;
            reference_solve = solve_time;

            if rank == 0 {
                println!("  true relative residual: {:.6e}", ref_rr);
                println!();
            }

            if ref_rr > 1e-8 {
                return Err(format!(
                    "Reference phase failed: true relative residual={:.6e}",
                    ref_rr
                )
                .into());
            }
        }
        ReferenceResult::Skipped(reason) => {
            reference_status = ReferenceSolveStatus::Skipped;
            if rank == 0 {
                println!("Reference phase skipped: {reason}");
                println!();
            }
        }
    }

    if rank == 0 {
        println!(
            "Iterative phase: {} + {} (pc_side={})",
            iterative_solver, iterative_pc, iterative_pc_side
        );
        println!("Iterative sanity checks:");
        println!(
            "  1) local operator GMRES(no PC), restart={}, max_it={}",
            full_gmres_restart, full_gmres_restart
        );
        println!(
            "  2) distributed wrapper GMRES(no PC), restart={}, max_it={}",
            full_gmres_restart, full_gmres_restart
        );
    }

    let (_, local_sanity_x, local_sanity_hist, _, _) = collective_phase_result(
        "iterative-sanity-local",
        run_phase(
            "iterative-sanity-local",
            "gmres",
            "none",
            "left",
            rtol,
            atol,
            dtol,
            full_gmres_restart,
            full_gmres_restart,
            &rhs,
            csr_op.clone(),
            comm.clone(),
            PhaseMode::Local,
            true,
            rank,
        ),
        &comm,
        rank,
    )?;
    let local_sanity_true_rr =
        compute_relative_residual_local(csr_op.as_ref(), &local_sanity_x, &rhs);

    let (_, wrapped_sanity_x, wrapped_sanity_hist, _, _) = collective_phase_result(
        "iterative-sanity-distributed",
        run_phase(
            "iterative-sanity-distributed",
            "gmres",
            "none",
            "left",
            rtol,
            atol,
            dtol,
            full_gmres_restart,
            full_gmres_restart,
            &rhs,
            csr_op.clone(),
            comm.clone(),
            PhaseMode::Distributed,
            true,
            rank,
        ),
        &comm,
        rank,
    )?;
    let wrapped_sanity_true_rr =
        compute_relative_residual(csr_op.as_ref(), &wrapped_sanity_x, &rhs, &comm);

    if rank == 0 {
        let duplicate_count = |hist: &[(usize, f64)]| -> usize {
            hist.windows(2).filter(|w| w[0].0 == w[1].0).count()
        };
        let local_restart_rr = local_sanity_hist
            .iter()
            .find(|(it, _)| *it == full_gmres_restart)
            .map(|(_, rr)| *rr)
            .unwrap_or(
                local_sanity_hist
                    .last()
                    .map(|(_, rr)| *rr)
                    .unwrap_or(f64::NAN),
            );
        let wrapped_restart_rr = wrapped_sanity_hist
            .iter()
            .find(|(it, _)| *it == full_gmres_restart)
            .map(|(_, rr)| *rr)
            .unwrap_or(
                wrapped_sanity_hist
                    .last()
                    .map(|(_, rr)| *rr)
                    .unwrap_or(f64::NAN),
            );
        println!("  Local sanity true residual: {:.6e}", local_sanity_true_rr);
        println!(
            "  Local sanity reported iterations: {}",
            local_sanity_hist.last().map(|(it, _)| *it).unwrap_or(0)
        );
        println!(
            "  Local sanity restart-boundary internal residual: {:.6e}",
            local_restart_rr
        );
        println!(
            "  Local sanity |true-internal| mismatch: {:.6e}",
            (local_sanity_true_rr - local_restart_rr).abs()
        );
        println!(
            "  Local sanity duplicate monitor iteration IDs: {}",
            duplicate_count(&local_sanity_hist)
        );
        println!(
            "  Wrapped sanity true residual: {:.6e}",
            wrapped_sanity_true_rr
        );
        println!(
            "  Wrapped sanity reported iterations: {}",
            wrapped_sanity_hist.last().map(|(it, _)| *it).unwrap_or(0)
        );
        println!(
            "  Wrapped sanity restart-boundary internal residual: {:.6e}",
            wrapped_restart_rr
        );
        println!(
            "  Wrapped sanity |true-internal| mismatch: {:.6e}",
            (wrapped_sanity_true_rr - wrapped_restart_rr).abs()
        );
        println!(
            "  Wrapped sanity duplicate monitor iteration IDs: {}",
            duplicate_count(&wrapped_sanity_hist)
        );
        println!();
    }

    if local_sanity_hist
        .last()
        .map(|(it, _)| *it > full_gmres_restart)
        .unwrap_or(false)
    {
        return Err(format!(
            "Invariant violation in iterative-sanity-local: monitor iter exceeded max_it={} (last={})",
            full_gmres_restart,
            local_sanity_hist.last().map(|(it, _)| *it).unwrap_or(0)
        )
        .into());
    }
    let sanity_pass = local_sanity_true_rr < 1e-6 && wrapped_sanity_true_rr < 1e-6;
    let sanity_status = if sanity_pass {
        IterativeSanityGateStatus::Passed
    } else {
        IterativeSanityGateStatus::ExperimentalFailed
    };

    let (stats, solution, convergence_history, setup_time, solve_time) = collective_phase_result(
        "iterative",
        run_phase(
            "iterative",
            &iterative_solver,
            &iterative_pc,
            &iterative_pc_side,
            rtol,
            atol,
            dtol,
            max_iters,
            restart,
            &rhs,
            csr_op.clone(),
            comm.clone(),
            PhaseMode::Distributed,
            false,
            rank,
        ),
        &comm,
        rank,
    )?;

    let iterative_relative_residual =
        compute_relative_residual(csr_op.as_ref(), &solution, &rhs, &comm);
    let iterative_monitor_residual = convergence_history.last().map(|(_, r)| *r);
    let use_reference_output = !sanity_pass && reference_solution.is_some();
    let output_solution_transformed: &[f64] = if use_reference_output {
        reference_solution
            .as_deref()
            .expect("reference solution available when fallback is enabled")
    } else {
        &solution
    };
    let output_relative_residual =
        compute_relative_residual(csr_op.as_ref(), output_solution_transformed, &rhs, &comm);
    let output_solution_original = transform.recover_solution(output_solution_transformed);
    let output_relative_residual_original = compute_relative_residual(
        original_csr_op.as_ref(),
        &output_solution_original,
        &rhs_orig,
        &comm,
    );

    if rank == 0 {
        println!();
        println!("Iterative phase completed");
        println!("=========================");
        println!("Solve time: {:.3}s", solve_time.as_secs_f64());
        println!("Iterations: {}", stats.iterations);
        println!(
            "Final true residual from solver stats: {:.2e}",
            stats.final_residual
        );
        if let Some((_, last_monitor_residual)) = convergence_history.last() {
            println!("Last monitor residual: {:.2e}", last_monitor_residual);
        }
        println!("Convergence reason: {:?}", stats.reason);
        println!(
            "True relative residual ||b-Ax||/||b|| (iterative solution): {:.2e}",
            iterative_relative_residual
        );
        if use_reference_output {
            println!(
                "True relative residual ||b-Ax||/||b|| (written output solution): {:.2e}",
                output_relative_residual
            );
        }
        println!(
            "True relative residual on original system (after inverse transform): {:.2e}",
            output_relative_residual_original
        );

        if let Some(rr_ref) = reference_rel_residual {
            println!(
                "Residual ratio versus reference: {:.2e}",
                iterative_relative_residual / rr_ref.max(1e-30)
            );
        }

        if !convergence_history.is_empty() {
            let initial_residual = convergence_history[0].1;
            let final_residual = convergence_history
                .last()
                .map(|v| v.1)
                .unwrap_or(initial_residual);
            println!("Convergence analysis:");
            println!("  Initial internal residual: {:.2e}", initial_residual);
            println!("  Final internal residual: {:.2e}", final_residual);
            println!(
                "  Reduction factor: {:.2e}",
                final_residual / initial_residual.max(1e-30)
            );
        }

        println!();
        println!("Residual summary table:");
        println!("  phase                      monitor_tag      monitor(last)    true(||b-Ax||)");
        println!(
            "  {:<26} {:<16} {:>14.6e} {:>14.6e}",
            "iterative",
            monitor_semantics_tag(&iterative_solver, &iterative_pc_side),
            iterative_monitor_residual.unwrap_or(f64::NAN),
            stats.final_residual,
        );

        let total_time = io_time + reference_setup + reference_solve + setup_time + solve_time;
        println!();
        println!("Performance breakdown:");
        println!(
            "  I/O time:    {:.3}s ({:.1}%)",
            io_time.as_secs_f64(),
            100.0 * io_time.as_secs_f64() / total_time.as_secs_f64()
        );
        println!(
            "  Setup time:  {:.3}s ({:.1}%)",
            (reference_setup + setup_time).as_secs_f64(),
            100.0 * (reference_setup + setup_time).as_secs_f64() / total_time.as_secs_f64()
        );
        println!(
            "  Solve time:  {:.3}s ({:.1}%)",
            (reference_solve + solve_time).as_secs_f64(),
            100.0 * (reference_solve + solve_time).as_secs_f64() / total_time.as_secs_f64()
        );
        println!("  Total time:  {:.3}s", total_time.as_secs_f64());
        println!();
    }

    if diagnostic_profile == DiagnosticProfile::RightKrylovResidualAudit {
        if rank == 0 {
            println!();
            println!(
                "Running diagnostic profile: forced right-preconditioned GMRES/FGMRES residual audit"
            );
        }
        let mut diagnostic_rows: Vec<(&str, SolveStats<f64>, Option<f64>)> = Vec::new();
        for solver_name in ["gmres", "fgmres"] {
            let (diag_stats, _diag_solution, diag_history, _, _) = collective_phase_result(
                "diagnostic-right-krylov",
                run_phase(
                    &format!("diagnostic-{solver_name}-right"),
                    solver_name,
                    &iterative_pc,
                    "right",
                    rtol,
                    atol,
                    dtol,
                    max_iters,
                    restart,
                    &rhs,
                    csr_op.clone(),
                    comm.clone(),
                    PhaseMode::Distributed,
                    false,
                    rank,
                ),
                &comm,
                rank,
            )?;
            diagnostic_rows.push((
                solver_name,
                diag_stats,
                diag_history.last().map(|(_, r)| *r),
            ));
        }
        if rank == 0 {
            println!("Residual summary table (diagnostic profile):");
            println!(
                "  phase                      monitor_tag      monitor(last)    true(||b-Ax||)"
            );
            for (solver_name, diag_stats, monitor_last) in diagnostic_rows {
                println!(
                    "  {:<26} {:<16} {:>14.6e} {:>14.6e}",
                    format!("diagnostic-{solver_name}-right"),
                    monitor_semantics_tag(solver_name, "right"),
                    monitor_last.unwrap_or(f64::NAN),
                    diag_stats.final_residual,
                );
            }
        }
    }

    if rank == 0 {
        println!("Writing solution to mpi_ksp_solution.mtx...");
        if use_reference_output {
            println!(
                "  Using direct reference solution for output because iterative sanity checks failed."
            );
        }
        write_vector_market("mpi_ksp_solution.mtx", &output_solution_original)?;
    }

    let output_status = if output_relative_residual_original < 1e-6 {
        WrittenOutputVerificationStatus::Verified
    } else if output_relative_residual_original < 1e-3 {
        WrittenOutputVerificationStatus::Marginal
    } else {
        WrittenOutputVerificationStatus::Failed
    };

    if rank == 0 {
        println!("Verification (true residual):");
        println!(
            "  Relative residual: {:.6e}",
            output_relative_residual_original
        );
        match output_status {
            WrittenOutputVerificationStatus::Verified => {
                println!("✓ written output verified successfully");
            }
            WrittenOutputVerificationStatus::Marginal => {
                println!("⚠ written output is only marginally acceptable on this configuration.");
            }
            WrittenOutputVerificationStatus::Failed => {
                println!("❌ written-output verification failed.");
            }
        }

        println!("Status summary:");
        for line in format_phase_status_summary(reference_status, output_status, sanity_status) {
            println!("  - {line}");
        }
        if sanity_status == IterativeSanityGateStatus::ExperimentalFailed {
            println!(
                "⚠ iterative phase failed sanity gate (experimental path); output may still be valid when backed by reference."
            );
        }
    }

    let iterative_path_success =
        stats.reason.is_converged() && output_status == WrittenOutputVerificationStatus::Verified;
    let overall_success = (reference_status == ReferenceSolveStatus::Succeeded
        && output_status == WrittenOutputVerificationStatus::Verified)
        || (reference_status == ReferenceSolveStatus::Skipped && iterative_path_success);
    if !overall_success && !allow_divergence {
        return Err(format!(
            "Run failed overall (reference_solve_status={:?}, written_output_status={:?}, iterative_sanity_status={:?}, iterative_reason={:?}, original_relative_residual={:.6e}); rerun with --allow-divergence to keep this as an experiment.",
            reference_status,
            output_status,
            sanity_status,
            stats.reason,
            output_relative_residual_original
        )
        .into());
    }

    comm.barrier();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        IterativeSanityGateStatus, ReferenceSolveStatus, WrittenOutputVerificationStatus,
        format_phase_status_summary,
    };

    #[test]
    fn phase_status_summary_uses_required_output_strings() {
        let lines = format_phase_status_summary(
            ReferenceSolveStatus::Succeeded,
            WrittenOutputVerificationStatus::Verified,
            IterativeSanityGateStatus::ExperimentalFailed,
        );
        assert!(lines.contains(&"reference solve status: succeeded".to_string()));
        assert!(lines.contains(
            &"written-output verification status: written output verified successfully".to_string()
        ));
        assert!(lines.contains(
            &"iterative sanity-gate status: iterative phase failed sanity gate".to_string()
        ));
    }
}
