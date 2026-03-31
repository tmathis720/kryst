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
//!   --allow-divergence         Return success even if iterative phase fails verification

#[cfg(feature = "complex")]
fn main() {
    eprintln!("mpi_amg_gmres_demo.rs is unavailable when built with --features complex");
}

use kryst::config::options::parse_all_options;
use kryst::context::ksp_context::KspContext;
use kryst::matrix::op::{CsrOp, LinOp, wrap_with_comm};
use kryst::parallel::{Comm, UniverseComm};
use kryst::solver::MonitorAction;
use kryst::utils::matrix_market::{MatrixMarketData, read_matrix_market, write_vector_market};
use std::collections::HashMap;
use std::env;
use std::str::FromStr;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[cfg(feature = "mpi")]
use kryst::parallel::MpiComm;
use kryst::preconditioner::dist::{DistCoarseSolverRoute, DistCoarseStrategy};
use kryst::utils::convergence::SolveStats;

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

    let monitor_data = Arc::new(Mutex::new(Vec::<(usize, f64)>::new()));
    let monitor_data_clone = monitor_data.clone();
    let phase_name_owned = phase_name.to_string();
    let monitor = Box::new(move |iter: usize, residual: f64, _reductions: usize| {
        if let Ok(mut data) = monitor_data_clone.lock() {
            data.push((iter, residual));
            if rank == 0 && (iter == 0 || iter % 200 == 0 || residual < rtol) {
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
    let op_arc = wrap_with_comm(base_op.clone(), comm.clone());
    ksp.set_operators_with_comm(op_arc, None, comm.clone());
    ksp.setup()?;
    let setup_time = start_setup.elapsed();

    comm.barrier();
    let mut solution = vec![0.0; rhs.len()];
    let start_solve = Instant::now();
    let stats = ksp.solve(rhs, &mut solution)?;
    let solve_time = start_solve.elapsed();

    if stats.iterations > max_iters {
        return Err(format!(
            "Invariant violation in {phase_name}: iterations {} exceed configured max {}",
            stats.iterations, max_iters
        )
        .into());
    }

    let history = monitor_data
        .lock()
        .map(|data| data.clone())
        .unwrap_or_default();
    Ok((stats, solution, history, setup_time, solve_time))
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
    let (ksp_opts, pc_opts) = parse_all_options(&args)?;
    let allow_divergence = args.iter().any(|a| a == "--allow-divergence");

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
    let max_iters = ksp_opts.maxits.unwrap_or(2000);
    let restart = ksp_opts.restart.unwrap_or(200);

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
        println!("  Max iterations: {}", max_iters);
        println!("  GMRES restart: {}", restart);
        println!("  Dist coarse route (selected): {:?}", selected_route);
        println!(
            "  Dist coarse fallback chain: {}",
            fallback_chain.join(" -> ")
        );
        println!("  Matrix file: {}", matrix_file);
        println!("  RHS file: {}", rhs_file);
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
    let rhs = rhs_data.to_vector()?;
    let csr_arc = Arc::new(matrix);
    let csr_op = Arc::new(CsrOp::new(csr_arc.clone()));

    let io_time = start_io.elapsed();
    let inspection = inspect_matrix(&matrix_data);

    let user_solver = ksp_opts.ksp_type.is_some();
    let user_pc = pc_opts.pc_type.is_some();
    let user_pc_side = ksp_opts.pc_side.is_some();

    let iterative_solver = ksp_opts.ksp_type.as_deref().unwrap_or("gmres").to_string();
    let iterative_pc = pc_opts.pc_type.as_deref().unwrap_or("none").to_string();
    let iterative_pc_side = if user_pc_side {
        ksp_opts.pc_side.as_deref().unwrap_or("left").to_string()
    } else if iterative_solver == "fgmres" {
        "right".to_string()
    } else {
        "left".to_string()
    };

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
        if inspection.zero_diagonal_rows > 0 && !user_pc {
            println!("  Default iterative PC policy: using 'none' (zero diagonal rows detected).");
        }
        if !inspection.is_likely_symmetric && !user_solver {
            println!("  Default iterative solver policy: using 'gmres' (nonsymmetric estimate).");
        }
        println!();
    }

    #[cfg(feature = "logging")]
    {
        env_logger::init();
    }

    let run_reference = inspection.nrows <= 1000;
    let mut reference_rel_residual = None;
    let mut reference_setup = Duration::from_secs(0);
    let mut reference_solve = Duration::from_secs(0);

    if run_reference {
        if rank == 0 {
            println!("Reference phase: preonly + lu");
        }
        let (stats_ref, x_ref, _, setup_ref, solve_ref) = run_phase(
            "reference",
            "preonly",
            "lu",
            "left",
            rtol,
            atol,
            dtol,
            max_iters,
            restart,
            &rhs,
            csr_op.clone(),
            comm.clone(),
            rank,
        )?;
        let ref_rr = compute_relative_residual(csr_op.as_ref(), &x_ref, &rhs, &comm);
        reference_rel_residual = Some(ref_rr);
        reference_setup = setup_ref;
        reference_solve = solve_ref;

        if rank == 0 {
            println!(
                "  reason: {:?} (converged: {}), iterations: {}",
                stats_ref.reason,
                stats_ref.reason.is_converged(),
                stats_ref.iterations
            );
            println!("  true relative residual: {:.6e}", ref_rr);
            println!();
        }

        if !stats_ref.reason.is_converged() || ref_rr > 1e-8 {
            return Err(format!(
                "Reference phase failed: reason={:?}, true relative residual={:.6e}",
                stats_ref.reason, ref_rr
            )
            .into());
        }
    }

    if rank == 0 {
        println!(
            "Iterative phase: {} + {} (pc_side={})",
            iterative_solver, iterative_pc, iterative_pc_side
        );
    }

    let (stats, solution, convergence_history, setup_time, solve_time) = run_phase(
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
        rank,
    )?;

    let relative_residual = compute_relative_residual(csr_op.as_ref(), &solution, &rhs, &comm);

    if rank == 0 {
        println!();
        println!("Iterative phase completed");
        println!("=========================");
        println!("Solve time: {:.3}s", solve_time.as_secs_f64());
        println!("Iterations: {}", stats.iterations);
        println!("Final internal residual: {:.2e}", stats.final_residual);
        println!("Convergence reason: {:?}", stats.reason);
        println!(
            "True relative residual ||b-Ax||/||b||: {:.2e}",
            relative_residual
        );

        if let Some(rr_ref) = reference_rel_residual {
            println!(
                "Residual ratio versus reference: {:.2e}",
                relative_residual / rr_ref.max(1e-30)
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

    if rank == 0 {
        println!("Writing solution to mpi_ksp_solution.mtx...");
        write_vector_market("mpi_ksp_solution.mtx", &solution)?;
    }

    if rank == 0 {
        println!("Verification (true residual):");
        println!("  Relative residual: {:.6e}", relative_residual);
        if stats.reason.is_converged() && relative_residual < 1e-6 {
            println!("✓ Iterative solution verified successfully.");
        } else if relative_residual < 1e-3 {
            println!("⚠ Iterative solve is only marginally acceptable on this configuration.");
        } else {
            println!("❌ Iterative solution verification failed.");
        }
    }

    let iterative_success = stats.reason.is_converged() && relative_residual < 1e-6;
    if !iterative_success && !allow_divergence {
        return Err(format!(
            "Iterative phase failed (reason={:?}, relative_residual={:.6e}); rerun with --allow-divergence to keep this as an experiment.",
            stats.reason, relative_residual
        )
        .into());
    }

    comm.barrier();
    Ok(())
}
