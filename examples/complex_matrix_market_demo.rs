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
    use std::path::Path;
    use std::sync::Arc;
    use std::time::Instant;

    use super::KError;
    use kryst::algebra::blas::nrm2;
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
                println!("Global DOFs: {}", problem.global_n);
                println!("Local DOFs (rank {rank}): {}", problem.local_n);
                println!("‖rhs‖₂ = {:.3e}", rhs_norm);
                println!(
                    "{:<36} {:>8} {:>12} {:>10} {:>12} {:>10}",
                    "Method", "Iters", "Residual", "Time(s)", "Reductions", "Status"
                );
                println!("{}", "-".repeat(96));
            }

            let runs = [
                RunSpec::fgmres_jacobi_right(50),
                RunSpec::fgmres_none_right(50),
            ];

            for spec in runs {
                match run_once(&problem, &spec) {
                    Ok(row) => {
                        if rank == 0 {
                            let status = if row.converged { "✓" } else { "✗" };
                            println!(
                                "{:<36} {:>8} {:>12.2e} {:>10.3} {:>12} {:>10}",
                                row.method,
                                row.iterations,
                                row.residual,
                                row.time_secs,
                                row.reductions,
                                status
                            );
                            if row.converged {
                                println!(
                                    "    → {:.2e} DOF/s (reason: {:?})",
                                    row.dof_per_sec, row.reason
                                );
                            } else {
                                println!("    → stopped with {:?}", row.reason);
                            }
                        }
                    }
                    Err(err) => {
                        if rank == 0 {
                            println!(
                                "{:<36} {:>8} {:>12} {:>10} {:>12} {:>10}",
                                spec.name, "FAIL", "N/A", "N/A", "N/A", "✗"
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
        backend_descr: String,
    }

    struct ResultRow {
        method: String,
        iterations: usize,
        residual: R,
        time_secs: f64,
        converged: bool,
        reductions: usize,
        reason: ConvergedReason,
        dof_per_sec: f64,
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

    fn run_once(problem: &Problem, spec: &RunSpec) -> Result<ResultRow, KError> {
        let mut solver = FgmresSolver::new(1e-8, 500, spec.restart);
        solver.variant = FgmresVariant::Pipelined;
        solver.reorth = ReorthPolicy::IfNeeded;
        solver.atol = 1e-12;
        solver.dtol = 1e6;

        let mut workspace = Workspace::new(problem.local_n);
        solver.setup_workspace(&mut workspace);

        let mut x = vec![S::zero(); problem.local_n];
        let b = &problem.rhs;

        let mut jacobi_pc: Option<Jacobi> = None;
        if matches!(spec.pc, PcKind::Jacobi) {
            let mut pc = Jacobi::new();
            pc.setup(problem.csr_for_pc.as_ref())?;
            jacobi_pc = Some(pc);
        }
        let pc_opt = jacobi_pc
            .as_mut()
            .map(|pc| pc as &mut dyn KPreconditioner<Scalar = S>);

        problem.comm.barrier();
        let start = Instant::now();
        let stats = solver.solve_k(
            problem.op.as_ref(),
            pc_opt,
            b,
            &mut x,
            spec.pc_side,
            &problem.comm,
            None,
            Some(&mut workspace),
        )?;
        problem.comm.barrier();
        let elapsed = start.elapsed().as_secs_f64();

        let converged = matches!(
            stats.reason,
            ConvergedReason::ConvergedAtol
                | ConvergedReason::ConvergedRtol
                | ConvergedReason::ConvergedHappyBreakdown
                | ConvergedReason::ConvergedTrustRegion
        );

        let reductions = stats.counters.num_global_reductions;
        let dof_per_sec = if elapsed > 0.0 {
            problem.global_n as f64 / elapsed
        } else {
            0.0
        };

        Ok(ResultRow {
            method: format!("{} (m = {})", spec.name, spec.restart),
            iterations: stats.iterations,
            residual: stats.final_residual,
            time_secs: elapsed,
            converged,
            reductions,
            reason: stats.reason,
            dof_per_sec,
        })
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

        Ok(Problem {
            op: op_arc,
            rhs,
            csr_for_pc,
            local_n: nrows,
            global_n: nrows,
            comm: comm.clone(),
            backend_descr: if comm.size() > 1 {
                "Generic CSR (complex, replicated)".into()
            } else {
                "Generic CSR (complex, serial)".into()
            },
        })
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
