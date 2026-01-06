#[cfg(feature = "complex")]
fn main() {
    eprintln!("matrix_market_demo.rs is unavailable when built with --features complex");
}

#[cfg(not(feature = "backend-faer"))]
#[cfg(not(feature = "complex"))]
fn main() {
    eprintln!("matrix_market_demo requires the backend-faer feature.");
}

#[cfg(all(feature = "backend-faer", feature = "complex"))]
#[cfg(not(feature = "complex"))]
fn main() {
    eprintln!(
        "matrix_market_demo is only available for real-valued builds.\n\
         Re-run without the `complex` feature (try `--no-default-features`)."
    );
}

#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
mod real_demo {
    //! Unified Matrix Market demo covering serial and MPI runs with multiple solver/preconditioner
    //! combinations. The example keeps the solver boundary matrix-free and reports iteration metrics
    //! that are relevant for driven-cavity style matrices.
    //!
    //! to run:
    //! cargo mpirun -n 2 --features=mpi,rayon,simd,dense-direct,faer-backend --example matrix_market_demo
    //! or for serial:
    //! cargo run --example matrix_market_demo
    //! (ensure the example is built with the "mpi" feature for MPI runs)

    use std::path::Path;
    use std::sync::Arc;
    use std::time::Instant;
    use std::env;

    use faer::Mat;

    use kryst::config::options::{KspOptions, PcOptions};
    use kryst::context::ksp_context::{KspContext, SolverType};
    use kryst::context::pc_context::PcType;
    use kryst::error::KError;
    #[cfg(feature = "mpi")]
    use kryst::matrix::DistCsrOp;
    use kryst::matrix::op::{CsrOp, LinOp};
    #[cfg(feature = "mpi")]
    use kryst::matrix::parcsr::builder::partition_rows;
    use kryst::matrix::sparse::CsrMatrix;
    #[cfg(not(feature = "mpi"))]
    use kryst::parallel::NoComm;
    use kryst::parallel::{Comm, UniverseComm};
    use kryst::preconditioner::amg::{AMG, CoarsenType, InterpType, RelaxType};
    use kryst::preconditioner::{PcSide, Preconditioner};
    use kryst::utils::convergence::ConvergedReason;
    use kryst::utils::matrix_market::read_matrix_market;

    #[cfg(feature = "mpi")]
    use kryst::parallel::MpiComm;

    type PcBuilder = Arc<dyn Fn() -> Result<Box<dyn Preconditioner>, KError> + Send + Sync>;
    type SolverConfigurator = Arc<dyn Fn(&mut KspContext) -> Result<(), KError> + Send + Sync>;

    #[derive(Clone, Copy, Debug, Default)]
    struct Analysis {
        nnz: usize,
        density: f64,
        approx_symmetric: bool,
        has_diag_zeros: bool,
    }

    struct Problem {
        op: Arc<dyn LinOp<S = f64>>,
        rhs: Vec<f64>,
        local_n: usize,
        global_n: usize,
        comm: UniverseComm,
        backend_descr: String,
    }

    struct ResultRow {
        method: String,
        iterations: usize,
        residual: f64,
        time_secs: f64,
        converged: bool,
        reductions: usize,
        reason: ConvergedReason,
        dof_per_sec: f64,
    }

    enum PcConfigSpec {
        Type {
            pc_type: PcType,
            options: Option<PcOptions>,
        },
        Builder(PcBuilder),
    }

    struct RunSpec {
        name: &'static str,
        solver: SolverType,
        pc_side: PcSide,
        pc: PcConfigSpec,
        setup: Option<SolverConfigurator>,
    }

    struct MenuPlan {
        specs: Vec<RunSpec>,
        notes: Vec<&'static str>,
    }

    fn is_comm_parallel(comm: &UniverseComm, size: usize) -> bool {
        #[cfg(feature = "mpi")]
        {
            matches!(comm, UniverseComm::Mpi(_)) && size > 1
        }
        #[cfg(not(feature = "mpi"))]
        {
            let _ = comm;
            let _ = size;
            false
        }
    }

    pub fn run() -> Result<(), KError> {
        #[cfg(feature = "logging")]
        let _ = env_logger::try_init();

        #[cfg(feature = "mpi")]
        let comm = UniverseComm::Mpi(Arc::new(MpiComm::new()));
        #[cfg(not(feature = "mpi"))]
        let comm = UniverseComm::NoComm(NoComm);

        let rank = comm.rank();
        let size = comm.size();
        let is_parallel = is_comm_parallel(&comm, size);

        if rank == 0 {
            println!("Matrix Market solver/preconditioner comparison (matrix-free)");
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

        let base = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples");
        let cases = [
            (
                "e05r0000/e05r0000.mtx",
                "e05r0000/e05r0000_rhs1.mtx",
                "Driven cavity (Re = 0)",
            ),
            (
                "e05r0300/e05r0300.mtx",
                "e05r0300/e05r0300_rhs1.mtx",
                "Driven cavity (Re = 300)",
            ),
            (
                "e30r0000/e30r0000.mtx",
                "e30r0000/e30r0000_rhs1.mtx",
                "Driven cavity 30x30 (Re = 0)",
            ),
            (
                "e30r1000/e30r1000.mtx",
                "e30r1000/e30r1000_rhs1.mtx",
                "Driven cavity 30x30 (Re = 1000)",
            ),
        ];

        for (mat_rel, rhs_rel, descr) in cases {
            let mat_path = base.join(mat_rel);
            let rhs_path = base.join(rhs_rel);

            #[cfg_attr(not(feature = "mpi"), allow(unused_mut))]
            let mut available = mat_path.exists() && rhs_path.exists();
            if is_parallel {
                #[cfg(feature = "mpi")]
                {
                    if let Some(mpi_comm) = as_mpi(&comm) {
                        available = broadcast_bool(available, &mpi_comm);
                    }
                }
            }

            if !available {
                if rank == 0 {
                    println!("⚠ Missing files for {descr}, skipping.");
                    println!();
                }
                continue;
            }

            let (problem, analysis) = match load_and_distribute(&mat_path, &rhs_path, &comm) {
                Ok(res) => res,
                Err(e) => {
                    if rank == 0 {
                        println!("❌ Failed to load {descr}: {e}");
                        println!();
                    }
                    continue;
                }
            };

            let rhs_norm_sq_local: f64 = problem.rhs.iter().map(|x| x * x).sum();
            let rhs_norm = problem.comm.all_reduce_f64(rhs_norm_sq_local).sqrt();

            if rank == 0 {
                println!("=== {descr} — {} ===", problem.backend_descr);
                println!("Global DOFs: {}", problem.global_n);
                println!("Local DOFs (rank {rank}): {}", problem.local_n);
                println!(
                    "Nonzeros: {} (density {:.3e})",
                    analysis.nnz, analysis.density
                );
                println!(
                    "Approx. symmetric: {} | diagonal issues: {}",
                    yes_no(analysis.approx_symmetric),
                    yes_no(analysis.has_diag_zeros)
                );
                println!("‖rhs‖₂ = {:.3e}", rhs_norm);
                println!(
                    "{:<34} {:>8} {:>12} {:>10} {:>12} {:>10}",
                    "Method", "Iters", "Residual", "Time(s)", "Reductions", "Status"
                );
                println!("{}", "-".repeat(92));
            }

            let menu_plan = build_menu(&analysis, is_parallel);
            if rank == 0 {
                for note in &menu_plan.notes {
                    println!("ℹ {note}");
                }
            }

            for spec in &menu_plan.specs {
                match run_once(&problem, spec) {
                    Ok(row) => {
                        if rank == 0 {
                            let status = if row.converged { "✓" } else { "✗" };
                            println!(
                                "{:<34} {:>8} {:>12.2e} {:>10.3} {:>12} {:>10}",
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
                    Err(e) => {
                        if rank == 0 {
                            println!(
                                "{:<34} {:>8} {:>12} {:>10} {:>12} {:>10}",
                                spec.name, "FAIL", "N/A", "N/A", "N/A", "✗"
                            );
                            println!("    → {}", e);
                        }
                    }
                }
                problem.comm.barrier();
            }

            if rank == 0 {
                println!("{}", "=".repeat(92));
                println!();
            }
            problem.comm.barrier();
        }

        if rank == 0 {
            println!("Example complete.");
        }

        Ok(())
    }

    fn yes_no(flag: bool) -> &'static str {
        if flag { "yes" } else { "no" }
    }

    fn run_once(problem: &Problem, spec: &RunSpec) -> Result<ResultRow, KError> {
        let mut ksp = KspContext::new();
        ksp.set_type(spec.solver)?;
        ksp.try_set_pc_side(spec.pc_side)?;

        match &spec.pc {
            PcConfigSpec::Type { pc_type, options } => {
                let opts_ref = options.as_ref();
                ksp.set_pc_type(*pc_type, opts_ref)?;
            }
            PcConfigSpec::Builder(builder) => {
                let pc = builder()?;
                ksp.set_pc_box_for_tests(pc);
            }
        }

        if let Some(hook) = &spec.setup {
            hook(&mut ksp)?;
        }

        ksp.set_tolerances(1e-8, 1e-12, 1e8, 1000);
        ksp.set_operators(problem.op.clone(), None);

        problem.comm.barrier();
        let mut x = vec![0.0; problem.local_n];
        let start = Instant::now();
        let stats = ksp.solve(&problem.rhs, &mut x)?;
        problem.comm.barrier();
        let elapsed = start.elapsed().as_secs_f64();
        let true_residual = true_residual_norm(problem, &x)?;

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
            method: spec.name.to_string(),
            iterations: stats.iterations,
            residual: true_residual,
            time_secs: elapsed,
            converged,
            reductions,
            reason: stats.reason,
            dof_per_sec,
        })
    }

    fn build_menu(analysis: &Analysis, is_parallel: bool) -> MenuPlan {
        let mut specs = Vec::new();
        let mut notes = Vec::new();

        let enable_stress_solvers = stress_solvers_enabled();

        if is_parallel {
            notes.push(
                "MPI runs use local diagonal blocks for ILUT/AMG preconditioners (block-Jacobi style).",
            );
        }
        notes.push("FGMRES entries are right-preconditioned in this demo.");
        if !enable_stress_solvers {
            notes.push("Set KRYST_ENABLE_STRESS_SOLVERS=1 to include TFQMR/BiCGStab runs.");
        }

        specs.push(RunSpec {
            name: "FGMRES(50) + ILUT (R)",
            solver: SolverType::Fgmres,
            pc_side: PcSide::Right,
            pc: PcConfigSpec::Type {
                pc_type: PcType::Ilut,
                options: Some(ilut_options()),
            },
            setup: Some(fgmres_hook()),
        });

        if !analysis.has_diag_zeros {
            specs.push(RunSpec {
                name: "FGMRES(50) + AMG (R)",
                solver: SolverType::Fgmres,
                pc_side: PcSide::Right,
                pc: PcConfigSpec::Builder(amg_builder(false)),
                setup: Some(fgmres_hook()),
            });
        } else {
            notes.push("AMG disabled due to near-zero or missing diagonal entries.");
        }

        if analysis.approx_symmetric && !analysis.has_diag_zeros {
            specs.push(RunSpec {
                name: "PCG(pipelined) + AMG (L)",
                solver: SolverType::Pcg,
                pc_side: PcSide::Left,
                pc: PcConfigSpec::Builder(amg_builder(true)),
                setup: Some(pcg_pipelined_hook()),
            });
            specs.push(RunSpec {
                name: "PCG + Jacobi (L)",
                solver: SolverType::Pcg,
                pc_side: PcSide::Left,
                pc: PcConfigSpec::Type {
                    pc_type: PcType::Jacobi,
                    options: None,
                },
                setup: None,
            });
        }

        if enable_stress_solvers {
            notes.push(
                "TFQMR/BiCGStab entries are stress tests for weak preconditioners or indefinite systems.",
            );
            specs.push(RunSpec {
                name: "TFQMR + ILUT (L)",
                solver: SolverType::Tfqmr,
                pc_side: PcSide::Left,
                pc: PcConfigSpec::Type {
                    pc_type: PcType::Ilut,
                    options: Some(ilut_options()),
                },
                setup: None,
            });

            specs.push(RunSpec {
                name: "BiCGStab + ILUT (L)",
                solver: SolverType::BiCgStab,
                pc_side: PcSide::Left,
                pc: PcConfigSpec::Type {
                    pc_type: PcType::Ilut,
                    options: Some(ilut_options()),
                },
                setup: None,
            });
        }

        if !is_parallel {
            if cfg!(feature = "dense-direct") {
                specs.push(RunSpec {
                    name: "PREONLY + Dense LU",
                    solver: SolverType::Preonly,
                    pc_side: PcSide::Left,
                    pc: PcConfigSpec::Type {
                        pc_type: PcType::Lu,
                        options: None,
                    },
                    setup: None,
                });
            } else {
                notes.push("Dense LU requires the dense-direct feature.");
            }

            #[cfg(feature = "superlu_dist")]
            {
                specs.push(RunSpec {
                    name: "PREONLY + SuperLU_DIST",
                    solver: SolverType::Preonly,
                    pc_side: PcSide::Left,
                    pc: PcConfigSpec::Type {
                        pc_type: PcType::SuperLuDist,
                        options: None,
                    },
                    setup: None,
                });
            }
        } else {
            notes.push("Direct solvers are skipped for MPI runs (require global matrices).");
        }

        MenuPlan { specs, notes }
    }

    fn stress_solvers_enabled() -> bool {
        match env::var("KRYST_ENABLE_STRESS_SOLVERS") {
            Ok(value) => matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            ),
            Err(_) => false,
        }
    }

    fn ilut_options() -> PcOptions {
        let mut opts = PcOptions::default();
        opts.ilut_drop_tol = Some(1e-4);
        opts.ilut_max_fill = Some(50);
        opts.ilu_reordering = Some("amd".into());
        opts
    }

    fn pcg_pipelined_hook() -> SolverConfigurator {
        Arc::new(|ksp: &mut KspContext| {
            let mut opts = KspOptions::default();
            opts.cg_pipelined = Some(true);
            opts.cg_replace_every = Some(50);
            ksp.set_from_options(&opts)?;
            Ok(())
        })
    }

    fn fgmres_hook() -> SolverConfigurator {
        Arc::new(|ksp: &mut KspContext| {
            ksp.set_restart(50);
            let mut opts = KspOptions::default();
            opts.fgmres_variant = Some("pipelined".into());
            opts.fgmres_reorth = Some("ifneeded".into());
            ksp.set_from_options(&opts)?;
            Ok(())
        })
    }

    fn amg_builder(require_spd: bool) -> PcBuilder {
        Arc::new(move || {
            let mut builder = AMG::builder()
                .coarsening_type(CoarsenType::HMIS)
                .interpolation_type(InterpType::Extended)
                .relaxation_type(RelaxType::L1Jacobi)
                .smoothing_sweeps(1, 1)
                .logging_level(0)
                .print_level(0);
            if require_spd {
                builder = builder.require_spd(true);
            }
            let amg = builder.build(&Mat::<f64>::zeros(0, 0))?;
            Ok(Box::new(amg) as Box<dyn Preconditioner>)
        })
    }

    fn analyze_matrix(matrix: &CsrMatrix<f64>) -> Analysis {
        let nrows = matrix.nrows();
        let ncols = matrix.ncols();
        let nnz = matrix.nnz();
        let density = if nrows == 0 || ncols == 0 {
            0.0
        } else {
            nnz as f64 / ((nrows as f64) * (ncols as f64))
        };
        let approx_symmetric = estimate_symmetry(matrix, 1e-12, 2_000);
        let has_diag_zeros = detect_diag_issues(matrix, 1e-14, 20_000);
        Analysis {
            nnz,
            density,
            approx_symmetric,
            has_diag_zeros,
        }
    }

    fn repair_diagonal_csr(a: &CsrMatrix<f64>, tol: f64, tau: f64) -> (CsrMatrix<f64>, usize) {
        let nrows = a.nrows();
        let ncols = a.ncols();

        let mut rp: Vec<usize> = Vec::with_capacity(nrows + 1);
        let mut ci: Vec<usize> = Vec::with_capacity(a.nnz() + nrows);
        let mut vv: Vec<f64> = Vec::with_capacity(a.nnz() + nrows);

        rp.push(0);
        let mut fixed = 0usize;

        for i in 0..nrows {
            let (cols, vals) = a.row(i);

            let row_abs_sum: f64 = vals.iter().map(|x| x.abs()).sum();
            let repl = (tau * row_abs_sum).max(tol);

            let mut diag_handled = false;

            for (&c, &v) in cols.iter().zip(vals.iter()) {
                if !diag_handled && i < ncols && c > i {
                    ci.push(i);
                    vv.push(repl);
                    fixed += 1;
                    diag_handled = true;
                }

                if c == i {
                    let new_v = if v.abs() <= tol {
                        fixed += 1;
                        repl
                    } else {
                        v
                    };
                    ci.push(c);
                    vv.push(new_v);
                    diag_handled = true;
                } else {
                    ci.push(c);
                    vv.push(v);
                }
            }

            if !diag_handled && i < ncols {
                ci.push(i);
                vv.push(repl);
                fixed += 1;
            }

            rp.push(ci.len());
        }

        (CsrMatrix::from_csr(nrows, ncols, rp, ci, vv), fixed)
    }

    fn estimate_symmetry(matrix: &CsrMatrix<f64>, tol: f64, max_checks: usize) -> bool {
        let n = matrix.nrows().min(matrix.ncols());
        let mut checks = 0usize;
        for i in 0..n {
            let (cols_i, vals_i) = matrix.row(i);
            for (&j, &v) in cols_i.iter().zip(vals_i.iter()) {
                if i == j || j >= n {
                    continue;
                }
                checks += 1;
                if checks > max_checks {
                    return true;
                }
                if let Some(v_t) = lookup(matrix, j, i) {
                    if (v_t - v).abs() > tol {
                        return false;
                    }
                } else {
                    return false;
                }
            }
        }
        true
    }

    fn detect_diag_issues(matrix: &CsrMatrix<f64>, tol: f64, max_rows: usize) -> bool {
        let n = matrix.nrows().min(matrix.ncols());
        let limit = n.min(max_rows);
        for i in 0..limit {
            match lookup(matrix, i, i) {
                Some(val) if val.abs() > tol => continue,
                _ => return true,
            }
        }
        false
    }

    fn lookup(matrix: &CsrMatrix<f64>, row: usize, col: usize) -> Option<f64> {
        if row >= matrix.nrows() {
            return None;
        }
        let (cols, vals) = matrix.row(row);
        for (&c, &v) in cols.iter().zip(vals.iter()) {
            if c == col {
                return Some(v);
            }
            if c > col {
                break;
            }
        }
        None
    }

    #[cfg(feature = "mpi")]
    fn slice_csr_rows(matrix: &CsrMatrix<f64>, start: usize, end: usize) -> CsrMatrix<f64> {
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

        CsrMatrix::from_csr(end - start, matrix.ncols(), local_rp, local_ci, local_vals)
    }

    fn load_and_distribute(
        mat_path: &Path,
        rhs_path: &Path,
        comm: &UniverseComm,
    ) -> Result<(Problem, Analysis), KError> {
        let rank = comm.rank();
        let size = comm.size();
        let is_parallel = is_comm_parallel(comm, size);

        let mut matrix_root: Option<CsrMatrix<f64>> = None;
        let mut rhs_root: Option<Vec<f64>> = None;

        if rank == 0 {
            let mat_mm = read_matrix_market(mat_path)?;
            let rhs_mm = read_matrix_market(rhs_path)?;
            let csr = mat_mm.to_csr_matrix()?;
            let (csr_fixed, fixed) = repair_diagonal_csr(&csr, 1e-14, 1e-8);
            if fixed > 0 {
                eprintln!("(info) repaired {fixed} diagonal entries (|diag|<=1e-14 or missing).");
            }
            matrix_root = Some(csr_fixed);
            rhs_root = Some(rhs_mm.to_vector()?);
        }

        if is_parallel {
            #[cfg(feature = "mpi")]
            {
                if let Some(mpi_comm) = as_mpi(comm) {
                    let matrix = broadcast_csr(matrix_root.as_ref(), &mpi_comm)?;
                    let rhs = broadcast_vec(rhs_root.as_ref().map(|v| v.as_slice()), &mpi_comm)?;
                    let analysis = analyze_matrix(&matrix);

                    let part = partition_rows(matrix.nrows() as u64, comm);
                    let part_usize: Vec<usize> = part.iter().map(|&x| x as usize).collect();
                    let row_start = part_usize[rank];
                    let row_end = part_usize[rank + 1];

                    let local = slice_csr_rows(&matrix, row_start, row_end);
                    let rhs_local = rhs[row_start..row_end].to_vec();
                    let op = DistCsrOp::from_local_rows(
                        matrix.nrows(),
                        row_start,
                        &local,
                        &part_usize,
                        comm.clone(),
                    )?;

                    let problem = Problem {
                        op: Arc::new(op),
                        rhs: rhs_local,
                        local_n: row_end - row_start,
                        global_n: matrix.nrows(),
                        comm: comm.clone(),
                        backend_descr: format!("DistCSR ({} ranks)", size),
                    };

                    return Ok((problem, analysis));
                }
            }
            #[cfg(not(feature = "mpi"))]
            {
                unreachable!("MPI branch compiled without mpi feature");
            }
        }

        let matrix = matrix_root.expect("matrix available on rank 0 in serial");
        let rhs = rhs_root.expect("rhs available on rank 0 in serial");
        let analysis = analyze_matrix(&matrix);
        let csr_arc = Arc::new(matrix);
        let op = CsrOp::new(csr_arc.clone()).with_comm(comm.clone());
        let problem = Problem {
            op: Arc::new(op),
            rhs,
            local_n: csr_arc.nrows(),
            global_n: csr_arc.nrows(),
            comm: comm.clone(),
            backend_descr: "CSR (serial)".into(),
        };
        Ok((problem, analysis))
    }

    fn true_residual_norm(problem: &Problem, x: &[f64]) -> Result<f64, KError> {
        let mut ax = vec![0.0; problem.local_n];
        problem.op.try_matvec(x, &mut ax)?;
        let mut r2_local = 0.0;
        for i in 0..problem.local_n {
            let ri = problem.rhs[i] - ax[i];
            r2_local += ri * ri;
        }
        Ok(problem.comm.all_reduce_f64(r2_local).sqrt())
    }

    #[cfg(feature = "mpi")]
    fn as_mpi(comm: &UniverseComm) -> Option<Arc<MpiComm>> {
        if let UniverseComm::Mpi(mpi) = comm {
            Some(mpi.clone())
        } else {
            None
        }
    }

    #[cfg(feature = "mpi")]
    fn broadcast_bool(value: bool, mpi_comm: &Arc<MpiComm>) -> bool {
        use mpi::collective::Root;
        use mpi::traits::Communicator;
        let root = 0;
        let mut flag = if mpi_comm.rank == root {
            u8::from(value)
        } else {
            0
        };
        let process = mpi_comm.world.process_at_rank(root as i32);
        process.broadcast_into(std::slice::from_mut(&mut flag));
        flag != 0
    }

    #[cfg(feature = "mpi")]
    fn broadcast_csr(
        matrix: Option<&CsrMatrix<f64>>,
        mpi_comm: &Arc<MpiComm>,
    ) -> Result<CsrMatrix<f64>, KError> {
        use mpi::collective::Root;
        use mpi::traits::Communicator;
        let root = 0;
        let world = &mpi_comm.world;

        let mut dims = [0u64, 0u64];
        if mpi_comm.rank == root {
            let mat = matrix.expect("root holds matrix");
            dims[0] = mat.nrows() as u64;
            dims[1] = mat.ncols() as u64;
        }
        let process = world.process_at_rank(root as i32);
        process.broadcast_into(&mut dims);

        let nrows = dims[0] as usize;
        let ncols = dims[1] as usize;

        let mut row_ptr_u64: Vec<u64> = if mpi_comm.rank == root {
            matrix
                .unwrap()
                .row_ptr()
                .iter()
                .map(|&x| x as u64)
                .collect()
        } else {
            Vec::new()
        };
        let mut col_idx_u64: Vec<u64> = if mpi_comm.rank == root {
            matrix
                .unwrap()
                .col_idx()
                .iter()
                .map(|&x| x as u64)
                .collect()
        } else {
            Vec::new()
        };
        let mut values: Vec<f64> = if mpi_comm.rank == root {
            matrix.unwrap().values().to_vec()
        } else {
            Vec::new()
        };

        broadcast_vec_u64(world, &mut row_ptr_u64, root as i32);
        broadcast_vec_u64(world, &mut col_idx_u64, root as i32);
        broadcast_vec_f64(world, &mut values, root as i32);

        let row_ptr: Vec<usize> = row_ptr_u64.iter().map(|&x| x as usize).collect();
        let col_idx: Vec<usize> = col_idx_u64.iter().map(|&x| x as usize).collect();

        Ok(CsrMatrix::from_csr(nrows, ncols, row_ptr, col_idx, values))
    }

    #[cfg(feature = "mpi")]
    fn broadcast_vec(vec_opt: Option<&[f64]>, mpi_comm: &Arc<MpiComm>) -> Result<Vec<f64>, KError> {
        use mpi::collective::Root;
        use mpi::traits::Communicator;
        let root = 0;
        let world = &mpi_comm.world;

        let mut len = if mpi_comm.rank == root {
            vec_opt.expect("root holds vector").len() as u64
        } else {
            0
        };
        let process = world.process_at_rank(root as i32);
        process.broadcast_into(std::slice::from_mut(&mut len));

        let mut data = if mpi_comm.rank == root {
            vec_opt.unwrap().to_vec()
        } else {
            vec![0.0; len as usize]
        };

        if len > 0 {
            process.broadcast_into(&mut data);
        }
        Ok(data)
    }

    #[cfg(feature = "mpi")]
    fn broadcast_vec_u64(
        world: &mpi::topology::SimpleCommunicator,
        data: &mut Vec<u64>,
        root: i32,
    ) {
        use mpi::collective::Root;
        use mpi::traits::Communicator;
        let mut len = if world.rank() == root {
            data.len() as u64
        } else {
            0
        };
        let process = world.process_at_rank(root);
        process.broadcast_into(std::slice::from_mut(&mut len));
        if world.rank() != root {
            data.resize(len as usize, 0);
        }
        if len > 0 {
            process.broadcast_into(data);
        }
    }

    #[cfg(feature = "mpi")]
    fn broadcast_vec_f64(
        world: &mpi::topology::SimpleCommunicator,
        data: &mut Vec<f64>,
        root: i32,
    ) {
        use mpi::collective::Root;
        use mpi::traits::Communicator;
        let mut len = if world.rank() == root {
            data.len() as u64
        } else {
            0
        };
        let process = world.process_at_rank(root);
        process.broadcast_into(std::slice::from_mut(&mut len));
        if world.rank() != root {
            data.resize(len as usize, 0.0);
        }
        if len > 0 {
            process.broadcast_into(data);
        }
    }
}

#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
#[cfg(not(feature = "complex"))]
fn main() -> Result<(), kryst::error::KError> {
    real_demo::run()
}
