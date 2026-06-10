//! Demonstrates AMG's local and canonical distributed-CSR paths on fidapm05.
//!
//! Run with:
//! ```text
//! cargo run --example fidapm05_amg_features
//! cargo run --example fidapm05_amg_features --features rayon
//! ```
//!
//! The distributed-CSR phase uses a single-rank `DistCsrOp` so the example is
//! runnable without MPI while still exercising the same halo-aware operator and
//! AMG route used by multi-rank applications.

#[cfg(any(feature = "complex", not(feature = "backend-faer")))]
fn main() {
    eprintln!(
        "fidapm05_amg_features requires backend-faer and a real-scalar build (default features)"
    );
}

#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
mod real {
    use faer::Mat;
    use kryst::matrix::DistCsrOp;
    use kryst::parallel::{NoComm, UniverseComm};
    use kryst::preconditioner::amg::{
        AMG, AMGBuilder, CoarseSolve, CoarsenType, InterpType, RelaxPhase, RelaxType,
    };
    use kryst::preconditioner::dist::DistCoarseStrategy;
    use kryst::preconditioner::{PcDistributedSupport, PcSide, Preconditioner};
    use kryst::solver::FgmresSolver;
    use kryst::utils::convergence::SolveStats;
    use kryst::utils::matrix_market::read_matrix_market;
    use kryst::utils::matrix_screening::repair_diagonal_csr;
    use std::path::{Path, PathBuf};
    use std::time::Instant;

    fn data_path(file: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("examples")
            .join("mtx")
            .join(file)
    }

    fn build_amg(strategy: DistCoarseStrategy, instrument: bool) -> Result<AMG, kryst::KError> {
        AMGBuilder::new()
            .require_spd(false)
            .verify_galerkin(false)
            .verify_p_rank(false)
            .coarsening_type(CoarsenType::HMIS)
            .interpolation_type(InterpType::Classical)
            .relaxation_type(RelaxType::Jacobi)
            .grid_relax_type_all(RelaxType::Jacobi)
            .smoothing_sweeps(2, 2)
            .coarse_threshold(64)
            .max_coarse_size(64)
            .coarse_solve(CoarseSolve::DirectDense)
            .num_grid_sweeps(RelaxPhase::Coarsest, 0)
            .logging_level(1)
            .dist_coarse_strategy(strategy)
            .dist_apply_instrumentation(instrument)
            .build(&Mat::<f64>::zeros(0, 0))
    }

    fn solve(
        label: &str,
        op: &dyn kryst::matrix::LinOp<S = f64>,
        rhs: &[f64],
        amg: &mut AMG,
        comm: &UniverseComm,
    ) -> Result<(Vec<f64>, SolveStats<f64>), kryst::KError> {
        let mut x = vec![0.0; rhs.len()];
        let mut solver = FgmresSolver::new(1e-9, 100, 42);
        let started = Instant::now();
        let stats =
            solver.solve_f64(op, Some(amg), rhs, &mut x, PcSide::Right, comm, None, None)?;
        println!(
            "{label:<24} iterations={:<4} reason={:?} reported_residual={:.3e} elapsed={:?}",
            stats.iterations,
            stats.reason,
            stats.final_residual,
            started.elapsed()
        );
        Ok((x, stats))
    }

    fn relative_residual(
        op: &dyn kryst::matrix::LinOp<S = f64>,
        rhs: &[f64],
        x: &[f64],
    ) -> Result<f64, kryst::KError> {
        let mut ax = vec![0.0; rhs.len()];
        op.try_matvec(x, &mut ax)?;
        let residual_sq = rhs
            .iter()
            .zip(ax)
            .map(|(b, ax)| (b - ax) * (b - ax))
            .sum::<f64>();
        let rhs_sq = rhs.iter().map(|b| b * b).sum::<f64>();
        Ok(residual_sq.sqrt() / rhs_sq.sqrt().max(f64::MIN_POSITIVE))
    }

    fn print_hierarchy(amg: &AMG) {
        if let Some(stats) = amg.stats() {
            println!(
                "  hierarchy: levels={} grid_complexity={:.3} operator_complexity={:.3} total_nnz={}",
                stats.num_levels, stats.grid_complexity, stats.operator_complexity, stats.total_nnz
            );
            for (level, info) in stats.levels.iter().enumerate() {
                println!(
                    "    level {level}: rows={} nnz(A)={} nnz(P)={}",
                    info.n, info.nnz_a, info.nnz_p
                );
            }
        }
    }

    fn print_dist_instrumentation(amg: &AMG) {
        if let Some(stats) = amg.dist_apply_stats() {
            println!(
                "  distributed route: mode={} coarse_route={} native_support={} root_gather={}",
                stats.mode_label(),
                stats.coarse_solver_route_label(),
                stats.reports_distributed_support(),
                stats.uses_root_gather()
            );
            println!(
                "  apply timings: local={:?} halo/matvec={:?} gather={:?} scatter={:?} communicated={} bytes",
                stats.local_apply,
                stats.halo_exchange,
                stats.gather,
                stats.scatter,
                stats.comm_bytes
            );
        }
    }

    pub fn run() -> Result<(), Box<dyn std::error::Error>> {
        let matrix_path = data_path("fidapm05.mtx");
        let rhs_path = data_path("fidapm05_rhs1.mtx");
        let raw_matrix = read_matrix_market(&matrix_path)?.to_csr_matrix()?;
        let (matrix, repaired_diagonals) = repair_diagonal_csr(&raw_matrix, 1e-14, 1e-8);
        let rhs = read_matrix_market(&rhs_path)?.to_vector()?;
        if matrix.nrows() != matrix.ncols() || matrix.nrows() != rhs.len() {
            return Err("fidapm05 matrix/RHS dimensions are inconsistent".into());
        }

        println!("AMG feature demonstration using:");
        println!("  matrix: {}", matrix_path.display());
        println!("  rhs:    {}", rhs_path.display());
        println!(
            "  shape={}x{} nnz={} rayon={}",
            matrix.nrows(),
            matrix.ncols(),
            matrix.nnz(),
            cfg!(feature = "rayon")
        );
        println!(
            "  AMG diagonal screening: repaired {} missing/near-zero entries with 1e-8",
            repaired_diagonals
        );

        let comm = UniverseComm::NoComm(NoComm);

        println!("\n1. Local nonsymmetric AMG + FGMRES(right)");
        let mut local_amg = build_amg(DistCoarseStrategy::RootGather, false)?;
        let setup_started = Instant::now();
        local_amg.setup(&matrix)?;
        println!("  setup elapsed={:?}", setup_started.elapsed());
        print_hierarchy(&local_amg);
        let (local_x, _) = solve("local AMG", &matrix, &rhs, &mut local_amg, &comm)?;
        println!(
            "  explicit relative residual={:.3e}",
            relative_residual(&matrix, &rhs, &local_x)?
        );

        println!("\n2. Canonical DistCsrOp + distributed_csr AMG + FGMRES(right)");
        let partition = vec![0, matrix.nrows()];
        let mut dist =
            DistCsrOp::from_local_rows(matrix.nrows(), 0, &matrix, &partition, comm.clone())?;
        let diagnostics = dist.plan_diagnostics();
        println!(
            "  DistCsr plan: local_rows={} local_nnz={} locality={:.1}% halo_send={} halo_recv={} overlap={:?}",
            matrix.nrows(),
            matrix.nnz(),
            100.0 * diagnostics.row_locality_ratio,
            diagnostics.halo_send_volume,
            diagnostics.halo_recv_volume,
            diagnostics.overlap_mode
        );

        let mut dist_amg = build_amg(DistCoarseStrategy::DistributedCsr, true)?;
        let setup_started = Instant::now();
        dist_amg.setup(&dist)?;
        println!(
            "  setup elapsed={:?} distributed_support={:?}",
            setup_started.elapsed(),
            dist_amg.distributed_support()
        );
        assert_eq!(
            dist_amg.distributed_support(),
            PcDistributedSupport::Distributed
        );
        let (dist_x, _) = solve("distributed_csr AMG", &dist, &rhs, &mut dist_amg, &comm)?;
        println!(
            "  explicit relative residual={:.3e}",
            relative_residual(&dist, &rhs, &dist_x)?
        );
        print_dist_instrumentation(&dist_amg);

        println!("\n3. Structure-preserving distributed numeric refresh");
        let scale = 1.05;
        let scaled_values: Vec<f64> = matrix.values().iter().map(|value| scale * value).collect();
        let scaled_rhs: Vec<f64> = rhs.iter().map(|value| scale * value).collect();
        dist.update_numeric(&scaled_values)?;
        let refresh_started = Instant::now();
        dist_amg.update_numeric(&dist)?;
        println!("  numeric refresh elapsed={:?}", refresh_started.elapsed());
        let (refreshed_x, _) = solve(
            "refreshed distributed AMG",
            &dist,
            &scaled_rhs,
            &mut dist_amg,
            &comm,
        )?;
        println!(
            "  explicit relative residual={:.3e}",
            relative_residual(&dist, &scaled_rhs, &refreshed_x)?
        );
        print_dist_instrumentation(&dist_amg);

        Ok(())
    }
}

#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    real::run()
}
