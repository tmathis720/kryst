//! Demonstrates AMG's local and canonical distributed-CSR paths on fidapm05.
//!
//! Run with:
//! ```text
//! cargo run --example fidapm05_amg_features
//! cargo run --example fidapm05_amg_features --features rayon
//! cargo mpirun -n 4 --example fidapm05_amg_features --features mpi_examples
//! ```
//!
//! Without MPI, the distributed-CSR phase uses a single-rank `DistCsrOp`. With
//! MPI enabled, rank 0 loads and repairs the Matrix Market system, broadcasts it,
//! and each rank owns a contiguous CSR row partition.

#[cfg(any(feature = "complex", not(feature = "backend-faer")))]
fn main() {
    eprintln!(
        "fidapm05_amg_features requires backend-faer and a real-scalar build (default features)"
    );
}

#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
mod real {
    use faer::Mat;
    use kryst::matrix::{CsrMatrix, DistCsrOp};
    #[cfg(feature = "mpi")]
    use kryst::parallel::MpiComm;
    #[cfg(not(feature = "mpi"))]
    use kryst::parallel::NoComm;
    use kryst::parallel::{Comm, UniverseComm};
    use kryst::preconditioner::amg::{
        AMG, AMGBuilder, CoarseSolve, CoarsenType, InterpType, RelaxPhase, RelaxType,
    };
    use kryst::preconditioner::dist::DistCoarseStrategy;
    use kryst::preconditioner::{PcSide, Preconditioner};
    use kryst::solver::FgmresSolver;
    use kryst::utils::convergence::SolveStats;
    use kryst::utils::matrix_market::read_matrix_market;
    use kryst::utils::matrix_screening::repair_diagonal_csr;
    use std::path::{Path, PathBuf};
    #[cfg(feature = "mpi")]
    use std::sync::Arc;
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
        if comm.rank() == 0 {
            println!(
                "{label:<24} iterations={:<4} reason={:?} reported_residual={:.3e} elapsed={:?}",
                stats.iterations,
                stats.reason,
                stats.final_residual,
                started.elapsed()
            );
        }
        Ok((x, stats))
    }

    fn global_relative_residual(
        op: &dyn kryst::matrix::LinOp<S = f64>,
        rhs: &[f64],
        x: &[f64],
        comm: &UniverseComm,
    ) -> Result<f64, kryst::KError> {
        let mut ax = vec![0.0; rhs.len()];
        op.try_matvec(x, &mut ax)?;
        let local_residual_sq = rhs
            .iter()
            .zip(ax)
            .map(|(b, ax)| (b - ax) * (b - ax))
            .sum::<f64>();
        let local_rhs_sq = rhs.iter().map(|b| b * b).sum::<f64>();
        let residual_sq = comm.all_reduce_f64(local_residual_sq);
        let rhs_sq = comm.all_reduce_f64(local_rhs_sq);
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

    fn print_dist_instrumentation(amg: &AMG, comm: &UniverseComm) {
        if let Some(stats) = amg.dist_apply_stats() {
            let global_bytes = comm.all_reduce_f64(stats.comm_bytes as f64) as usize;
            if comm.rank() == 0 {
                println!(
                    "  distributed route: mode={} coarse_route={} native_support={} root_route={}",
                    stats.mode_label(),
                    stats.coarse_solver_route_label(),
                    stats.reports_distributed_support(),
                    stats.uses_root_gather()
                );
                println!(
                    "  gather diagnostics: setup_fine_matrix={} apply_vectors={}",
                    stats.setup_uses_fine_matrix_gather(),
                    stats.apply_uses_root_vector_gather()
                );
                println!(
                    "  rank-0 apply timings: local={:?} halo/matvec={:?} gather={:?} scatter={:?}",
                    stats.local_apply, stats.halo_exchange, stats.gather, stats.scatter
                );
                println!("  summed communication volume={} bytes", global_bytes);
            }
        }
    }

    fn slice_csr_rows(matrix: &CsrMatrix<f64>, start: usize, end: usize) -> CsrMatrix<f64> {
        let start_nnz = matrix.row_ptr()[start];
        let end_nnz = matrix.row_ptr()[end];
        let row_ptr = (start..=end)
            .map(|row| matrix.row_ptr()[row] - start_nnz)
            .collect();
        CsrMatrix::from_csr(
            end - start,
            matrix.ncols(),
            row_ptr,
            matrix.col_idx()[start_nnz..end_nnz].to_vec(),
            matrix.values()[start_nnz..end_nnz].to_vec(),
        )
    }

    #[cfg(feature = "mpi")]
    fn broadcast_vec_u64(world: &mpi::topology::SimpleCommunicator, data: &mut Vec<u64>) {
        use mpi::collective::Root;
        use mpi::traits::Communicator;
        let root = 0;
        let process = world.process_at_rank(root);
        let mut len = if world.rank() == root {
            data.len() as u64
        } else {
            0
        };
        process.broadcast_into(std::slice::from_mut(&mut len));
        if world.rank() != root {
            data.resize(len as usize, 0);
        }
        if len > 0 {
            process.broadcast_into(data);
        }
    }

    #[cfg(feature = "mpi")]
    fn broadcast_vec_f64(world: &mpi::topology::SimpleCommunicator, data: &mut Vec<f64>) {
        use mpi::collective::Root;
        use mpi::traits::Communicator;
        let root = 0;
        let process = world.process_at_rank(root);
        let mut len = if world.rank() == root {
            data.len() as u64
        } else {
            0
        };
        process.broadcast_into(std::slice::from_mut(&mut len));
        if world.rank() != root {
            data.resize(len as usize, 0.0);
        }
        if len > 0 {
            process.broadcast_into(data);
        }
    }

    #[cfg(feature = "mpi")]
    fn broadcast_problem(
        matrix: Option<&CsrMatrix<f64>>,
        rhs: Option<&[f64]>,
        mpi: &Arc<MpiComm>,
    ) -> Result<(CsrMatrix<f64>, Vec<f64>), kryst::KError> {
        use mpi::collective::Root;
        use mpi::traits::Communicator;
        let root_rank = 0usize;
        let process = mpi.world.process_at_rank(root_rank as i32);
        let mut dims = [0u64; 2];
        if mpi.rank == root_rank {
            let matrix = matrix.expect("root owns matrix");
            dims = [matrix.nrows() as u64, matrix.ncols() as u64];
        }
        process.broadcast_into(&mut dims);

        let mut row_ptr: Vec<u64> = matrix
            .map(|m| m.row_ptr().iter().map(|&v| v as u64).collect())
            .unwrap_or_default();
        let mut col_idx: Vec<u64> = matrix
            .map(|m| m.col_idx().iter().map(|&v| v as u64).collect())
            .unwrap_or_default();
        let mut values = matrix.map(|m| m.values().to_vec()).unwrap_or_default();
        let mut rhs = rhs.map(<[f64]>::to_vec).unwrap_or_default();
        broadcast_vec_u64(&mpi.world, &mut row_ptr);
        broadcast_vec_u64(&mpi.world, &mut col_idx);
        broadcast_vec_f64(&mpi.world, &mut values);
        broadcast_vec_f64(&mpi.world, &mut rhs);
        Ok((
            CsrMatrix::from_csr(
                dims[0] as usize,
                dims[1] as usize,
                row_ptr.into_iter().map(|v| v as usize).collect(),
                col_idx.into_iter().map(|v| v as usize).collect(),
                values,
            ),
            rhs,
        ))
    }

    fn load_problem(
        comm: &UniverseComm,
    ) -> Result<(CsrMatrix<f64>, Vec<f64>, usize), Box<dyn std::error::Error>> {
        let matrix_path = data_path("fidapm05.mtx");
        let rhs_path = data_path("fidapm05_rhs1.mtx");
        let mut matrix = None;
        let mut rhs = None;
        let mut repaired_diagonals = 0usize;
        if comm.rank() == 0 {
            let raw_matrix = read_matrix_market(&matrix_path)?.to_csr_matrix()?;
            let (repaired, repaired_count) = repair_diagonal_csr(&raw_matrix, 1e-14, 1e-8);
            matrix = Some(repaired);
            rhs = Some(read_matrix_market(&rhs_path)?.to_vector()?);
            repaired_diagonals = repaired_count;
        }

        #[cfg(feature = "mpi")]
        if let UniverseComm::Mpi(mpi) = comm {
            let (broadcast_matrix, broadcast_rhs) =
                broadcast_problem(matrix.as_ref(), rhs.as_deref(), mpi)?;
            return Ok((broadcast_matrix, broadcast_rhs, repaired_diagonals));
        }
        Ok((
            matrix.expect("serial rank owns matrix"),
            rhs.expect("serial rank owns RHS"),
            repaired_diagonals,
        ))
    }

    pub fn run() -> Result<(), Box<dyn std::error::Error>> {
        #[cfg(feature = "mpi")]
        let comm = UniverseComm::Mpi(Arc::new(MpiComm::new()));
        #[cfg(not(feature = "mpi"))]
        let comm = UniverseComm::NoComm(NoComm);

        let rank = comm.rank();
        let size = comm.size();
        let matrix_path = data_path("fidapm05.mtx");
        let rhs_path = data_path("fidapm05_rhs1.mtx");
        let (matrix, rhs, repaired_diagonals) = load_problem(&comm)?;
        if matrix.nrows() != matrix.ncols() || matrix.nrows() != rhs.len() {
            return Err("fidapm05 matrix/RHS dimensions are inconsistent".into());
        }

        if rank == 0 {
            println!("AMG MPI feature demonstration using:");
            println!("  matrix: {}", matrix_path.display());
            println!("  rhs:    {}", rhs_path.display());
            println!(
                "  shape={}x{} nnz={} ranks={} rayon={}",
                matrix.nrows(),
                matrix.ncols(),
                matrix.nnz(),
                size,
                cfg!(feature = "rayon")
            );
            println!(
                "  AMG diagonal screening: repaired {} missing/near-zero entries with 1e-8",
                repaired_diagonals
            );
        }

        if size == 1 {
            println!("\n1. Local nonsymmetric AMG + FGMRES(right)");
            let mut local_amg = build_amg(DistCoarseStrategy::RootGather, false)?;
            let setup_started = Instant::now();
            local_amg.setup(&matrix)?;
            println!("  setup elapsed={:?}", setup_started.elapsed());
            print_hierarchy(&local_amg);
            let (local_x, _) = solve("local AMG", &matrix, &rhs, &mut local_amg, &comm)?;
            println!(
                "  explicit relative residual={:.3e}",
                global_relative_residual(&matrix, &rhs, &local_x, &comm)?
            );
        }

        let partition = DistCsrOp::partition_rows_balanced(matrix.nrows(), &comm);
        let row_start = partition[rank];
        let row_end = partition[rank + 1];
        let local_matrix = slice_csr_rows(&matrix, row_start, row_end);
        let local_rhs = rhs[row_start..row_end].to_vec();
        let mut dist = DistCsrOp::from_local_rows(
            matrix.nrows(),
            row_start,
            &local_matrix,
            &partition,
            comm.clone(),
        )?;
        let diagnostics = dist.plan_diagnostics();
        println!(
            "  rank {rank}: owns rows {row_start}..{row_end}, nnz={}, locality={:.1}%, halo_send={}, halo_recv={}, overlap={:?}",
            local_matrix.nnz(),
            100.0 * diagnostics.row_locality_ratio,
            diagnostics.halo_send_volume,
            diagnostics.halo_recv_volume,
            diagnostics.overlap_mode
        );
        comm.barrier();
        if rank == 0 {
            println!("\nDistributed canonical DistCsrOp + distributed_csr AMG + FGMRES(right)");
        }

        let mut dist_amg = build_amg(DistCoarseStrategy::DistributedCsr, true)?;
        let setup_started = Instant::now();
        dist_amg.setup(&dist)?;
        if rank == 0 {
            let support = dist_amg.distributed_support();
            println!(
                "  rank-0 setup elapsed={:?} distributed_support={:?}",
                setup_started.elapsed(),
                support
            );
            if !dist_amg
                .dist_apply_stats()
                .is_some_and(|stats| stats.reports_distributed_support())
            {
                println!(
                    "  note: distributed_csr currently uses distributed SpMV residual correction around local AMG"
                );
            }
        }
        let (dist_x, _) = solve(
            "distributed_csr AMG",
            &dist,
            &local_rhs,
            &mut dist_amg,
            &comm,
        )?;
        let residual = global_relative_residual(&dist, &local_rhs, &dist_x, &comm)?;
        if rank == 0 {
            println!("  explicit global relative residual={residual:.3e}");
        }
        print_dist_instrumentation(&dist_amg, &comm);

        if rank == 0 {
            println!("\nStructure-preserving distributed numeric refresh");
        }
        let scale = 1.05;
        let scaled_values: Vec<f64> = local_matrix
            .values()
            .iter()
            .map(|value| scale * value)
            .collect();
        let scaled_rhs: Vec<f64> = local_rhs.iter().map(|value| scale * value).collect();
        dist.update_numeric(&scaled_values)?;
        let refresh_started = Instant::now();
        dist_amg.update_numeric(&dist)?;
        if rank == 0 {
            println!(
                "  rank-0 numeric refresh elapsed={:?}",
                refresh_started.elapsed()
            );
        }
        let (refreshed_x, _) = solve(
            "refreshed distributed AMG",
            &dist,
            &scaled_rhs,
            &mut dist_amg,
            &comm,
        )?;
        let residual = global_relative_residual(&dist, &scaled_rhs, &refreshed_x, &comm)?;
        if rank == 0 {
            println!("  explicit global relative residual={residual:.3e}");
        }
        print_dist_instrumentation(&dist_amg, &comm);

        Ok(())
    }
}

#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    real::run()
}
