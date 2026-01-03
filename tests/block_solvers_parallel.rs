#![cfg(not(feature = "complex"))]

mod fixtures;

use kryst::context::ksp_context::Workspace;
use kryst::error::KError;
use kryst::parallel::UniverseComm;
use kryst::preconditioner::PcSide;
use kryst::solver::LinearSolver;
use kryst::solver::block::{BlockKrylovOptions, bicgstab::BlockBicgstabSolver, gmres::BlockGmresSolver};
use kryst::utils::convergence::ConvergedReason;

fn block_rhs(n: usize, p: usize) -> Vec<f64> {
    let mut b = vec![0.0; n * p];
    for col in 0..p {
        for row in 0..n {
            b[col * n + row] = 1.0 + (row + col) as f64;
        }
    }
    b
}

fn assert_converged(stats: kryst::utils::convergence::SolveStats<f64>) {
    assert!(matches!(
        stats.reason,
        ConvergedReason::ConvergedAtol | ConvergedReason::ConvergedRtol
    ));
}

#[cfg(feature = "rayon")]
#[test]
fn rayon_block_gmres_converges() -> Result<(), KError> {
    let comm = UniverseComm::Rayon(kryst::parallel::RayonComm::new());
    let a = fixtures::csr_poisson_1d(48);
    let n = a.nrows();
    let b = block_rhs(n, 2);
    let mut x = vec![0.0; n * 2];
    let mut ws = Workspace::default();
    let mut opts = BlockKrylovOptions::default();
    opts.block_size = 2;
    opts.restart_blocks = 6;
    opts.max_iters = 2000;
    let mut solver = BlockGmresSolver::new(opts);
    let stats = solver.solve(&a, None, &b, &mut x, PcSide::Left, &comm, None, Some(&mut ws))?;
    assert_converged(stats);
    Ok(())
}

#[cfg(feature = "rayon")]
#[test]
fn rayon_block_bicgstab_converges() -> Result<(), KError> {
    let comm = UniverseComm::Rayon(kryst::parallel::RayonComm::new());
    let a = fixtures::csr_poisson_1d(48);
    let n = a.nrows();
    let b = block_rhs(n, 2);
    let mut x = vec![0.0; n * 2];
    let mut ws = Workspace::default();
    let mut opts = BlockKrylovOptions::default();
    opts.block_size = 2;
    opts.max_iters = 200;
    let mut solver = BlockBicgstabSolver::new(opts);
    let stats = solver.solve(&a, None, &b, &mut x, PcSide::Left, &comm, None, Some(&mut ws))?;
    assert_converged(stats);
    Ok(())
}

#[cfg(feature = "mpi")]
#[test]
fn mpi_block_gmres_converges() -> Result<(), KError> {
    let Some(comm) = kryst::parallel::MpiComm::try_new() else {
        return Ok(());
    };
    let comm = UniverseComm::Mpi(std::sync::Arc::new(comm));
    let a = fixtures::csr_poisson_1d(48);
    let n = a.nrows();
    let b = block_rhs(n, 2);
    let mut x = vec![0.0; n * 2];
    let mut ws = Workspace::default();
    let mut opts = BlockKrylovOptions::default();
    opts.block_size = 2;
    opts.restart_blocks = 6;
    opts.max_iters = 2000;
    let mut solver = BlockGmresSolver::new(opts);
    let stats = solver.solve(&a, None, &b, &mut x, PcSide::Left, &comm, None, Some(&mut ws))?;
    assert_converged(stats);
    Ok(())
}

#[cfg(feature = "mpi")]
#[test]
fn mpi_block_bicgstab_converges() -> Result<(), KError> {
    let Some(comm) = kryst::parallel::MpiComm::try_new() else {
        return Ok(());
    };
    let comm = UniverseComm::Mpi(std::sync::Arc::new(comm));
    let a = fixtures::csr_poisson_1d(48);
    let n = a.nrows();
    let b = block_rhs(n, 2);
    let mut x = vec![0.0; n * 2];
    let mut ws = Workspace::default();
    let mut opts = BlockKrylovOptions::default();
    opts.block_size = 2;
    opts.max_iters = 200;
    let mut solver = BlockBicgstabSolver::new(opts);
    let stats = solver.solve(&a, None, &b, &mut x, PcSide::Left, &comm, None, Some(&mut ws))?;
    assert_converged(stats);
    Ok(())
}
