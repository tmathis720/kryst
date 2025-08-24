//! Demonstration of basic and preconditioned GMRES usage.
//!
//! This example solves a small non-symmetric system and shows how to
//! optionally apply a Jacobi preconditioner. It also demonstrates error
//! handling when the right-hand side contains invalid values.

use faer::mat;
use kryst::core::traits::MatVec;
use kryst::parallel::UniverseComm;
use kryst::preconditioner::{Jacobi, Preconditioner, PcSide};
use kryst::{LinearSolver, solver::GmresSolver};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Build a well-conditioned non-symmetric system
    let a = mat![
        [4.0, 1.0, 0.0, 0.0],
        [1.0, 3.0, 1.0, 0.0],
        [0.0, 1.0, 2.0, 1.0],
        [0.0, 0.0, 1.0, 3.0],
    ];

    let x_true = vec![1.0, 2.0, 3.0, 4.0];
    let mut b = vec![0.0; 4];
    a.as_ref().matvec(&x_true, &mut b);

    let comm = UniverseComm::NoComm(kryst::parallel::NoComm);

    // Basic GMRES solve
    let mut solver = GmresSolver::new(30, 1e-10, 100);
    let mut x = vec![0.0; 4];
    let stats = solver.solve(&a, None, &b, &mut x, PcSide::Left, &comm, None, None)?;
    println!("Basic GMRES: iterations = {}, residual = {:.2e}", stats.iterations, stats.final_residual);

    // GMRES with Jacobi preconditioner
    let mut pc = Jacobi::new();
    pc.setup(&a)?;
    let mut x_pc = vec![0.0; 4];
    let stats_pc = solver.solve(&a, Some(&pc), &b, &mut x_pc, PcSide::Left, &comm, None, None)?;
    println!("Preconditioned GMRES: iterations = {}, residual = {:.2e}", stats_pc.iterations, stats_pc.final_residual);

    // Demonstrate IEEE safety check with NaN in rhs
    let bad_b = vec![1.0, 2.0, f64::NAN, 4.0];
    let mut x_bad = vec![0.0; 4];
    match solver.solve(&a, None, &bad_b, &mut x_bad, PcSide::Left, &comm, None, None) {
        Ok(_) => println!("Unexpected success with NaN RHS"),
        Err(e) => println!("\u{2713} Detected invalid input: {}", e),
    }

    Ok(())
}
