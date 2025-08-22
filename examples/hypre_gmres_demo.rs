//! Demonstration of HYPRE-inspired GMRES improvements
//!
//! This example showcases the critical updates made to the kryst GMRES implementation
//! based on the HYPRE GMRES C code, including:
//!
//! - IEEE safety checks for NaN/Inf detection
//! - Convergence factor monitoring for stagnation detection  
//! - Minimum iteration enforcement
//! - Real residual checking with false convergence detection
//! - Better default values and workspace management
//! - Reference solution monitoring capabilities

use kryst::solver::gmres::{GmresSolver, Preconditioning};
use kryst::solver::legacy::LinearSolver;
use kryst::preconditioner::Jacobi;
use kryst::preconditioner::legacy::Preconditioner;
use kryst::core::traits::{MatVec, Indexing};
use kryst::parallel::UniverseComm;

/// Simple dense matrix for demonstration
#[derive(Clone)]
struct DenseMat {
    data: Vec<Vec<f64>>,
}

impl MatVec<Vec<f64>> for DenseMat {
    fn matvec(&self, x: &Vec<f64>, y: &mut Vec<f64>) {
        for (i, row) in self.data.iter().enumerate() {
            y[i] = row.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
        }
    }
}

impl Indexing for DenseMat {
    fn nrows(&self) -> usize {
        self.data.len()
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== HYPRE-Inspired GMRES Improvements Demo ===\n");

    // Create a well-conditioned non-symmetric system
    let a = DenseMat {
        data: vec![
            vec![4.0, 1.0, 0.0, 0.0],
            vec![1.0, 3.0, 1.0, 0.0], 
            vec![0.0, 1.0, 2.0, 1.0],
            vec![0.0, 0.0, 1.0, 3.0],
        ]
    };
    
    let x_true = vec![1.0, 2.0, 3.0, 4.0];
    let b = {
        let mut b = vec![0.0; 4];
        a.matvec(&x_true, &mut b);
        b
    };

    // Demo 1: Basic GMRES with HYPRE-inspired defaults
    println!("1. Basic GMRES with HYPRE-inspired defaults");
    println!("   - Default restart: 30 (vs HYPRE's 5, more robust)");
    println!("   - IEEE safety checks enabled");
    println!("   - Real residual checking enabled");
    
    let mut x1 = vec![0.0; 4];
    let mut solver1 = GmresSolver::new(30, 1e-10, 100);
    let stats1 = solver1.solve(&a, None, &b, &mut x1, &UniverseComm::NoComm(kryst::parallel::NoComm), None, None)?;
    
    println!("   Result: {} iterations, residual: {:.2e}", stats1.iterations, stats1.final_residual);
    println!("   Converged: {:?}", stats1.reason);
    println!();

    // Demo 2: GMRES with minimum iteration enforcement
    println!("2. GMRES with minimum iteration enforcement");
    println!("   - Minimum 5 iterations before convergence check");
    
    let mut x2 = vec![0.0; 4];
    let mut solver2 = GmresSolver::new(30, 1e-10, 100)
        .with_min_iter(5);
    let stats2 = solver2.solve(&a, None, &b, &mut x2, &UniverseComm::NoComm(kryst::parallel::NoComm), None, None)?;
    
    println!("   Result: {} iterations, residual: {:.2e}", stats2.iterations, stats2.final_residual);
    println!("   Converged: {:?}", stats2.reason);
    println!();

    // Demo 3: GMRES with convergence factor monitoring
    println!("3. GMRES with convergence factor monitoring");
    println!("   - Monitors convergence rate for stagnation");
    println!("   - cf_tol = 0.9 (exits if convergence rate > 0.9)");
    
    let mut x3 = vec![0.0; 4];
    let mut solver3 = GmresSolver::new(30, 1e-10, 100)
        .with_cf_tol(0.9);
    let stats3 = solver3.solve(&a, None, &b, &mut x3, &UniverseComm::NoComm(kryst::parallel::NoComm), None, None)?;
    
    println!("   Result: {} iterations, residual: {:.2e}", stats3.iterations, stats3.final_residual);
    println!("   Converged: {:?}", stats3.reason);
    println!();

    // Demo 4: GMRES with reference solution monitoring
    println!("4. GMRES with reference solution monitoring");
    println!("   - Monitors error against known solution");
    
    let mut x4 = vec![0.0; 4];
    let mut solver4 = GmresSolver::new(30, 1e-10, 100)
        .with_ref_solution(x_true.clone());
    let stats4 = solver4.solve(&a, None, &b, &mut x4, &UniverseComm::NoComm(kryst::parallel::NoComm), None, None)?;
    
    println!("   Result: {} iterations, residual: {:.2e}", stats4.iterations, stats4.final_residual);
    println!("   Solution error: {:.2e}", 
        x4.iter().zip(&x_true).map(|(xi, ti)| (xi - ti).powi(2)).sum::<f64>().sqrt());
    println!();

    // Demo 5: GMRES with preconditioner and all features
    println!("5. GMRES with Jacobi preconditioner and all HYPRE features");
    
    let mut pc = Jacobi::new();
    pc.setup(&a)?;
    
    let mut x5 = vec![0.0; 4];
    let mut solver5 = GmresSolver::new(30, 1e-10, 100)
        .with_preconditioning(Preconditioning::Left)
        .with_min_iter(2)
        .with_cf_tol(0.95)
        .with_ref_solution(x_true.clone());
    
    let stats5 = solver5.solve(&a, Some(&pc), &b, &mut x5, &UniverseComm::NoComm(kryst::parallel::NoComm), None, None)?;
    
    println!("   Result: {} iterations, residual: {:.2e}", stats5.iterations, stats5.final_residual);
    println!("   Solution error: {:.2e}", 
        x5.iter().zip(&x_true).map(|(xi, ti)| (xi - ti).powi(2)).sum::<f64>().sqrt());
    println!("   Converged: {:?}", stats5.reason);
    println!();

    // Demo 6: IEEE safety check demonstration
    println!("6. IEEE safety check demonstration");
    println!("   - Testing with problematic input");
    
    // Create a vector with NaN
    let bad_b = vec![1.0, 2.0, f64::NAN, 4.0];
    let mut x6 = vec![0.0; 4];
    let mut solver6 = GmresSolver::new(30, 1e-10, 100);
    
    match solver6.solve(&a, None, &bad_b, &mut x6, &UniverseComm::NoComm(kryst::parallel::NoComm), None, None) {
        Ok(_) => println!("   Unexpected: solver should have detected NaN"),
        Err(e) => println!("   ✓ IEEE safety check caught: {}", e),
    }
    println!();

    println!("=== Key HYPRE-Inspired Improvements ===");
    println!("✓ IEEE safety checks for NaN/Inf detection");
    println!("✓ Convergence factor monitoring for stagnation detection");
    println!("✓ Minimum iteration enforcement");
    println!("✓ Real residual checking with false convergence detection");
    println!("✓ Better default values (restart=30, robust epsilon)");
    println!("✓ Enhanced workspace management");
    println!("✓ Reference solution monitoring capabilities");
    println!("✓ Improved error handling and logging");

    Ok(())
}
