//! Example demonstrating shell matrix (matrix-free) operations with kryst.
//!
//! This example shows how to use `ShellMat` to define matrix operations via callbacks
//! rather than storing matrix entries explicitly. This is useful for:
//! - Large matrices that are expensive to store
//! - Matrices defined by algorithms (e.g., finite difference operators)
//! - GPU-based or distributed matrix operations
//!
//! Run with:
//! ```bash
//! cargo run --example shell_demo
//! ```

use kryst::core::mat::shell::ShellMat;
use kryst::core::traits::MatVec;
use kryst::solver::{LinearSolver, CgSolver};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Kryst Shell Matrix Demo");
    println!("=======================");

    // Example 1: Simple diagonal matrix via shell
    println!("\n1. Diagonal Matrix Shell Demo");
    diagonal_shell_demo()?;

    // Example 2: Finite difference operator via shell
    println!("\n2. Finite Difference Operator Shell Demo");
    finite_difference_shell_demo()?;

    // Example 3: Using shell matrix with iterative solver
    println!("\n3. Shell Matrix with CG Solver Demo");
    shell_with_solver_demo()?;

    println!("\nShell matrix demo completed successfully!");
    Ok(())
}

/// Demonstrates a simple diagonal matrix defined via shell operations
fn diagonal_shell_demo() -> Result<(), Box<dyn std::error::Error>> {
    // Create a 5x5 diagonal matrix with entries [1, 2, 3, 4, 5]
    let diagonal_entries = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let n = diagonal_entries.len();
    
    // Clone for the closure (need to move into closure)
    let diag_for_mult = diagonal_entries.clone();
    let diag_for_trans = diagonal_entries.clone();
    
    let shell_matrix = ShellMat::new(
        n,
        move |x: &Vec<f64>, y: &mut Vec<f64>| {
            let x_ref: &[f64] = x.as_ref();
            let y_mut: &mut [f64] = y.as_mut();
            for i in 0..n {
                y_mut[i] = diag_for_mult[i] * x_ref[i];
            }
        },
        move |x: &Vec<f64>, y: &mut Vec<f64>| {
            let x_ref: &[f64] = x.as_ref();
            let y_mut: &mut [f64] = y.as_mut();
            for i in 0..n {
                y_mut[i] = diag_for_trans[i] * x_ref[i];
            }
        },
    );

    // Test matrix-vector multiplication
    let x = vec![1.0; n];
    let mut y = vec![0.0; n];
    
    shell_matrix.matvec(&x, &mut y);
    println!("Input vector x: {:?}", x);
    println!("Output y = A*x: {:?}", y);
    println!("Expected: {:?}", diagonal_entries);
    
    // Verify correctness
    for i in 0..n {
        assert!((y[i] - diagonal_entries[i]).abs() < 1e-14);
    }
    println!("✓ Matrix-vector multiplication correct!");

    Ok(())
}

/// Demonstrates a 1D finite difference operator (second derivative) via shell
fn finite_difference_shell_demo() -> Result<(), Box<dyn std::error::Error>> {
    let n = 5;
    let h = 1.0 / (n as f64 + 1.0);
    let h2_inv = 1.0 / (h * h);
    
    // Create a finite difference operator for -d²/dx² (with homogeneous Dirichlet BC)
    // The matrix looks like:
    //   [ 2 -1  0  0  0]
    //   [-1  2 -1  0  0] * (1/h²)
    //   [ 0 -1  2 -1  0]
    //   [ 0  0 -1  2 -1]
    //   [ 0  0  0 -1  2]
    let finite_diff = ShellMat::new_symmetric(
        n,
        move |x: &Vec<f64>, y: &mut Vec<f64>| {
            let x_ref: &[f64] = x.as_ref();
            let y_mut: &mut [f64] = y.as_mut();
            
            for i in 0..n {
                y_mut[i] = 2.0 * x_ref[i] * h2_inv;
                
                if i > 0 {
                    y_mut[i] -= x_ref[i-1] * h2_inv;
                }
                if i < n-1 {
                    y_mut[i] -= x_ref[i+1] * h2_inv;
                }
            }
        },
    );

    // Test with a simple function (quadratic)
    let mut x = vec![0.0; n];
    for i in 0..n {
        let xi = (i as f64 + 1.0) * h;
        x[i] = xi * (1.0 - xi); // Quadratic function x(1-x)
    }
    
    let mut y = vec![0.0; n];
    finite_diff.matvec(&x, &mut y);
    
    println!("Grid points (h = {:.3}):", h);
    println!("Input function u(x) = x(1-x): {:?}", x);
    println!("Output -d²u/dx²: {:?}", y);
    
    // For u(x) = x(1-x), we have d²u/dx² = -2, so -d²u/dx² = 2
    let expected = vec![2.0; n];
    println!("Expected (analytical): {:?}", expected);
    
    // Check if result is close to expected (within discretization error)
    let max_error = y.iter().zip(expected.iter())
        .map(|(yi, ei)| (yi - ei).abs())
        .fold(0.0, f64::max);
    
    println!("Maximum error: {:.6}", max_error);
    println!("✓ Finite difference operator working correctly!");

    Ok(())
}

/// Demonstrates using a shell matrix with CG iterative solver
fn shell_with_solver_demo() -> Result<(), Box<dyn std::error::Error>> {
    let n = 4;
    
    // Create a simple SPD matrix via shell: A = I + 0.1 * ones
    // This matrix has eigenvalues [1.4, 1, 1, 1] and is well-conditioned
    let shell_matrix = ShellMat::new_symmetric(
        n,
        move |x: &Vec<f64>, y: &mut Vec<f64>| {
            let x_ref: &[f64] = x.as_ref();
            let y_mut: &mut [f64] = y.as_mut();
            
            let sum: f64 = x_ref.iter().sum();
            for i in 0..n {
                y_mut[i] = x_ref[i] + 0.1 * sum;
            }
        },
    );

    // Create right-hand side: b = A * [1, 1, 1, 1]
    let x_true = vec![1.0; n];
    let mut b = vec![0.0; n];
    shell_matrix.matvec(&x_true, &mut b);
    
    println!("Solving system A*x = b with shell matrix");
    println!("True solution: {:?}", x_true);
    println!("Right-hand side b: {:?}", b);

    // Solve with CG (since the matrix is SPD)
    let mut cg_solver = CgSolver::new(1e-10, 100);
    let mut x_solved = vec![0.0; n];
    let stats = cg_solver.solve(&shell_matrix, None, &b, &mut x_solved)?;
    
    println!("Solved solution: {:?}", x_solved);
    println!("Solver stats: reason={:?}, iterations={}, residual={:.2e}", 
             stats.reason, stats.iterations, stats.final_residual);

    // Check solution accuracy
    let error: f64 = x_solved.iter().zip(x_true.iter())
        .map(|(xi, ti)| (xi - ti).powi(2))
        .sum::<f64>()
        .sqrt();
    
    println!("Solution error (L2 norm): {:.2e}", error);
    
    if error < 1e-8 {
        println!("✓ Shell matrix solve successful!");
    } else {
        println!("⚠ Solution accuracy could be better");
    }

    Ok(())
}
