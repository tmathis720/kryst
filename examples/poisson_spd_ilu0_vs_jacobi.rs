//! Compare Jacobi vs ILU(0) on a 1D Poisson system.
//!
//! This example assembles a simple symmetric positive definite tridiagonal matrix,
//! then runs CG first with plain Jacobi and then with the HYPRE-style ILU(0)
//! preconditioner. The iteration counts and residual norms highlight the
//! advantage of ILU(0) for SPD problems where the fill pattern is fixed.
//!
//! To run:
//! ```bash
//! cargo run --example poisson_spd_ilu0_vs_jacobi --features backend-faer
//! ```

#[cfg(not(feature = "backend-faer"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("poisson_spd_ilu0_vs_jacobi requires the backend-faer feature.");
    Ok(())
}

#[cfg(feature = "backend-faer")]
use faer::Mat;
#[cfg(feature = "backend-faer")]
use kryst::config::options::PcOptions;
#[cfg(feature = "backend-faer")]
use kryst::context::ksp_context::{KspContext, SolverType};
#[cfg(feature = "backend-faer")]
use kryst::context::pc_context::PcType;
#[cfg(feature = "backend-faer")]
use kryst::error::KError;
#[cfg(feature = "backend-faer")]
use kryst::matrix::op::LinOp;
#[cfg(feature = "backend-faer")]
use kryst::utils::convergence::SolveStats;
#[cfg(feature = "backend-faer")]
use std::error::Error;
#[cfg(feature = "backend-faer")]
use std::sync::Arc;

#[cfg(feature = "backend-faer")]
fn make_poisson_1d(n: usize) -> Mat<f64> {
    Mat::from_fn(n, n, |i, j| {
        if i == j {
            2.0
        } else if (i as isize - j as isize).abs() == 1 {
            -1.0
        } else {
            0.0
        }
    })
}

#[cfg(feature = "backend-faer")]
fn run_cg_with_pc(
    matrix: Arc<Mat<f64>>,
    rhs: &[f64],
    pc_type: PcType,
    pc_opts: Option<&PcOptions>,
) -> Result<SolveStats<f64>, KError> {
    let mut ctx = KspContext::new();
    ctx.set_type(SolverType::Cg)?
        .set_pc_type(pc_type, pc_opts)?
        .set_tolerances(1e-8, 1e-12, 1e3, 1000);
    let operator: Arc<dyn LinOp<S = f64>> = matrix;
    ctx.try_set_operators(operator, None)?;
    ctx.setup()?;
    let mut sol = vec![0.0; rhs.len()];
    ctx.solve(rhs, &mut sol)
}

#[cfg(feature = "backend-faer")]
fn main() -> Result<(), Box<dyn Error>> {
    let size = 256;
    let matrix = Arc::new(make_poisson_1d(size));
    let rhs: Vec<f64> = vec![1.0; size];

    println!("Poisson SPD example: Jacobi vs ILU(0)");
    println!("======================================");

    let jacobi_stats = run_cg_with_pc(matrix.clone(), &rhs, PcType::Jacobi, None)?;
    println!(
        "Jacobi:    iterations = {:3}, residual = {:.2e}",
        jacobi_stats.iterations, jacobi_stats.final_residual
    );

    let mut ilu_opts = PcOptions::default();
    ilu_opts.ilu_type = Some("ilu0".to_string());
    ilu_opts.ilu_triangular_solve = Some("exact".to_string());

    let ilu_stats = run_cg_with_pc(matrix, &rhs, PcType::Ilu, Some(&ilu_opts))?;
    println!(
        "ILU(0):    iterations = {:3}, residual = {:.2e}",
        ilu_stats.iterations, ilu_stats.final_residual
    );
    println!("ILU(0) demonstrates better convergence on this SPD problem.");

    Ok(())
}
