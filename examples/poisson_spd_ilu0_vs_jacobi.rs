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

#[cfg(feature = "complex")]
fn main() {
    eprintln!("poisson_spd_ilu0_vs_jacobi is disabled when the complex feature is enabled.");
}

#[cfg(all(not(feature = "backend-faer"), not(feature = "complex")))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("poisson_spd_ilu0_vs_jacobi requires the backend-faer feature.");
    Ok(())
}

#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
use faer::Mat;
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
use kryst::algebra::prelude::*;
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
use kryst::config::options::PcOptions;
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
use kryst::context::ksp_context::{KspContext, SolverType};
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
use kryst::context::pc_context::PcType;
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
use kryst::error::KError;
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
use kryst::matrix::op::LinOp;
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
use kryst::utils::convergence::SolveStats;
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
use std::error::Error;
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
use std::sync::Arc;

#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
fn make_poisson_1d(n: usize) -> Mat<S> {
    Mat::from_fn(n, n, |i, j| {
        if i == j {
            S::from_real(2.0)
        } else if (i as isize - j as isize).abs() == 1 {
            S::from_real(-1.0)
        } else {
            S::default()
        }
    })
}

#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
fn run_cg_with_pc(
    matrix: Arc<Mat<S>>,
    rhs: &[S],
    pc_type: PcType,
    pc_opts: Option<&PcOptions>,
) -> Result<SolveStats<R>, KError> {
    let mut ctx = KspContext::new();
    ctx.set_type(SolverType::Cg)?
        .set_pc_type(pc_type, pc_opts)?
        .set_tolerances(1e-8, 1e-12, 1e3, 1000);
    let operator: Arc<dyn LinOp<S = S>> = matrix;
    ctx.try_set_operators(operator, None)?;
    ctx.setup()?;
    let mut sol = vec![S::default(); rhs.len()];
    ctx.solve(rhs, &mut sol)
}

#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
fn main() -> Result<(), Box<dyn Error>> {
    let size = 256;
    let matrix = Arc::new(make_poisson_1d(size));
    let rhs: Vec<S> = vec![S::from_real(1.0); size];

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
