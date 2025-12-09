//! Demonstrate ILUTP on a convection–diffusion operator.
//!
//! This simple example assembles a nonsymmetric 1D convection–diffusion matrix
//! and runs GMRES with and without an ILUTP preconditioner. The ILUTP setup uses
//! threshold dropping, pivot tolerance, and per-row fill control to handle the
//! mild nonsymmetry.
//!
//! To run:
//! ```bash
//! cargo run --example convection_diffusion_ilutp --features backend-faer
//! ```

#[cfg(not(feature = "backend-faer"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("convection_diffusion_ilutp requires the backend-faer feature.");
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
fn convection_diffusion_1d(n: usize, eps: f64, beta: f64) -> Mat<f64> {
    let h = 1.0 / (n as f64 + 1.0);
    Mat::from_fn(n, n, |i, j| {
        if i == j {
            2.0 * eps / (h * h)
        } else if j + 1 == i {
            -eps / (h * h) - beta / (2.0 * h)
        } else if i + 1 == j {
            -eps / (h * h) + beta / (2.0 * h)
        } else {
            0.0
        }
    })
}

#[cfg(feature = "backend-faer")]
fn run_gmres_with_pc(
    matrix: Arc<Mat<f64>>,
    rhs: &[f64],
    pc_type: PcType,
    pc_opts: Option<&PcOptions>,
) -> Result<SolveStats<f64>, KError> {
    let mut ctx = KspContext::new();
    ctx.set_type(SolverType::Gmres)?;
    ctx.set_pc_type(pc_type, pc_opts)?;
    ctx.set_restart(50);
    ctx.set_tolerances(1e-8, 1e-12, 1e3, 1000);
    let operator: Arc<dyn LinOp<S = f64>> = matrix;
    ctx.try_set_operators(operator, None)?;
    ctx.setup()?;
    let mut sol = vec![0.0; rhs.len()];
    ctx.solve(rhs, &mut sol)
}

#[cfg(feature = "backend-faer")]
fn main() -> Result<(), Box<dyn Error>> {
    let size = 128;
    let eps = 1e-2;
    let beta = 2.0;
    let matrix = Arc::new(convection_diffusion_1d(size, eps, beta));
    let rhs: Vec<f64> = vec![1.0; size];

    println!("Convection–diffusion ILUTP demo");
    println!("===============================");

    let no_pc_stats = run_gmres_with_pc(matrix.clone(), &rhs, PcType::None, None)?;
    println!(
        "GMRES (no PC): iterations = {:3}, residual = {:.2e}",
        no_pc_stats.iterations, no_pc_stats.final_residual
    );

    let mut ilutp_opts = PcOptions::default();
    ilutp_opts.ilutp_max_fill = Some(20);
    ilutp_opts.ilutp_drop_tol = Some(1e-4);
    ilutp_opts.ilutp_perm_tol = Some(0.1);

    let ilutp_stats = run_gmres_with_pc(matrix, &rhs, PcType::Ilutp, Some(&ilutp_opts))?;
    println!(
        "GMRES + ILUTP: iterations = {:3}, residual = {:.2e}",
        ilutp_stats.iterations, ilutp_stats.final_residual
    );
    println!("ILUTP keeps the convection–diffusion solve stable without pivot breakdowns.");

    Ok(())
}
