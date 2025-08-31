//! HYPRE-Inspired AMG Preconditioner Demo (updated API, no legacy bridge)

use faer::Mat;
use kryst::error::KError;
use kryst::preconditioner::amg::{AMGBuilder, CoarsenType, InterpType, RelaxType};
use kryst::preconditioner::{PcSide, Preconditioner};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("    HYPRE-Inspired AMG Preconditioner Demo (dense faer::Mat)");
    println!("============================================================\n");

    // Create test problems of varying difficulty
    demo_symmetric_positive_definite()?;
    demo_anisotropic_problem()?;
    demo_configuration_builder()?;
    demo_safety_features()?;

    println!("  All AMG demos completed successfully!");
    Ok(())
}

/// Create a 2D Laplacian matrix directly (5-point stencil) and add I for better conditioning.
fn create_2d_laplacian(nx: usize, ny: usize) -> Mat<f64> {
    let n = nx * ny;
    let mut a = Mat::zeros(n, n);

    for i in 0..ny {
        for j in 0..nx {
            let idx = i * nx + j;
            let mut diag = 0.0;

            // Left neighbor
            if j > 0 {
                a[(idx, idx - 1)] = -1.0;
                diag += 1.0;
            }
            // Right neighbor
            if j + 1 < nx {
                a[(idx, idx + 1)] = -1.0;
                diag += 1.0;
            }
            // Top neighbor
            if i > 0 {
                a[(idx, idx - nx)] = -1.0;
                diag += 1.0;
            }
            // Bottom neighbor
            if i + 1 < ny {
                a[(idx, idx + nx)] = -1.0;
                diag += 1.0;
            }

            a[(idx, idx)] = diag + 1.0;
        }
    }

    a
}

/// Create an anisotropic diffusion matrix: strong horizontal (−1), weak vertical (−anisotropy).
fn create_anisotropic_matrix(nx: usize, ny: usize, anisotropy: f64) -> Mat<f64> {
    let n = nx * ny;
    let mut a = Mat::zeros(n, n);

    for i in 0..ny {
        for j in 0..nx {
            let idx = i * nx + j;
            let mut diag = 0.0;

            // Horizontal (strong)
            if j > 0 {
                a[(idx, idx - 1)] = -1.0;
                diag += 1.0;
            }
            if j + 1 < nx {
                a[(idx, idx + 1)] = -1.0;
                diag += 1.0;
            }

            // Vertical (weak)
            if i > 0 {
                a[(idx, idx - nx)] = -anisotropy;
                diag += anisotropy;
            }
            if i + 1 < ny {
                a[(idx, idx + nx)] = -anisotropy;
                diag += anisotropy;
            }

            a[(idx, idx)] = diag + 0.1; // small regularization
        }
    }

    a
}

/// Demonstrate AMG on symmetric positive definite problems (dense Mat + new Preconditioner API)
fn demo_symmetric_positive_definite() -> Result<(), KError> {
    println!("Testing AMG on Symmetric Positive Definite Problem");
    println!("--------------------------------------------------");

    // 2D Laplacian (5×5 grid)
    let a = create_2d_laplacian(5, 5);
    let n = a.nrows();

    println!("Matrix size: {}x{}", a.nrows(), a.ncols());

    // Default AMG using builder
    let mut amg_default = AMGBuilder::new().build(&a)?;
    amg_default.setup(&a)?;
    println!("  Default AMG construction & setup successful");

    // HYPRE-style AMG configuration
    let mut amg_hypre = AMGBuilder::new()
        .max_levels(15)
        .strong_threshold(0.25)
        .coarsening_type(CoarsenType::HMIS)
        .interpolation_type(InterpType::Extended)
        .relaxation_type(RelaxType::GaussSeidel)
        .smoothing_sweeps(2, 2)
        .enable_logging()
        .build(&a)?;
    amg_hypre.setup(&a)?;
    println!("  HYPRE-style AMG construction & setup successful");

    // Apply preconditioner (no legacy bridge; AMG implements `Preconditioner`)
    let x = vec![1.0; n];
    let mut y = vec![0.0; n];
    amg_hypre.apply(PcSide::Left, &x, &mut y)?;
    println!("  AMG preconditioning applied successfully");

    println!("   Preconditioning effect:");
    println!(
        "   Input norm:  {:.6}",
        x.iter().map(|v| v * v).sum::<f64>().sqrt()
    );
    println!(
        "   Output norm: {:.6}",
        y.iter().map(|v| v * v).sum::<f64>().sqrt()
    );
    println!();
    Ok(())
}

/// Demonstrate AMG on anisotropic problems (dense Mat)
fn demo_anisotropic_problem() -> Result<(), KError> {
    println!("   Testing AMG on Anisotropic Problem");
    println!("-------------------------------------");

    let a = create_anisotropic_matrix(4, 4, 0.01);
    println!("Anisotropic matrix size: {}x{}", a.nrows(), a.ncols());

    let strategies = [
        ("HMIS", CoarsenType::HMIS),
        ("RS", CoarsenType::RS),
        ("PMIS", CoarsenType::PMIS),
        ("Falgout", CoarsenType::Falgout),
    ];

    for (name, strategy) in &strategies {
        let mut amg = AMGBuilder::new()
            .max_levels(5)
            .strong_threshold(0.5) // higher threshold for anisotropy
            .coarsening_type(*strategy)
            .interpolation_type(InterpType::Extended)
            .build(&a)?;
        amg.setup(&a)?;
        println!("  {name} coarsening strategy successful");
    }

    println!();
    Ok(())
}

/// Demonstrate the configuration builder pattern (dense Mat)
fn demo_configuration_builder() -> Result<(), KError> {
    println!("    Testing AMG Configuration Builder");
    println!("------------------------------------");

    // Simple 1D 3-point stencil (n = 9)
    let n = 9;
    let a = Mat::from_fn(n, n, |i, j| {
        if i == j {
            4.0
        } else if (i as isize - j as isize).abs() == 1 {
            -1.0
        } else {
            0.0
        }
    });

    // Variants
    let configs = [
        (
            "Conservative",
            AMGBuilder::new()
                .max_levels(3)
                .strong_threshold(0.5)
                .coarsening_type(CoarsenType::RS)
                .interpolation_type(InterpType::Classical),
        ),
        (
            "Aggressive",
            AMGBuilder::new()
                .max_levels(10)
                .strong_threshold(0.1)
                .coarsening_type(CoarsenType::HMIS)
                .interpolation_type(InterpType::Extended)
                .smoothing_sweeps(3, 3),
        ),
        (
            "Robust",
            AMGBuilder::new()
                .max_levels(8)
                .strong_threshold(0.25)
                .coarsening_type(CoarsenType::Falgout)
                .interpolation_type(InterpType::Standard)
                .enable_logging()
                .enable_printing(),
        ),
    ];

    for (name, builder) in configs {
        match builder.build(&a) {
            Ok(mut amg) => {
                amg.setup(&a)?;
                println!("  {name} configuration successful");
            }
            Err(e) => println!("   {name} configuration failed: {e}"),
        }
    }

    println!();
    Ok(())
}

/// Demonstrate safety features and error handling (dense Mat)
fn demo_safety_features() -> Result<(), KError> {
    println!("    Testing AMG Safety Features");
    println!("-------------------------------");

    // Empty matrix handling
    let empty_matrix = Mat::zeros(0, 0);
    match AMGBuilder::new().build(&empty_matrix) {
        Ok(_) => println!("   Empty matrix should have failed"),
        Err(_) => println!("  Empty matrix correctly rejected"),
    }

    // Non-square matrix handling
    let non_square = Mat::zeros(3, 4);
    match AMGBuilder::new().build(&non_square) {
        Ok(_) => println!("   Non-square matrix should have failed"),
        Err(_) => println!("  Non-square matrix correctly rejected"),
    }

    // Invalid configuration
    let valid_matrix = Mat::identity(5, 5);
    match AMGBuilder::new().max_levels(0).build(&valid_matrix) {
        Ok(_) => println!("   Invalid config should have failed"),
        Err(_) => println!("  Invalid configuration correctly rejected"),
    }

    // Invalid strong threshold
    match AMGBuilder::new()
        .strong_threshold(1.5)
        .build(&valid_matrix)
    {
        Ok(_) => println!("   Invalid threshold should have failed"),
        Err(_) => println!("  Invalid threshold correctly rejected"),
    }

    // Note: We can't easily create NaN in this context, so we skip that test here.
    println!("ℹ️  IEEE NaN/Inf checks would be tested with problematic matrices");

    // Robust build + apply
    let mut robust_amg = AMGBuilder::new()
        .max_levels(5)
        .strong_threshold(0.25)
        .coarsening_type(CoarsenType::HMIS)
        .interpolation_type(InterpType::Extended)
        .smoothing_sweeps(1, 1)
        .enable_logging()
        .build(&valid_matrix)?;
    robust_amg.setup(&valid_matrix)?;
    println!("  Robust AMG with safety checks successful");

    // Preconditioning operation
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let mut y = vec![0.0; 5];
    robust_amg.apply(PcSide::Left, &x, &mut y)?;
    println!("  Safe preconditioning operation completed\n");

    Ok(())
}
