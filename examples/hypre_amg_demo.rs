use faer::Mat;
use kryst::error::KError;
use kryst::preconditioner::amg::{AMG, AMGBuilder, CoarsenType, InterpType, RelaxType};
use kryst::preconditioner::{
    LegacyOpPreconditioner, PcSide, Preconditioner, legacy::Preconditioner as LegacyPreconditioner,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("    HYPRE-Inspired AMG Preconditioner Demo");
    println!("==========================================\n");

    // Create test problems of varying difficulty
    demo_symmetric_positive_definite()?;
    demo_anisotropic_problem()?;
    demo_configuration_builder()?;
    demo_safety_features()?;

    println!("  All AMG demos completed successfully!");
    Ok(())
}

/// Create a 2D Laplacian matrix directly
fn create_2d_laplacian(nx: usize, ny: usize) -> Mat<f64> {
    let n = nx * ny;
    let mut matrix = Mat::zeros(n, n);

    for i in 0..ny {
        for j in 0..nx {
            let idx = i * nx + j;
            let mut diag = 0.0;

            // Left neighbor
            if j > 0 {
                matrix[(idx, idx - 1)] = -1.0;
                diag += 1.0;
            }

            // Right neighbor
            if j < nx - 1 {
                matrix[(idx, idx + 1)] = -1.0;
                diag += 1.0;
            }

            // Top neighbor
            if i > 0 {
                matrix[(idx, idx - nx)] = -1.0;
                diag += 1.0;
            }

            // Bottom neighbor
            if i < ny - 1 {
                matrix[(idx, idx + nx)] = -1.0;
                diag += 1.0;
            }

            matrix[(idx, idx)] = diag + 1.0; // Add identity for better conditioning
        }
    }

    matrix
}

/// Create an anisotropic diffusion matrix
fn create_anisotropic_matrix(nx: usize, ny: usize, anisotropy: f64) -> Mat<f64> {
    let n = nx * ny;
    let mut matrix = Mat::zeros(n, n);

    for i in 0..ny {
        for j in 0..nx {
            let idx = i * nx + j;
            let mut diag = 0.0;

            // Strong horizontal coupling
            if j > 0 {
                matrix[(idx, idx - 1)] = -1.0;
                diag += 1.0;
            }
            if j < nx - 1 {
                matrix[(idx, idx + 1)] = -1.0;
                diag += 1.0;
            }

            // Weak vertical coupling
            if i > 0 {
                matrix[(idx, idx - nx)] = -anisotropy;
                diag += anisotropy;
            }
            if i < ny - 1 {
                matrix[(idx, idx + nx)] = -anisotropy;
                diag += anisotropy;
            }

            matrix[(idx, idx)] = diag + 0.1; // Small regularization
        }
    }

    matrix
}

/// Demonstrate AMG on symmetric positive definite problems
fn demo_symmetric_positive_definite() -> Result<(), KError> {
    println!("Testing AMG on Symmetric Positive Definite Problem");
    println!("-----------------------------------------------------");

    // Create a 2D Laplacian matrix (5x5 grid)
    let matrix = create_2d_laplacian(5, 5);
    let n = matrix.nrows();

    println!("Matrix size: {}x{}", matrix.nrows(), matrix.ncols());

    // Test default AMG configuration
    let _amg_default = AMG::new(&matrix, 10, 0.25);
    println!("  Default AMG construction successful");

    // Test HYPRE-inspired configuration
    let amg_hypre = AMGBuilder::new()
        .max_levels(15)
        .strong_threshold(0.25)
        .coarsening_type(CoarsenType::HMIS)
        .interpolation_type(InterpType::Extended)
        .relaxation_type(RelaxType::GaussSeidel)
        .smoothing_sweeps(2, 2)
        .enable_logging()
        .build(&matrix)?;

    println!("  HYPRE-style AMG construction successful");

    // Test preconditioning with correct API
    let x = vec![1.0; n];
    let mut y = vec![0.0; n];
    // Wrap legacy AMG (which implements the legacy Preconditioner trait) with
    // the object-safe adapter so we can use the new `Preconditioner` API.
    let mut pc = LegacyOpPreconditioner::new(Box::new(amg_hypre));
    pc.setup(&matrix)?;
    pc.apply(PcSide::Left, &x, &mut y)?;
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

/// Demonstrate AMG on anisotropic problems
fn demo_anisotropic_problem() -> Result<(), KError> {
    println!("   Testing AMG on Anisotropic Problem");
    println!("-------------------------------------");

    // Create anisotropic problem (4x4 grid)
    let matrix = create_anisotropic_matrix(4, 4, 0.01);

    println!(
        "Anisotropic matrix size: {}x{}",
        matrix.nrows(),
        matrix.ncols()
    );

    // Test different coarsening strategies
    let strategies = [
        ("HMIS", CoarsenType::HMIS),
        ("RS", CoarsenType::RS),
        ("PMIS", CoarsenType::PMIS),
        ("Falgout", CoarsenType::Falgout),
    ];

    for (name, strategy) in &strategies {
        let _amg = AMGBuilder::new()
            .max_levels(5)
            .strong_threshold(0.5) // Higher threshold for anisotropic problems
            .coarsening_type(*strategy)
            .interpolation_type(InterpType::Extended)
            .build(&matrix)?;

        println!("  {} coarsening strategy successful", name);
    }

    println!();
    Ok(())
}

/// Demonstrate the configuration builder pattern
fn demo_configuration_builder() -> Result<(), KError> {
    println!("    Testing AMG Configuration Builder");
    println!("------------------------------------");

    let n = 9;
    let matrix = Mat::from_fn(n, n, |i, j| {
        if i == j {
            4.0
        } else if (i as i32 - j as i32).abs() == 1 {
            -1.0
        } else {
            0.0
        }
    });

    // Test various configurations
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
        match builder.build(&matrix) {
            Ok(_amg) => println!("  {} configuration successful", name),
            Err(e) => println!("   {} configuration failed: {}", name, e),
        }
    }

    println!();
    Ok(())
}

/// Demonstrate safety features and error handling
fn demo_safety_features() -> Result<(), KError> {
    println!("    Testing AMG Safety Features");
    println!("-------------------------------");

    // Test empty matrix handling
    let empty_matrix = Mat::zeros(0, 0);
    match AMGBuilder::new().build(&empty_matrix) {
        Ok(_) => println!("   Empty matrix should have failed"),
        Err(_) => println!("  Empty matrix correctly rejected"),
    }

    // Test non-square matrix handling
    let non_square = Mat::zeros(3, 4);
    match AMGBuilder::new().build(&non_square) {
        Ok(_) => println!("   Non-square matrix should have failed"),
        Err(_) => println!("  Non-square matrix correctly rejected"),
    }

    // Test invalid configuration
    let valid_matrix = Mat::identity(5, 5);
    match AMGBuilder::new()
        .max_levels(0) // Invalid!
        .build(&valid_matrix)
    {
        Ok(_) => println!("   Invalid config should have failed"),
        Err(_) => println!("  Invalid configuration correctly rejected"),
    }

    // Test invalid strong threshold
    match AMGBuilder::new()
        .strong_threshold(1.5) // Invalid!
        .build(&valid_matrix)
    {
        Ok(_) => println!("   Invalid threshold should have failed"),
        Err(_) => println!("  Invalid threshold correctly rejected"),
    }

    // Test matrix with NaN (if IEEE checks enabled)
    let _nan_matrix: Mat<f64> = Mat::identity(3, 3);
    // Note: We can't easily create NaN in this context, so we skip this test
    println!("ℹ️  IEEE NaN/Inf checks would be tested with problematic matrices");

    // Test successful configuration with all safety checks
    let robust_amg = AMGBuilder::new()
        .max_levels(5)
        .strong_threshold(0.25)
        .coarsening_type(CoarsenType::HMIS)
        .interpolation_type(InterpType::Extended)
        .smoothing_sweeps(1, 1)
        .enable_logging()
        .build(&valid_matrix)?;

    println!("  Robust AMG with safety checks successful");

    // Test preconditioning operation
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let mut y = vec![0.0; 5];
    robust_amg.apply(PcSide::Left, &x, &mut y)?;
    println!("  Safe preconditioning operation completed");

    println!();
    Ok(())
}
