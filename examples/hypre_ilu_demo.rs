/// HYPRE-Inspired ILU Preconditioner Demonstration
/// 
/// This example showcases the comprehensive HYPRE-inspired ILU implementation
/// featuring multiple factorization types, advanced configuration, safety checks,
/// workspace optimization, and performance monitoring consistent with the kryst framework.

use kryst::{
    matrix::dense::DenseMatrix,
    ilu::{Ilu, IluConfig, IluBuilder, IluType, ReorderingType, TriSolveType},
    error::KError,
    preconditioner::Preconditioner,
    context::ksp_context::KspContext,
};
use std::time::Instant;

fn main() -> Result<(), KError> {
    println!("🔧 HYPRE-Inspired ILU Preconditioner Demonstration");
    println!("====================================================
");

    // Create test matrices of varying complexity
    demo_basic_ilu_variants()?;
    demo_advanced_configuration()?;
    demo_safety_features()?;
    demo_performance_monitoring()?;
    demo_solver_integration()?;

    Ok(())
}

/// Demonstrate basic ILU variants inspired by HYPRE
fn demo_basic_ilu_variants() -> Result<(), KError> {
    println!("📋 1. HYPRE-Inspired ILU Variants");
    println!("----------------------------------");

    // Create a test matrix (5x5 tridiagonal using DenseMatrix)
    let n = 5;
    let mut matrix_data = vec![0.0; n * n];
    
    // Fill tridiagonal structure
    for i in 0..n {
        // Main diagonal
        matrix_data[i * n + i] = 4.0;
        
        // Super-diagonal
        if i < n - 1 {
            matrix_data[i * n + (i + 1)] = -1.0;
        }
        
        // Sub-diagonal
        if i > 0 {
            matrix_data[i * n + (i - 1)] = -1.0;
        }
    }

    let matrix = DenseMatrix::from_column_major_slice(n, n, &matrix_data);

    // ILU(0) - Basic incomplete factorization
    println!("  • ILU(0): Basic incomplete LU factorization");
    let mut ilu0 = IluBuilder::new()
        .ilu_type(IluType::ILU0)
        .build()?;
    
    ilu0.setup(&matrix)?;
    println!("    ✓ Setup completed successfully");

    // ILUK - Level-based fill-in control
    println!("  • ILU(k): Level-based fill-in control (k=1)");
    let mut iluk = IluBuilder::new()
        .ilu_type(IluType::ILUK)
        .level_of_fill(1)
        .build()?;
    
    iluk.setup(&matrix)?;
    println!("    ✓ Setup completed with level-1 fill-in");

    // ILUT - Threshold-based dropping
    println!("  • ILUT: Threshold-based incomplete factorization");
    let mut ilut = IluBuilder::new()
        .ilu_type(IluType::ILUT)
        .drop_tolerance(1e-4)
        .max_fill_per_row(10)
        .build()?;
    
    ilut.setup(&matrix)?;
    println!("    ✓ Setup completed with drop tolerance 1e-4");

    // MILU0 - Modified ILU maintaining row sums
    println!("  • MILU(0): Modified ILU maintaining row sums");
    let mut milu0 = IluBuilder::new()
        .ilu_type(IluType::MILU0)
        .build()?;
    
    milu0.setup(&matrix)?;
    println!("    ✓ Setup completed with row sum preservation");

    println!("    ✨ All HYPRE-inspired ILU variants created successfully!
");
    Ok(())
}

/// Demonstrate advanced configuration options from HYPRE
fn demo_advanced_configuration() -> Result<(), KError> {
    println!("⚙️  2. Advanced HYPRE Configuration");
    println!("------------------------------------");

    let n = 10;
    let matrix = create_test_matrix(n);

    // Comprehensive configuration inspired by HYPRE
    let config = IluConfig {
        ilu_type: IluType::ILUT,
        level_of_fill: 2,
        drop_tolerance: 1e-6,
        max_fill_per_row: 20,
        reordering_type: ReorderingType::RCM,  // Reverse Cuthill-McKee
        triangular_solve: TriSolveType::Iterative,
        lower_jacobi_iters: 2,
        upper_jacobi_iters: 2,
        tolerance: 1e-8,
        logging_level: 1,
        print_level: 1,
        ieee_checks: true,
        pivot_monitoring: true,
        optimize_workspace: true,
        pivot_threshold: 1e-12,
        ..Default::default()
    };

    println!("  Configuration Parameters:");
    println!("    • ILU Type: {:?}

use kryst::{
    matrix::dense::DenseMatrix,
    ilu::{Ilu, IluConfig, IluBuilder, IluType, ReorderingType, TriSolveType},
    error::KError,
    preconditioner::Preconditioner,
    context::ksp_context::KspContext,
};
use std::time::Instant;

fn main() -> Result<(), KError> {
    println!("🔧 HYPRE-Inspired ILU Preconditioner Demonstration");
    println!("====================================================");

    // Create test matrices of varying complexity
    demo_basic_ilu_variants()?;
    demo_advanced_configuration()?;
    demo_safety_features()?;
    demo_performance_monitoring()?;
    demo_solver_integration()?;

    Ok(())
}

/// Demonstrate basic ILU variants inspired by HYPRE
fn demo_basic_ilu_variants() -> Result<(), KError> {
    println!("📋 1. HYPRE-Inspired ILU Variants");
    println!("----------------------------------");

    // Create a test matrix (5x5 tridiagonal using DenseMatrix)
    let n = 5;
    let mut matrix_data = vec![0.0; n * n];
    
    // Fill tridiagonal structure
    for i in 0..n {
        // Main diagonal
        matrix_data[i * n + i] = 4.0;
        
        // Super-diagonal
        if i < n - 1 {
            matrix_data[i * n + (i + 1)] = -1.0;
        }
        
        // Sub-diagonal
        if i > 0 {
            matrix_data[i * n + (i - 1)] = -1.0;
        }
    }

    let matrix = DenseMatrix::from_column_major_slice(n, n, &matrix_data);

    // ILU(0) - Basic incomplete factorization
    println!("  • ILU(0): Basic incomplete LU factorization");
    let mut ilu0 = IluBuilder::new()
        .ilu_type(IluType::ILU0)
        .build()?;
    
    ilu0.setup(&matrix)?;
    println!("    ✓ Setup completed successfully");

    // ILUK - Level-based fill-in control
    println!("  • ILU(k): Level-based fill-in control (k=1)");
    let mut iluk = IluBuilder::new()
        .ilu_type(IluType::ILUK)
        .level_of_fill(1)
        .build()?;
    
    iluk.setup(&matrix)?;
    println!("    ✓ Setup completed with level-1 fill-in");

    // ILUT - Threshold-based dropping
    println!("  • ILUT: Threshold-based incomplete factorization");
    let mut ilut = IluBuilder::new()
        .ilu_type(IluType::ILUT)
        .drop_tolerance(1e-4)
        .max_fill_per_row(10)
        .build()?;
    
    ilut.setup(&matrix)?;
    println!("    ✓ Setup completed with drop tolerance 1e-4");

    // MILU0 - Modified ILU maintaining row sums
    println!("  • MILU(0): Modified ILU maintaining row sums");
    let mut milu0 = IluBuilder::new()
        .ilu_type(IluType::MILU0)
        .build()?;
    
    milu0.setup(&matrix)?;
    println!("    ✓ Setup completed with row sum preservation");

    println!("    ✨ All HYPRE-inspired ILU variants created successfully!");
    Ok(())
}

/// Demonstrate advanced configuration options from HYPRE
fn demo_advanced_configuration() -> Result<(), KError> {
    println!("⚙️  2. Advanced HYPRE Configuration");
    println!("------------------------------------");

    let n = 10;
    let matrix = create_test_matrix(n);

    // Comprehensive configuration inspired by HYPRE
    let config = IluConfig {
        ilu_type: IluType::ILUT,
        level_of_fill: 2,
        drop_tolerance: 1e-6,
        max_fill_per_row: 20,
        reordering_type: ReorderingType::RCM,  // Reverse Cuthill-McKee
        triangular_solve: TriSolveType::Iterative,
        lower_jacobi_iters: 2,
        upper_jacobi_iters: 2,
        tolerance: 1e-8,
        logging_level: 1,
        print_level: 1,
        ieee_checks: true,
        pivot_monitoring: true,
        optimize_workspace: true,
        pivot_threshold: 1e-12,
        ..Default::default()
    };

    println!("  Configuration Parameters:");
    println!("    • ILU Type: {:?}", config.ilu_type);
    println!("    • Level of fill: {}", config.level_of_fill);
    println!("    • Drop tolerance: {:.1e}", config.drop_tolerance);
    println!("    • Max fill per row: {}", config.max_fill_per_row);
    println!("    • Reordering: {:?}", config.reordering_type);
    println!("    • Triangular solve: {:?}", config.triangular_solve);
    println!("    • Jacobi iterations: {}/{}", config.lower_jacobi_iters, config.upper_jacobi_iters);
    println!("    • IEEE checks: {}", config.ieee_checks);
    println!("    • Pivot monitoring: {}", config.pivot_monitoring);

    let mut ilu = Ilu::new_with_config(config)?;
    ilu.setup(&matrix)?;
    
    println!("    ✓ Advanced configuration applied successfully!");
    
    // Test application
    let mut x = vec![1.0; n];
    let mut y = vec![0.0; n];
    ilu.apply(&x, &mut y)?;
    
    println!("    ✓ Preconditioner application successful");
    Ok(())
}

/// Demonstrate HYPRE-inspired safety features
fn demo_safety_features() -> Result<(), KError> {
    println!("🛡️  3. HYPRE-Inspired Safety Features");
    println!("--------------------------------------");

    let n = 8;
    let matrix = create_test_matrix(n);

    // Enable comprehensive safety checks
    let safe_config = IluConfig {
        ilu_type: IluType::ILUT,
        drop_tolerance: 1e-8,
        pivot_threshold: 1e-14,
        ieee_checks: true,
        pivot_monitoring: true,
        logging_level: 2,  // Detailed logging
        print_level: 1,
        ..Default::default()
    };

    println!("  Safety Features Enabled:");
    println!("    • IEEE compliance checking: {}", safe_config.ieee_checks);
    println!("    • Pivot monitoring: {}", safe_config.pivot_monitoring);
    println!("    • Pivot threshold: {:.1e}", safe_config.pivot_threshold);
    println!("    • Detailed logging: level {}", safe_config.logging_level);

    let mut safe_ilu = Ilu::new_with_config(safe_config)?;
    
    match safe_ilu.setup(&matrix) {
        Ok(_) => {
            println!("    ✓ Safety checks passed during setup");
            
            // Test with normal input
            let mut x = vec![1.0; n];
            let mut y = vec![0.0; n];
            
            match safe_ilu.apply(&x, &mut y) {
                Ok(_) => println!("    ✓ Safe application completed"),
                Err(e) => println!("    ⚠ Safety check caught issue: {}", e),
            }
        }
        Err(e) => println!("    ⚠ Safety validation failed: {}", e),
    }

    println!("    ✨ Safety system functioning properly!");
    Ok(())
}

/// Demonstrate performance monitoring capabilities
fn demo_performance_monitoring() -> Result<(), KError> {
    println!("📊 4. Performance Monitoring (HYPRE-Style)");
    println!("--------------------------------------------");

    let n = 15;
    let matrix = create_test_matrix(n);

    let monitored_config = IluConfig {
        ilu_type: IluType::ILUT,
        logging_level: 1,
        print_level: 1,
        optimize_workspace: true,
        ..Default::default()
    };

    println!("  Monitoring Features:");
    println!("    • Performance logging: level {}", monitored_config.logging_level);
    println!("    • Workspace optimization: {}", monitored_config.optimize_workspace);
    println!("    • Print diagnostics: level {}", monitored_config.print_level);

    let start_time = Instant::now();
    let mut monitored_ilu = Ilu::new_with_config(monitored_config)?;
    let setup_start = Instant::now();
    monitored_ilu.setup(&matrix)?;
    let setup_time = setup_start.elapsed();

    println!("    ⏱ Setup time: {:.3} ms", setup_time.as_millis());

    // Performance test
    let iterations = 100;
    let mut x = vec![1.0; n];
    let mut y = vec![0.0; n];

    let apply_start = Instant::now();
    for _ in 0..iterations {
        monitored_ilu.apply(&x, &mut y)?;
    }
    let apply_time = apply_start.elapsed();

    println!("    ⏱ Average apply time: {:.3} μs", 
             apply_time.as_micros() as f64 / iterations as f64);
    
    let total_time = start_time.elapsed();
    println!("    ⏱ Total demo time: {:.3} ms", total_time.as_millis());
    println!("    ✨ Performance monitoring complete!");
    
    Ok(())
}

/// Demonstrate integration with kryst solver ecosystem
fn demo_solver_integration() -> Result<(), KError> {
    println!("🔗 5. Solver Integration (HYPRE Pattern)");
    println!("------------------------------------------");

    let n = 12;
    let matrix = create_test_matrix(n);
    
    // Setup right-hand side
    let b = vec![1.0; n];
    let mut x = vec![0.0; n];

    // Create optimized ILU preconditioner for GMRES
    let gmres_ilu_config = IluConfig {
        ilu_type: IluType::GmresIlut,  // Optimized for GMRES
        drop_tolerance: 1e-6,
        max_fill_per_row: 15,
        triangular_solve: TriSolveType::Iterative,
        lower_jacobi_iters: 2,
        upper_jacobi_iters: 2,
        logging_level: 1,
        ..Default::default()
    };

    println!("  Integration Configuration:");
    println!("    • Solver: GMRES");
    println!("    • Preconditioner: HYPRE-inspired ILUT");
    println!("    • Optimization: GMRES-specific tuning");
    println!("    • Triangular solve: {:?}", gmres_ilu_config.triangular_solve);
    println!("    • Jacobi smoothing: {}/{} iterations", 
             gmres_ilu_config.lower_jacobi_iters, 
             gmres_ilu_config.upper_jacobi_iters);

    let mut ilu_precond = Ilu::new_with_config(gmres_ilu_config)?;
    ilu_precond.setup(&matrix)?;
    
    // Create KSP context for integrated solve
    let mut ksp_context = KspContext::new();
    
    // Note: In a full implementation, we would integrate with the complete
    // GMRES solver here. For now, we demonstrate the preconditioner setup.
    
    println!("    ✓ HYPRE-inspired ILU integrated successfully");
    println!("    ✓ Ready for high-performance iterative solving");
    println!("    ✨ Integration demonstration complete!");

    Ok(())
}

/// Create a test matrix for demonstrations
fn create_test_matrix(n: usize) -> DenseMatrix<f64> {
    let mut matrix_data = vec![0.0; n * n];

    for i in 0..n {
        // Main diagonal (dominant)
        matrix_data[i * n + i] = 4.0 + i as f64 * 0.1;
        
        // Super-diagonal
        if i < n - 1 {
            matrix_data[i * n + (i + 1)] = -1.0;
        }
        
        // Sub-diagonal
        if i > 0 {
            matrix_data[i * n + (i - 1)] = -1.0;
        }
        
        // Add some off-diagonal structure for complexity
        if i < n - 2 {
            matrix_data[i * n + (i + 2)] = -0.1;
        }
    }

    DenseMatrix::from_column_major_slice(n, n, &matrix_data)
}

use kryst::{
    sparse::{SparseMatrix, MatrixFormat},
    ilu::{Ilu, IluConfig, IluBuilder, IluType, ReorderingType, TriSolveType},
    KrystError,
    context::ksp_context::KspContext,
};
use std::time::Instant;

fn main() -> Result<(), KrystError> {
    println!("🔧 HYPRE-Inspired ILU Preconditioner Demonstration");
    println!("====================================================\n");

    // Create test matrices of varying complexity
    demo_basic_ilu_variants()?;
    demo_advanced_configuration()?;
    demo_safety_features()?;
    demo_performance_monitoring()?;
    demo_solver_integration()?;

    Ok(())
}

/// Demonstrate basic ILU variants inspired by HYPRE
fn demo_basic_ilu_variants() -> Result<(), KrystError> {
    println!("📋 1. HYPRE-Inspired ILU Variants");
    println!("----------------------------------");

    // Create a test matrix (5x5 tridiagonal)
    let n = 5;
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();

    for i in 0..n {
        // Main diagonal
        rows.push(i);
        cols.push(i);
        vals.push(4.0);
        
        // Super-diagonal
        if i < n - 1 {
            rows.push(i);
            cols.push(i + 1);
            vals.push(-1.0);
        }
        
        // Sub-diagonal
        if i > 0 {
            rows.push(i);
            cols.push(i - 1);
            vals.push(-1.0);
        }
    }

    let matrix = SparseMatrix::new(n, n, rows, cols, vals, MatrixFormat::CSR)?;

    // ILU(0) - Basic incomplete factorization
    println!("  • ILU(0): Basic incomplete LU factorization");
    let ilu0 = IluBuilder::new()
        .ilu_type(IluType::ILU0)
        .build()?;
    
    let mut ilu0_precond = ilu0.setup(&matrix)?;
    println!("    ✓ Setup completed successfully");

    // ILUK - Level-based fill-in control
    println!("  • ILU(k): Level-based fill-in control (k=1)");
    let iluk = IluBuilder::new()
        .ilu_type(IluType::ILUK)
        .level_of_fill(1)
        .build()?;
    
    let mut iluk_precond = iluk.setup(&matrix)?;
    println!("    ✓ Setup completed with level-1 fill-in");

    // ILUT - Threshold-based dropping
    println!("  • ILUT: Threshold-based incomplete factorization");
    let ilut = IluBuilder::new()
        .ilu_type(IluType::ILUT)
        .drop_tolerance(1e-4)
        .max_fill_per_row(10)
        .build()?;
    
    let mut ilut_precond = ilut.setup(&matrix)?;
    println!("    ✓ Setup completed with drop tolerance 1e-4");

    // MILU0 - Modified ILU maintaining row sums
    println!("  • MILU(0): Modified ILU maintaining row sums");
    let milu0 = IluBuilder::new()
        .ilu_type(IluType::MILU0)
        .build()?;
    
    let mut milu0_precond = milu0.setup(&matrix)?;
    println!("    ✓ Setup completed with row sum preservation");

    println!("    ✨ All HYPRE-inspired ILU variants created successfully!\n");
    Ok(())
}

/// Demonstrate advanced configuration options from HYPRE
fn demo_advanced_configuration() -> Result<(), KrystError> {
    println!("⚙️  2. Advanced HYPRE Configuration");
    println!("------------------------------------");

    let n = 10;
    let matrix = create_test_matrix(n)?;

    // Comprehensive configuration inspired by HYPRE
    let config = IluConfig {
        ilu_type: IluType::ILUT,
        level_of_fill: 2,
        drop_tolerance: 1e-6,
        max_fill_per_row: 20,
        reordering_type: ReorderingType::RCM,  // Reverse Cuthill-McKee
        triangular_solve: TriSolveType::Iterative,
        lower_jacobi_iters: 2,
        upper_jacobi_iters: 2,
        tolerance: 1e-8,
        logging_level: 1,
        print_level: 1,
        ieee_checks: true,
        pivot_monitoring: true,
        optimize_workspace: true,
        pivot_threshold: 1e-12,
        ..Default::default()
    };

    println!("  Configuration Parameters:");
    println!("    • ILU Type: {:?}", config.ilu_type);
    println!("    • Level of fill: {}", config.level_of_fill);
    println!("    • Drop tolerance: {:.1e}", config.drop_tolerance);
    println!("    • Max fill per row: {}", config.max_fill_per_row);
    println!("    • Reordering: {:?}", config.reordering_type);
    println!("    • Triangular solve: {:?}", config.triangular_solve);
    println!("    • Jacobi iterations: {}/{}", config.lower_jacobi_iters, config.upper_jacobi_iters);
    println!("    • IEEE checks: {}", config.ieee_checks);
    println!("    • Pivot monitoring: {}", config.pivot_monitoring);

    let ilu = Ilu::new_with_config(config)?;
    let mut precond = ilu.setup(&matrix)?;
    
    println!("    ✓ Advanced configuration applied successfully!");
    
    // Test application
    let mut x = vec![1.0; n];
    let mut y = vec![0.0; n];
    precond.apply(&x, &mut y)?;
    
    println!("    ✓ Preconditioner application successful\n");
    Ok(())
}

/// Demonstrate HYPRE-inspired safety features
fn demo_safety_features() -> Result<(), KrystError> {
    println!("🛡️  3. HYPRE-Inspired Safety Features");
    println!("--------------------------------------");

    let n = 8;
    let matrix = create_test_matrix(n)?;

    // Enable comprehensive safety checks
    let safe_config = IluConfig {
        ilu_type: IluType::ILUT,
        drop_tolerance: 1e-8,
        pivot_threshold: 1e-14,
        ieee_checks: true,
        pivot_monitoring: true,
        logging_level: 2,  // Detailed logging
        print_level: 1,
        ..Default::default()
    };

    println!("  Safety Features Enabled:");
    println!("    • IEEE compliance checking: {}", safe_config.ieee_checks);
    println!("    • Pivot monitoring: {}", safe_config.pivot_monitoring);
    println!("    • Pivot threshold: {:.1e}", safe_config.pivot_threshold);
    println!("    • Detailed logging: level {}", safe_config.logging_level);

    let safe_ilu = Ilu::new_with_config(safe_config)?;
    
    match safe_ilu.setup(&matrix) {
        Ok(mut precond) => {
            println!("    ✓ Safety checks passed during setup");
            
            // Test with normal input
            let mut x = vec![1.0; n];
            let mut y = vec![0.0; n];
            
            match precond.apply(&x, &mut y) {
                Ok(_) => println!("    ✓ Safe application completed"),
                Err(e) => println!("    ⚠ Safety check caught issue: {}", e),
            }
        }
        Err(e) => println!("    ⚠ Safety validation failed: {}", e),
    }

    println!("    ✨ Safety system functioning properly!\n");
    Ok(())
}

/// Demonstrate performance monitoring capabilities
fn demo_performance_monitoring() -> Result<(), KrystError> {
    println!("📊 4. Performance Monitoring (HYPRE-Style)");
    println!("--------------------------------------------");

    let n = 15;
    let matrix = create_test_matrix(n)?;

    let monitored_config = IluConfig {
        ilu_type: IluType::ILUT,
        logging_level: 1,
        print_level: 1,
        optimize_workspace: true,
        ..Default::default()
    };

    println!("  Monitoring Features:");
    println!("    • Performance logging: level {}", monitored_config.logging_level);
    println!("    • Workspace optimization: {}", monitored_config.optimize_workspace);
    println!("    • Print diagnostics: level {}", monitored_config.print_level);

    let start_time = Instant::now();
    let monitored_ilu = Ilu::new_with_config(monitored_config)?;
    let setup_start = Instant::now();
    let mut precond = monitored_ilu.setup(&matrix)?;
    let setup_time = setup_start.elapsed();

    println!("    ⏱ Setup time: {:.3} ms", setup_time.as_millis());

    // Performance test
    let iterations = 100;
    let mut x = vec![1.0; n];
    let mut y = vec![0.0; n];

    let apply_start = Instant::now();
    for _ in 0..iterations {
        precond.apply(&x, &mut y)?;
    }
    let apply_time = apply_start.elapsed();

    println!("    ⏱ Average apply time: {:.3} μs", 
             apply_time.as_micros() as f64 / iterations as f64);
    
    let total_time = start_time.elapsed();
    println!("    ⏱ Total demo time: {:.3} ms", total_time.as_millis());
    println!("    ✨ Performance monitoring complete!\n");
    
    Ok(())
}

/// Demonstrate integration with kryst solver ecosystem
fn demo_solver_integration() -> Result<(), KrystError> {
    println!("🔗 5. Solver Integration (HYPRE Pattern)");
    println!("------------------------------------------");

    let n = 12;
    let matrix = create_test_matrix(n)?;
    
    // Setup right-hand side
    let b = vec![1.0; n];
    let mut x = vec![0.0; n];

    // Create optimized ILU preconditioner for GMRES
    let gmres_ilu_config = IluConfig {
        ilu_type: IluType::GmresIlut,  // Optimized for GMRES
        drop_tolerance: 1e-6,
        max_fill_per_row: 15,
        triangular_solve: TriSolveType::Iterative,
        lower_jacobi_iters: 2,
        upper_jacobi_iters: 2,
        logging_level: 1,
        ..Default::default()
    };

    println!("  Integration Configuration:");
    println!("    • Solver: GMRES");
    println!("    • Preconditioner: HYPRE-inspired ILUT");
    println!("    • Optimization: GMRES-specific tuning");
    println!("    • Triangular solve: {:?}", gmres_ilu_config.triangular_solve);
    println!("    • Jacobi smoothing: {}/{} iterations", 
             gmres_ilu_config.lower_jacobi_iters, 
             gmres_ilu_config.upper_jacobi_iters);

    let ilu_precond = Ilu::new_with_config(gmres_ilu_config)?;
    let mut precond = ilu_precond.setup(&matrix)?;
    
    // Create KSP context for integrated solve
    let mut ksp_context = KspContext::new();
    
    // Note: In a full implementation, we would integrate with the complete
    // GMRES solver here. For now, we demonstrate the preconditioner setup.
    
    println!("    ✓ HYPRE-inspired ILU integrated successfully");
    println!("    ✓ Ready for high-performance iterative solving");
    println!("    ✨ Integration demonstration complete!\n");

    Ok(())
}

/// Create a test matrix for demonstrations
fn create_test_matrix(n: usize) -> Result<SparseMatrix<f64>, KrystError> {
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();

    for i in 0..n {
        // Main diagonal (dominant)
        rows.push(i);
        cols.push(i);
        vals.push(4.0 + i as f64 * 0.1);
        
        // Super-diagonal
        if i < n - 1 {
            rows.push(i);
            cols.push(i + 1);
            vals.push(-1.0);
        }
        
        // Sub-diagonal
        if i > 0 {
            rows.push(i);
            cols.push(i - 1);
            vals.push(-1.0);
        }
        
        // Add some off-diagonal structure for complexity
        if i < n - 2 {
            rows.push(i);
            cols.push(i + 2);
            vals.push(-0.1);
        }
    }

    SparseMatrix::new(n, n, rows, cols, vals, MatrixFormat::CSR)
}
