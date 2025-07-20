<!--
    kryst: PETSc-style Krylov solvers and preconditioners for Rust.
    This README describes the main features, usage, and documentation pointers.
-->

# kryst

[![Crates.io](https://img.shields.io/crates/v/kryst.svg)](https://crates.io/crates/kryst)
[![Documentation](https://docs.rs/kryst/badge.svg)](https://docs.rs/kryst)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

High-performance Krylov subspace and preconditioned iterative solvers for dense and sparse linear systems, with advanced preconditioning strategies and automated parameter optimization.

## Features

### Iterative Solvers
- **Krylov Methods**: CG, PCG, GMRES, FGMRES, BiCGStab, CGS, QMR, TFQMR, MINRES, CGNR
- **Direct Methods**: LU and QR factorization via PREONLY solver type
- **Parallel Support**: Shared-memory (Rayon) and distributed-memory (MPI) parallelism

### Preconditioners

#### Basic Preconditioners
- **Jacobi**: Diagonal scaling preconditioner
- **Block Jacobi**: Block-wise diagonal preconditioning
- **SOR/SSOR**: Successive Over-Relaxation methods
- **None**: No preconditioning (identity)

#### Incomplete Factorizations
- **ILU(0)**: Zero fill-in incomplete LU factorization
- **ILU(k)**: Incomplete LU with k levels of fill-in
- **ILUT**: Threshold-based incomplete LU factorization
- **ILUTP**: ILUT with partial pivoting
- **ILUP**: Incomplete LU with partial pivoting

#### Advanced Preconditioners
- **Chebyshev**: Enhanced polynomial preconditioning with eigenvalue estimation
- **AMG**: Algebraic Multigrid with configurable smoothing parameters
- **ASM**: Additive Schwarz Method (domain decomposition)
- **Approximate Inverse**: SPAI-type approximate inverse preconditioners

#### Composite Preconditioning (Phase III)
- **PC-Chaining**: Sequential application of multiple preconditioners
- **Enhanced Chebyshev**: Matrix-aware polynomial preconditioning with automatic eigenvalue bounds
- **Smoothed AMG**: Configurable pre- and post-smoothing parameters

### Monitoring & Automation (Phase IV)
- **Iteration Monitoring**: Real-time convergence tracking and analysis
- **Parameter Tuning**: Automated optimization with grid search
- **Data Export**: CSV/JSON output for external analysis
- **Performance Metrics**: Time-based optimization with configurable timeouts

### Architecture
- **PETSc-style API**: Unified KSP context for runtime solver selection
- **Command-line Options**: Complete options database with 50+ parameters
- **Trait-based Design**: Extensible for custom matrices and preconditioners
- **Memory Efficiency**: In-place operations and configurable workspace management
- **High Performance**: Optimized inner kernels with SIMD and parallelization

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
kryst = "1.0"
```

### Feature Flags

```toml
[features]
default = ["rayon", "logging"]  # Shared-memory parallelism + monitoring
rayon = ["dep:rayon"]           # Rayon-based parallel execution  
mpi = ["dep:mpi"]              # Distributed-memory parallelism via MPI
logging = ["dep:log"]          # Iteration monitoring and profiling
```

## Quick Start

### Basic Usage with KspContext (Recommended)

```rust
use kryst::{KspContext, SolverType, PcType};
use faer::Mat;

// Create a 100x100 test system
let n = 100;
let matrix = Mat::<f64>::from_fn(n, n, |i, j| {
    if i == j { 4.0 } else if (i as i32 - j as i32).abs() == 1 { -1.0 } else { 0.0 }
});
let rhs = vec![1.0; n];
let mut solution = vec![0.0; n];

// Configure solver and preconditioner
let mut ksp = KspContext::new();
ksp.set_type(SolverType::Gmres).unwrap()
   .set_pc_type(PcType::Jacobi).unwrap();
ksp.rtol = 1e-8;
ksp.maxits = 1000;

// Solve the system
ksp.setup(&matrix, n).unwrap();
let stats = ksp.solve(&matrix, &rhs, &mut solution).unwrap();
println!("Converged in {} iterations with residual {:.2e}", 
         stats.iterations, stats.residual_norm);
```

### Advanced Features: Composite Preconditioning

```rust
use kryst::{KspContext, SolverType, PcOptions};

let mut ksp = KspContext::new();
ksp.set_type(SolverType::Cg).unwrap();

// Use PC-chaining for composite preconditioning
let mut pc_opts = PcOptions::default();
pc_opts.pc_chain = Some("jacobi,chebyshev".to_string());
pc_opts.chebyshev_degree = Some(5);
ksp.set_pc_options(pc_opts);

ksp.setup(&matrix, n).unwrap();
let stats = ksp.solve(&matrix, &rhs, &mut solution).unwrap();
```

### Enhanced AMG with Smoothing

```rust
use kryst::{KspContext, SolverType, PcType, PcOptions};

let mut ksp = KspContext::new();
ksp.set_type(SolverType::Gmres).unwrap()
   .set_pc_type(PcType::Amg).unwrap();

// Configure AMG smoothing parameters
let mut pc_opts = PcOptions::default();
pc_opts.amg_levels = Some(4);
pc_opts.amg_strength_threshold = Some(0.25);
pc_opts.amg_nu_pre = Some(2);   // Pre-smoothing steps
pc_opts.amg_nu_post = Some(1);  // Post-smoothing steps
ksp.set_pc_options(pc_opts);

ksp.setup(&matrix, n).unwrap();
let stats = ksp.solve(&matrix, &rhs, &mut solution).unwrap();
```

### Iteration Monitoring and Analysis

```rust
use kryst::{IterationMonitor, ParameterTuner};
use std::time::Duration;

// Monitor convergence behavior
let mut monitor = IterationMonitor::new();
// In practice, integrate monitor with solver iteration callbacks

// Automated parameter tuning
let mut tuner = ParameterTuner::new();
tuner.set_solver_types(vec![SolverType::Cg, SolverType::Gmres]);
tuner.set_pc_types(vec![PcType::Jacobi, PcType::Chebyshev, PcType::Amg]);
tuner.set_tolerances(vec![1e-6, 1e-8]);
tuner.set_max_config_time(Duration::from_secs(30));

let (best_config, all_results) = tuner.tune_parameters(&matrix, &rhs, 5).unwrap();
println!("Best configuration: {:?}", best_config);
```

### Command-line Interface (PETSc-style)

```rust
use kryst::{parse_all_options, KspContext};

// Parse command-line options
let args: Vec<String> = std::env::args().collect();
let (ksp_opts, pc_opts) = parse_all_options(&args).unwrap();

// Configure from options  
let mut ksp = KspContext::new();
ksp.set_from_all_options(&ksp_opts, &pc_opts).unwrap();
ksp.setup(&matrix, n).unwrap();
let stats = ksp.solve(&matrix, &rhs, &mut solution).unwrap();
```

Run your program with PETSc-style options:
```bash
# Basic solver configuration
./my_program -ksp_type gmres -ksp_rtol 1e-8 -pc_type jacobi

# Direct solvers
./my_program -ksp_type preonly -pc_type lu     # Direct LU solver  
./my_program -ksp_type preonly -pc_type qr     # Direct QR solver

# Advanced preconditioning
./my_program -ksp_type cg -pc_type amg -amg_nu_pre 2 -amg_nu_post 1
./my_program -ksp_type gmres -pc_chain "jacobi,chebyshev" -chebyshev_degree 5

# Show all available options
./my_program -help
```

## Supported Command-line Options

### KSP (Krylov Solver) Options
- `-ksp_type <solver>` - Solver type: `cg`, `pcg`, `gmres`, `fgmres`, `bicgstab`, `cgs`, `qmr`, `tfqmr`, `minres`, `cgnr`, `preonly`
- `-ksp_rtol <float>` - Relative convergence tolerance (default: 1e-6)
- `-ksp_atol <float>` - Absolute convergence tolerance (default: 1e-12)
- `-ksp_dtol <float>` - Divergence tolerance (default: 1e3)
- `-ksp_max_it <int>` - Maximum number of iterations (default: 1000)
- `-ksp_gmres_restart <int>` - GMRES restart parameter (default: 50)
- `-ksp_pc_side <side>` - Preconditioning side: `left`, `right`, `symmetric`

### PC (Preconditioner) Options

#### Basic Preconditioner Options
- `-pc_type <pc>` - Preconditioner type: `jacobi`, `blockjacobi`, `sor`, `none`

#### Incomplete Factorization Options
- `-pc_type <pc>` - ILU variants: `ilu0`, `ilu`, `ilut`, `ilutp`, `ilup`
- `-pc_ilu_levels <int>` - ILU fill levels (default: 0)
- `-pc_ilut_drop_tol <float>` - ILUT drop tolerance (default: 1e-3)
- `-pc_ilut_max_fill <int>` - ILUT maximum fill per row (default: 10)

#### Enhanced Preconditioner Options (Phase III)
- `-pc_type chebyshev` - Enhanced Chebyshev with eigenvalue estimation
- `-chebyshev_degree <int>` - Polynomial degree (default: 3)
- `-pc_type amg` - Algebraic multigrid with smoothing control
- `-amg_levels <int>` - Number of AMG levels (default: 4)
- `-amg_strength_threshold <float>` - Strong connection threshold (default: 0.25)
- `-amg_nu_pre <int>` - Pre-smoothing steps (default: 1)
- `-amg_nu_post <int>` - Post-smoothing steps (default: 1)

#### Composite Preconditioning Options
- `-pc_chain <string>` - Sequential preconditioner chain (e.g., "jacobi,chebyshev")
- `-pc_type asm` - Additive Schwarz Method
- `-pc_type approxinverse` - Approximate inverse preconditioner

#### Domain Decomposition Options
- `-asm_overlap <int>` - ASM subdomain overlap (default: 1)
- `-asm_type <type>` - ASM variant: `restrict`, `interpolate`, `basic`

### Usage Examples
```bash
# Enhanced Chebyshev preconditioning
-ksp_type cg -pc_type chebyshev -chebyshev_degree 6

# AMG with custom smoothing
-ksp_type gmres -pc_type amg -amg_nu_pre 2 -amg_nu_post 1

# Composite preconditioning (PC-chaining)  
-ksp_type cg -pc_chain "jacobi,chebyshev" -chebyshev_degree 4

# High-accuracy direct solve
-ksp_type preonly -pc_type lu

# BiCGStab with threshold ILU
-ksp_type bicgstab -pc_type ilut -pc_ilut_drop_tol 1e-4

# GMRES with additive Schwarz
-ksp_type gmres -pc_type asm -asm_overlap 2
```

## Monitoring and Automation

### Iteration Monitoring (Phase IV)

Track solver convergence with real-time monitoring:

```rust
use kryst::{IterationMonitor, KspContext, SolverType, PcType};

// Create and configure monitor
let mut monitor = IterationMonitor::new();

// Configure solver
let mut ksp = KspContext::new();
ksp.set_type(SolverType::Gmres).unwrap()
   .set_pc_type(PcType::Jacobi).unwrap();

// Solve with monitoring
ksp.setup(&matrix, n).unwrap();
let stats = ksp.solve(&matrix, &rhs, &mut solution).unwrap();

// Record iteration data (integrate with solver callbacks in practice)
for i in 0..stats.iterations {
    let residual = /* get residual norm */;
    monitor.record_iteration(i as u32, residual);
}

// Analyze convergence
let stats = monitor.get_statistics();
println!("Mean convergence rate: {:.4}", stats.mean_convergence_rate);
println!("Convergence variance: {:.4}", stats.convergence_variance);

// Export data for analysis
monitor.write_to_csv("convergence_history.csv").unwrap();
```

### Automated Parameter Tuning (Phase IV)

Optimize solver/preconditioner combinations automatically:

```rust
use kryst::{ParameterTuner, SolverType, PcType};
use std::time::Duration;

let mut tuner = ParameterTuner::new();

// Configure search space
tuner.set_solver_types(vec![SolverType::Cg, SolverType::Gmres, SolverType::BiCgStab]);
tuner.set_pc_types(vec![PcType::Jacobi, PcType::Chebyshev, PcType::Amg]);
tuner.set_tolerances(vec![1e-6, 1e-8, 1e-10]);
tuner.set_max_config_time(Duration::from_secs(60));

// Add PC-chain configurations for composite preconditioning
tuner.add_pc_chain_config("jacobi,chebyshev");
tuner.add_pc_chain_config("sor,amg");

// Run automated tuning
let (best_config, all_results) = tuner.tune_parameters(&matrix, &rhs, 10).unwrap();

println!("Best configuration found:");
println!("  Solver: {:?}", best_config.solver_type);
println!("  Preconditioner: {:?}", best_config.pc_type);
println!("  Tolerance: {:.2e}", best_config.tolerance);
if let Some(chain) = &best_config.pc_chain {
    println!("  PC Chain: {}", chain);
}

// Export results for further analysis
tuner.export_results(&all_results, "tuning_results.json").unwrap();
```

### Advanced Monitoring Features

```rust
// Detect convergence stagnation
if monitor.detect_stagnation(10, 1e-2) {
    println!("Warning: Convergence appears to have stagnated");
}

// Get convergence rate analysis
let rate_analysis = monitor.analyze_convergence_rate();
match rate_analysis {
    Some((rate, r_squared)) => {
        println!("Linear convergence rate: {:.4} (R²: {:.4})", rate, r_squared);
    }
    None => println!("Insufficient data for convergence analysis"),
}

// Set up real-time monitoring callbacks
let mut ksp = KspContext::new();
ksp.add_monitor(|iter, residual| {
    println!("Iteration {}: residual = {:.3e}", iter, residual);
    
    // Custom termination criteria
    if iter > 100 && residual < 1e-10 {
        // Could trigger early termination
    }
});
```

### Profiling and Performance Analysis

Enable detailed timing and performance information:

```toml
[dependencies]
kryst = { version = "1.0", features = ["logging"] }
```

Run with environment variables for detailed profiling:

```bash
# Trace-level logging shows detailed stage timing
RUST_LOG=trace cargo run --features=logging

# Debug-level shows major operations  
RUST_LOG=debug cargo run --features=logging

# Info-level shows high-level progress
RUST_LOG=info cargo run --features=logging
```

Profiling output includes:
- **KSPSetup**: Preconditioner setup and workspace allocation timing
- **KSPSolve**: Complete solve time breakdown
- **PCSetup**: Individual preconditioner setup timing  
- **WorkspaceAllocation**: Memory allocation timing
- **MatVec**: Matrix-vector product timing
- **PCApply**: Preconditioner application timing

## Solver Algorithms

### Krylov Methods
- **CG**: Conjugate Gradient for symmetric positive definite systems
- **PCG**: Preconditioned Conjugate Gradient
- **GMRES**: Generalized Minimal Residual with restart
- **FGMRES**: Flexible GMRES for variable preconditioning
- **BiCGStab**: BiConjugate Gradient Stabilized for nonsymmetric systems
- **CGS**: Conjugate Gradient Squared
- **QMR**: Quasi-Minimal Residual method
- **TFQMR**: Transpose-Free QMR
- **MINRES**: Minimal Residual for symmetric indefinite systems
- **CGNR**: Conjugate Gradient on the Normal Equations

### Direct Methods
- **PREONLY**: Single-step direct solve using LU or QR factorization
- Supports both `-pc_type lu` and `-pc_type qr`
- Ideal for well-conditioned systems where direct methods are preferred

## Preconditioner Details

### Basic Preconditioners
- **Jacobi**: Diagonal scaling `M⁻¹ = diag(A)⁻¹`
- **Block Jacobi**: Block-wise diagonal preconditioning with configurable block sizes
- **SOR/SSOR**: Successive Over-Relaxation with configurable relaxation parameter
- **None**: Identity preconditioning (no preconditioning)

### Incomplete Factorizations
- **ILU(0)**: Zero fill-in incomplete LU factorization
- **ILU(k)**: Incomplete LU with k levels of fill-in
- **ILUT**: ILU with threshold-based dropping strategy
- **ILUTP**: ILUT with partial pivoting for numerical stability
- **ILUP**: Incomplete LU with partial pivoting

### Advanced Preconditioners

#### Enhanced Chebyshev (Phase III)
- **Matrix-aware**: Automatic eigenvalue bound estimation using power iteration
- **Configurable Degree**: Polynomial degree optimization
- **Storage Efficient**: Reuses matrix storage for eigenvalue computation

#### Enhanced AMG (Phase III)
- **Smoothed Multigrid**: Configurable pre- and post-smoothing parameters
- **Adaptive Coarsening**: Automatic grid hierarchy construction
- **Strength Threshold**: Customizable strong connection criteria

#### Composite Preconditioning (Phase III)
- **PC-Chaining**: Sequential application of multiple preconditioners
- **Flexible Combinations**: Mix any preconditioner types (e.g., "jacobi,amg,chebyshev")
- **Automatic Setup**: Transparent handling of composite preconditioner construction

### Domain Decomposition
- **ASM**: Additive Schwarz Method with configurable overlap
- **Approximate Inverse**: SPAI-type sparse approximate inverse

## Performance Features

### Parallelization
- **Shared Memory**: Rayon-based parallel execution for matrix operations and preconditioner application
- **Distributed Memory**: MPI support for distributed linear algebra operations (via mpi feature)
- **SIMD Optimization**: Leverages hardware acceleration through optimized inner kernels via faer
- **Parallel Preconditioners**: Thread-safe preconditioner application with work stealing

### Memory Management
- **In-place Operations**: Minimizes memory allocations during iteration
- **Workspace Reuse**: Preallocated workspace vectors for Krylov methods
- **Block Operations**: Efficient cache usage through blocked algorithms
- **Sparse Patterns**: Memory-efficient storage for sparse matrices and preconditioners

### Algorithm Optimizations
- **Eigenvalue Estimation**: Fast power iteration for Chebyshev eigenvalue bounds
- **Adaptive Restart**: GMRES restart optimization based on convergence behavior
- **Early Termination**: Configurable stopping criteria with multiple tolerance options
- **Matrix Preprocessing**: Reordering and scaling for improved conditioning

## Matrix Support

### Dense Matrices
- Full support via `faer::Mat<T>` integration
- Optimized BLAS-level operations
- Support for f32, f64 precision
- Efficient dense matrix-vector products

### Sparse Matrices  
- Custom CSR format implementation
- Efficient sparse matrix-vector products
- Pattern-based optimization for preconditioners
- Memory-efficient storage with configurable sparsity patterns

### Matrix-Free Methods
- Trait-based `MatVec` interface for custom matrix implementations
- Support for implicit matrix representations
- Easy integration of matrix-free operators
- Efficient for PDE discretizations and other structured problems

## Examples and Demonstrations

The library includes comprehensive demonstration programs:

### Basic Usage Examples
```bash
# Options and CLI interface demonstration
cargo run --example options_demo -- -ksp_type gmres -pc_type jacobi -ksp_rtol 1e-8

# Direct solver usage
cargo run --example dense_direct

# Setup and workspace reuse patterns
cargo run --example setup_reuse_demo
```

### Advanced Feature Examples  
```bash
# Convergence behavior analysis
cargo run --example convergence_demo

# Matrix market file I/O (auto-generates test data if files not found)
cargo run --example matrix_market_demo

# Iteration monitoring demonstration
cargo run --example monitor

# MPI parallel examples (requires MPI)
mpirun -n 4 cargo run --example mpi_parallel_demo --features mpi
mpirun -n 2 cargo run --example mpi_amg_gmres_demo --features mpi
```

**Note**: Large Matrix Market example files (*.mtx) are excluded from the published crate to stay within size limits. The `matrix_market_demo` example will auto-generate test data if the example files are not found. For the complete Matrix Market example files, clone the repository from GitHub.

### Command-line Examples
```bash
# Enhanced Chebyshev preconditioning
cargo run --example options_demo -- -ksp_type cg -pc_type chebyshev -chebyshev_degree 6

# AMG with custom smoothing parameters
cargo run --example options_demo -- -ksp_type gmres -pc_type amg -amg_nu_pre 3 -amg_nu_post 2

# Composite preconditioning with PC-chaining
cargo run --example options_demo -- -ksp_type cg -pc_chain "jacobi,chebyshev" -chebyshev_degree 4

# High-precision direct solve
cargo run --example options_demo -- -ksp_type preonly -pc_type lu

# Complex preconditioner combinations
cargo run --example options_demo -- -ksp_type fgmres -pc_type ilut -pc_ilut_drop_tol 1e-5
```

## Benchmarks and Performance

Performance benchmarks are available via:

```bash
cargo bench
```

Benchmark categories include:
- **Solver Comparison**: GMRES vs BiCGStab vs CG performance on various problems
- **Preconditioner Effectiveness**: Impact of different preconditioners on convergence
- **Direct vs Iterative**: Performance comparison for different problem sizes
- **Parallel Scaling**: Shared-memory (Rayon) and distributed-memory (MPI) performance
- **Phase III Features**: PC-chaining and enhanced preconditioning performance
- **Memory Usage**: Workspace allocation and memory efficiency analysis

Sample benchmark results (varies by system and problem):
```
solver_comparison/gmres    time: 45.2 ms  (convergence: 23 iterations)
solver_comparison/bicgstab time: 38.7 ms  (convergence: 31 iterations)  
solver_comparison/cg       time: 22.1 ms  (convergence: 18 iterations)
pc_effectiveness/jacobi    time: 156 ms   (convergence: 89 iterations)
pc_effectiveness/amg       time: 67.3 ms  (convergence: 12 iterations)
pc_chaining/jacobi+cheby   time: 43.8 ms  (convergence: 15 iterations)
```

## Custom Extensions

### Custom Solvers
```rust
use kryst::{LinearSolver, MatVec, Preconditioner, SolveStats, KError};

struct MyCustomSolver {
    tolerance: f64,
    max_iterations: usize,
}

impl<M, V> LinearSolver<M, V> for MyCustomSolver 
where 
    M: MatVec<V>,
    V: Clone,
{
    fn solve(
        &mut self, 
        matrix: &M, 
        preconditioner: Option<&dyn Preconditioner<M, V>>, 
        rhs: &V, 
        solution: &mut V
    ) -> Result<SolveStats, KError> {
        // Custom solver implementation
        // Return solve statistics
        Ok(SolveStats {
            iterations: 0,
            residual_norm: 0.0,
            converged: true,
        })
    }
}
```

### Custom Preconditioners
```rust
use kryst::{Preconditioner, PcSide, KError};

struct MyCustomPreconditioner {
    // Preconditioner data structures
    factorization: Option<Vec<f64>>,
}

impl<M, V> Preconditioner<M, V> for MyCustomPreconditioner {
    fn setup(&mut self, matrix: &M) -> Result<(), KError> {
        // Preconditioner setup/factorization phase
        // Store factorization data
        Ok(())
    }
    
    fn apply(&self, side: PcSide, x: &V, y: &mut V) -> Result<(), KError> {
        // Apply M⁻¹x → y (or x M⁻¹ → y for right preconditioning)
        match side {
            PcSide::Left => {
                // Left preconditioning: solve Mz = x, return z in y
            },
            PcSide::Right => {
                // Right preconditioning: solve zM = x, return z in y  
            },
        }
        Ok(())
    }
}
```

### Matrix-Free Operators
```rust
use kryst::{MatVec, KError};

struct LaplacianOperator {
    n: usize,  // Grid size
    h: f64,    // Grid spacing
}

impl MatVec<Vec<f64>> for LaplacianOperator {
    fn matvec(&self, x: &Vec<f64>, y: &mut Vec<f64>) -> Result<(), KError> {
        // Implement matrix-vector product y = Ax
        // For 1D Laplacian: -u''(x) ≈ -(u[i+1] - 2u[i] + u[i-1])/h²
        
        for i in 0..self.n {
            if i == 0 || i == self.n - 1 {
                y[i] = x[i];  // Boundary conditions
            } else {
                y[i] = (-x[i-1] + 2.0*x[i] - x[i+1]) / (self.h * self.h);
            }
        }
        Ok(())
    }
    
    fn size(&self) -> (usize, usize) {
        (self.n, self.n)
    }
}

// Usage with KspContext
let laplacian = LaplacianOperator { n: 1000, h: 0.001 };
let mut ksp = KspContext::new();
ksp.set_type(SolverType::Cg).unwrap()
   .set_pc_type(PcType::Jacobi).unwrap();

// Can use matrix-free operator directly
// ksp.setup(&laplacian, 1000).unwrap();
```

## Documentation and Resources

- **[API Documentation](https://docs.rs/kryst)** - Complete API reference with examples
- **[Repository](https://github.com/tmathis720/kryst)** - Source code, issues, and discussions
- **[Examples Directory](https://github.com/tmathis720/kryst/tree/main/examples)** - Comprehensive demonstration programs
- **[Benchmarks](https://github.com/tmathis720/kryst/tree/main/benches)** - Performance comparison suite
- **[Phase III/IV Summary](PHASE_III_IV_SUMMARY.md)** - Advanced preconditioning and automation features

### Mathematical References
- Saad, Y. (2003). *Iterative Methods for Sparse Linear Systems*, 2nd Edition. SIAM.
- Barrett, R. et al. (1994). *Templates for the Solution of Linear Systems: Building Blocks for Iterative Methods*. SIAM.
- Trefethen, L.N. & Bau, D. (1997). *Numerical Linear Algebra*. SIAM.
- Briggs, W.L., Henson, V.E. & McCormick, S.F. (2000). *A Multigrid Tutorial*, 2nd Edition. SIAM.

### Software References
- PETSc Documentation: [https://petsc.org/release/documentation/](https://petsc.org/release/documentation/)
- Trilinos Documentation: [https://trilinos.github.io/](https://trilinos.github.io/)

## Testing and Validation

Run the comprehensive test suite:

```bash
# All tests
cargo test

# Specific test categories
cargo test --lib solver
cargo test --lib preconditioner
cargo test --lib context
cargo test --lib utils

# Integration tests
cargo test test_phase_iii_iv_integration
cargo test test_options_integration
cargo test test_preconditioner_integration

# With specific features
cargo test --features "rayon"
cargo test --features "mpi" 
cargo test --features "logging"

# Performance testing
cargo test --release
```

### Test Coverage
- **Unit Tests**: 148+ individual component tests
- **Integration Tests**: End-to-end solver and preconditioner validation
- **Options Tests**: CLI parsing and configuration validation
- **Phase Tests**: Advanced feature validation (PC-chaining, monitoring, tuning)
- **Performance Tests**: Benchmark validation and regression testing

## Migration Guide

### From Version 0.x to 1.0

**New Features:**
- Enhanced Chebyshev preconditioner with eigenvalue estimation
- AMG with configurable pre/post smoothing parameters
- PC-chaining for composite preconditioning
- Iteration monitoring and automated parameter tuning
- Expanded CLI options (50+ parameters)

**Breaking Changes:**
- None! Version 1.0 maintains full backward compatibility

**Recommended Upgrades:**
```rust
// Old approach
ksp.set_pc_type(PcType::Chebyshev).unwrap();

// Enhanced approach (optional)
let mut pc_opts = PcOptions::default();
pc_opts.chebyshev_degree = Some(6);
ksp.set_pc_options(pc_opts);
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/tmathis720/kryst.git
   cd kryst
   ```

2. Install Rust (stable toolchain recommended):
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

3. Optional: Install MPI for distributed features:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install libopenmpi-dev
   
   # macOS  
   brew install open-mpi
   ```

4. Run tests and benchmarks:
   ```bash
   cargo test
   cargo bench
   cargo test --features "mpi"  # If MPI is available
   ```

### Areas for Contribution

#### High Priority
- **GPU Acceleration**: CUDA/OpenCL backends for matrix operations
- **Additional Solvers**: LOBPCG, IDR(s), BiCGStab(l) variants
- **Matrix Formats**: Coordinate (COO), block sparse (BSR) formats
- **Performance**: SIMD optimizations, better cache utilization

#### Medium Priority  
- **Multigrid Variants**: Classical AMG, smoothed aggregation
- **Eigenvalue Solvers**: Integration with Krylov eigenvalue methods
- **Nonlinear Solvers**: Newton-Krylov, JFNK methods
- **Adaptive Methods**: Adaptive restart, dynamic tolerance adjustment

#### Lower Priority
- **Complex Arithmetic**: Complex-valued linear systems support
- **Mixed Precision**: fp16/fp32/fp64 combinations for accuracy/performance tradeoffs
- **Advanced I/O**: HDF5, NetCDF matrix I/O support
- **Visualization**: Integration with plotting libraries for convergence analysis

### Code Style and Standards

- Follow Rust standard formatting: `cargo fmt`
- Ensure clippy compliance: `cargo clippy`
- Add comprehensive tests for new features
- Include benchmark tests for performance-critical code
- Document public APIs with examples
- Follow semantic versioning for releases

### Pull Request Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and add tests
4. Ensure all tests pass: `cargo test`
5. Run formatting and linting: `cargo fmt && cargo clippy`
6. Commit your changes: `git commit -m 'Add amazing feature'`
7. Push to the branch: `git push origin feature/amazing-feature`
8. Open a Pull Request with a clear description

---

**kryst** provides a comprehensive, high-performance linear algebra toolkit for the Rust ecosystem, with particular focus on iterative methods for large-scale scientific computing applications. The library combines the mathematical rigor of established numerical libraries like PETSc with the safety and performance characteristics of Rust, making it ideal for research, scientific computing, and production applications requiring robust linear system solvers.
