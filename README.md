<!--
    kryst: PETSc-style Krylov solvers and preconditioners for Rust.
    This README describes the main features, usage, and documentation pointers.
-->

# kryst

[![Crates.io](https://img.shields.io/crates/v/kryst.svg)](https://crates.io/crates/kryst)
[![Documentation](https://docs.rs/kryst/badge.svg)](https://docs.rs/kryst)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

High-performance Krylov subspace and preconditioned iterative solvers for dense and sparse linear systems, with shared and distributed memory parallelism.

## Features

### Solvers
- **Krylov Methods**: GMRES, BiCGStab, CG, PCG, MINRES, CGS, QMR, TFQMR, CGNR
- **Advanced Variants**: Flexible GMRES (FGMRES), Pipelined Communication-Avoiding GMRES (PCA-GMRES)
- **Direct Methods**: LU and QR factorization via PREONLY solver type
- **Parallel Implementations**: Shared-memory (Rayon) and distributed-memory (MPI) support

### Preconditioners
- **Basic**: Jacobi (diagonal scaling), Block Jacobi, SOR/SSOR
- **Incomplete Factorizations**: ILU(0), ILU(k), ILUT, ILUP 
- **Polynomial**: Chebyshev smoothing
- **Multilevel**: Algebraic Multigrid (AMG)
- **Domain Decomposition**: Additive Schwarz Method (ASM)
- **Approximate Inverse**: SPAI-type approximate inverse

### Architecture
- **PETSc-style API**: Unified KSP context for runtime solver selection
- **Command-line Options**: Complete options database like PETSc
- **Trait-based Design**: Extensible for custom matrices and preconditioners
- **Memory Efficiency**: In-place operations and configurable storage
- **High Performance**: Optimized inner kernels with SIMD and parallelization

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
kryst = "0.6"
```

### Feature Flags

```toml
[features]
default = ["rayon"]          # Shared-memory parallelism
rayon = ["dep:rayon"]        # Rayon-based parallel execution  
mpi = ["dep:mpi"]           # Distributed-memory parallelism via MPI
```

## Quick Start

### Basic Solver Usage

```rust
use kryst::solver::GmresSolver;
use kryst::core::traits::{MatVec, LinearSolver};

// Set up your matrix A and vectors b, x
let mut solver = GmresSolver::new(30, 1e-8, 200);
let stats = solver.solve(&A, None, &b, &mut x).unwrap();
println!("Converged: {} in {} iterations", stats.converged, stats.iterations);
```

### PETSc-style Unified Interface

```rust
use kryst::context::ksp_context::KspContext;

// Configure solver and preconditioner at runtime
let mut ksp = KspContext::new();
ksp.set_type_from_str("gmres")?
   .set_pc_type_from_str("jacobi")?
   .set_tolerances(1e-8, 1e-12, 1e3, 1000);

let stats = ksp.solve(&A, &b, &mut x)?;
```

### Command-line Options (PETSc-style)

```rust
use kryst::config::options::parse_all_options;
use kryst::context::ksp_context::KspContext;

// Parse command-line options
let args: Vec<String> = std::env::args().collect();
let (ksp_opts, pc_opts) = parse_all_options(&args)?;

// Configure from options  
let mut ksp = KspContext::new();
ksp.set_from_all_options(&ksp_opts, &pc_opts)?;
let stats = ksp.solve(&A, &b, &mut x)?;
```

Run your program with PETSc-style options:
```bash
./my_program -ksp_type gmres -ksp_rtol 1e-8 -pc_type jacobi
./my_program -ksp_type preonly -pc_type lu     # Direct LU solver
./my_program -ksp_type cg -ksp_max_it 500 -pc_type ilu0  
./my_program -help  # Show all available options
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
- `-pc_type <pc>` - Preconditioner type: `jacobi`, `ilu0`, `ilu`, `ilut`, `ilup`, `chebyshev`, `amg`, `asm`, `lu`, `qr`, `none`
- `-pc_ilu_levels <int>` - ILU fill levels (default: 0)
- `-pc_chebyshev_degree <int>` - Chebyshev polynomial degree (default: 3)
- `-pc_ilut_drop_tol <float>` - ILUT drop tolerance (default: 1e-3)
- `-pc_ilut_max_fill <int>` - ILUT maximum fill per row (default: 10)

### Usage Examples
```bash
# GMRES with Jacobi preconditioning
-ksp_type gmres -ksp_rtol 1e-8 -pc_type jacobi

# Direct LU solver (single iteration)
-ksp_type preonly -pc_type lu

# Direct QR solver (single iteration)  
-ksp_type preonly -pc_type qr

# CG solver with ILU(0) preconditioning
-ksp_type cg -ksp_max_it 500 -ksp_rtol 1e-12 -pc_type ilu0

# BiCGStab with no preconditioning
-ksp_type bicgstab -pc_type none

# GMRES with AMG preconditioning
-ksp_type gmres -pc_type amg

# Flexible GMRES with ILUT preconditioning
-ksp_type fgmres -pc_type ilut -pc_ilut_drop_tol 1e-4
```

## Solver Details

### Krylov Methods
- **GMRES**: Generalized Minimal Residual with restart
- **FGMRES**: Flexible GMRES for variable preconditioning
- **PCA-GMRES**: Pipelined Communication-Avoiding GMRES
- **BiCGStab**: Bi-Conjugate Gradient Stabilized
- **CG/PCG**: Conjugate Gradient (preconditioned)
- **MINRES**: Minimal Residual for symmetric indefinite systems
- **CGS**: Conjugate Gradient Squared
- **QMR/TFQMR**: Quasi-Minimal Residual methods
- **CGNR**: CG on Normal Equations

### Direct Methods
- **PREONLY**: Single-step direct solve using LU or QR factorization
- Supports both `-pc_type lu` and `-pc_type qr`
- Ideal for well-conditioned systems where direct methods are preferred

### Preconditioners
- **Jacobi**: Diagonal scaling `M⁻¹ = diag(A)⁻¹`
- **Block Jacobi**: Block-wise diagonal inverse
- **ILU(k)**: Incomplete LU with k levels of fill
- **ILUT**: ILU with threshold dropping
- **ILUP**: ILU with partial pivoting
- **SOR/SSOR**: Successive Over-Relaxation methods
- **Chebyshev**: Polynomial smoothing
- **AMG**: Algebraic Multigrid (V-cycle)
- **ASM**: Additive Schwarz Method (overlapping domain decomposition)

## Performance Features

### Parallelization
- **Shared Memory**: Rayon-based parallel execution for matrix operations, vector operations, and preconditioner application
- **Distributed Memory**: MPI support for distributed linear algebra operations
- **Communication Avoiding**: PCA-GMRES reduces communication costs in parallel environments
- **SIMD Optimization**: Leverages hardware acceleration through optimized inner kernels

### Memory Management
- **In-place Operations**: Minimizes memory allocations during iteration
- **Configurable Storage**: Preallocated vs. dynamic storage options
- **Block Operations**: Efficient cache usage through blocked algorithms
- **Sparse Patterns**: Memory-efficient storage for sparse matrices and preconditioners

## Matrix Support

### Dense Matrices
- Full support via `faer::Mat<T>` integration
- Optimized BLAS-like operations
- Multiple precision types (f32, f64, complex)

### Sparse Matrices  
- CSR/CSC format support
- Efficient sparse matrix-vector products
- Pattern-based optimization for preconditioners

### Custom Matrices
- Trait-based `MatVec` interface
- Support for matrix-free methods
- Easy integration of custom matrix types

## Examples

The library includes several demonstration programs:

```bash
# Basic options demonstration
cargo run --example options_demo -- -ksp_type gmres -pc_type jacobi

# Direct solver examples  
cargo run --example options_demo -- -ksp_type preonly -pc_type lu
cargo run --example dense_direct

# Convergence behavior analysis
cargo run --example convergence_demo

# Setup and reuse patterns
cargo run --example setup_reuse_demo
```

## Benchmarks

Performance benchmarks are available via:

```bash
cargo bench
```

Includes comparisons between:
- Different solver types (GMRES vs BiCGStab vs CG)
- Preconditioner effectiveness  
- Direct vs iterative methods
- Serial vs parallel performance

## Extensions and Custom Components

### Custom Solvers
```rust
use kryst::solver::LinearSolver;
use kryst::core::traits::MatVec;

struct MyCustomSolver {
    // solver parameters
}

impl<M, V> LinearSolver<M, V> for MyCustomSolver 
where M: MatVec<V> 
{
    fn solve(&mut self, a: &M, pc: Option<&dyn Preconditioner<M, V>>, 
             b: &V, x: &mut V) -> Result<SolveStats, KError> {
        // custom solver implementation
    }
}
```

### Custom Preconditioners
```rust
use kryst::preconditioner::{Preconditioner, PcSide};

struct MyPreconditioner {
    // preconditioner data
}

impl<M, V> Preconditioner<M, V> for MyPreconditioner {
    fn setup(&mut self, a: &M) -> Result<(), KError> {
        // setup/factorization phase
    }
    
    fn apply(&self, side: PcSide, x: &V, y: &mut V) -> Result<(), KError> {
        // apply M⁻¹x → y
    }
}
```

## Documentation and Resources

- **[API Documentation](https://docs.rs/kryst)** - Complete API reference
- **[Repository](https://github.com/tmathis720/kryst)** - Source code and issues
- **[Examples](https://github.com/tmathis720/kryst/tree/main/examples)** - Demonstration programs
- **[Benchmarks](https://github.com/tmathis720/kryst/tree/main/benches)** - Performance comparisons

### References
- Saad, Y. (2003). *Iterative Methods for Sparse Linear Systems*, 2nd Edition. SIAM.
- Barrett, R. et al. (1994). *Templates for the Solution of Linear Systems*. SIAM.
- PETSc Documentation: [https://petsc.org/release/documentation/](https://petsc.org/release/documentation/)

## Testing

Run the comprehensive test suite:

```bash
# All tests
cargo test

# Specific test categories
cargo test solver_
cargo test preconditioner_
cargo test options_
cargo test preonly_

# With parallel features
cargo test --features "rayon"
cargo test --features "mpi"
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

1. Clone the repository
2. Install Rust (stable toolchain recommended)
3. Optional: Install MPI library for distributed features
4. Run tests: `cargo test`
5. Run benchmarks: `cargo bench`

### Areas for Contribution
- Additional solver algorithms (e.g., LOBPCG, IDR)
- New preconditioner types (e.g., multigrid variants)
- GPU acceleration (CUDA/OpenCL backends)
- Additional matrix formats (coordinate, block sparse)
- Performance optimizations
- Documentation improvements

---

**kryst** aims to provide a comprehensive, high-performance linear algebra toolkit for the Rust ecosystem, with particular focus on iterative methods for large-scale scientific computing applications.
