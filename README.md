<!--
    kryst: PETSc-style Krylov solvers and preconditioners for Rust.
    This README describes the main features, usage, and documentation pointers.
-->

# kryst

Krylov subspace and preconditioned iterative solvers for dense and sparse linear systems, with shared and distributed memory parallelism.

## Features
- GMRES, CG, BiCGStab, MINRES, and other Krylov solvers
- Preconditioners: Jacobi, ILU, Chebyshev, AMG, Additive Schwarz, and more
- Dense and sparse matrix support
- Shared-memory parallelism via Rayon
- Distributed-memory parallelism via MPI (optional)
- Block and pipelined communication-avoiding variants
- **PETSc-style command-line options database**
- **Unified KSP context for runtime solver selection**
- Extensible trait-based design for custom matrices and preconditioners

## Usage

Add to your `Cargo.toml`:
```toml
[dependencies]
kryst = "0.5"
```

Enable parallel or MPI features as needed:
```toml
[features]
default = ["rayon"]
mpi = ["dep:mpi"]
```

## Examples

### Basic Solver Usage

```rust
use kryst::solver::GmresSolver;
// ... set up your matrix A and vectors b, x ...
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
./my_program -ksp_type cg -ksp_max_it 500 -pc_type ilu0
./my_program -help  # Show all available options
```

## Supported Command-line Options

### KSP (Krylov Solver) Options
- `-ksp_type <solver>` - Solver type: `cg`, `pcg`, `gmres`, `bicgstab`, `cgs`, `qmr`, `tfqmr`, `minres`, `cgnr`, `preonly`
- `-ksp_rtol <float>` - Relative convergence tolerance (default: 1e-6)
- `-ksp_atol <float>` - Absolute convergence tolerance (default: 1e-12)
- `-ksp_dtol <float>` - Divergence tolerance (default: 1e3)
- `-ksp_max_it <int>` - Maximum number of iterations (default: 1000)
- `-ksp_gmres_restart <int>` - GMRES restart parameter (default: 50)
- `-ksp_pc_side <side>` - Preconditioning side: `left`, `right`, `symmetric`

### PC (Preconditioner) Options
- `-pc_type <pc>` - Preconditioner type: `jacobi`, `ilu0`, `none`
- `-pc_ilu_levels <int>` - ILU fill levels (default: 0)
- `-pc_chebyshev_degree <int>` - Chebyshev polynomial degree (default: 3)
- `-pc_ilut_drop_tol <float>` - ILUT drop tolerance (default: 1e-3)
- `-pc_ilut_max_fill <int>` - ILUT maximum fill per row (default: 10)

### Examples
```bash
# GMRES with Jacobi preconditioning
-ksp_type gmres -ksp_rtol 1e-8 -pc_type jacobi

# CG solver with strict tolerance
-ksp_type cg -ksp_max_it 500 -ksp_rtol 1e-12 -pc_type ilu0

# BiCGStab with no preconditioning
-ksp_type bicgstab -pc_type none
```

## Documentation

- [API Docs (docs.rs)](https://docs.rs/kryst)
- [Repository](https://github.com/yourusername/kryst)

## License

## License

MIT

## Contributing

Contributions, bug reports, and feature requests are welcome! Open an issue or pull request on GitHub.
