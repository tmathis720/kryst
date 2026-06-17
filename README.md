# kryst

[![Crates.io](https://img.shields.io/crates/v/kryst.svg)](https://crates.io/crates/kryst)
[![Documentation](https://docs.rs/kryst/badge.svg)](https://docs.rs/kryst)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`kryst` is a Rust library for solving linear systems with Krylov methods,
preconditioners, matrix-free operators, sparse and dense matrix backends, and
optional shared-memory or MPI execution.

The main entry point is `KspContext`: choose a solver, choose a preconditioner,
attach operators, call `setup()`, then call `solve()`. Lower-level solver,
matrix, communicator, and preconditioner traits are also exposed for custom
integrations.

## Features

- Krylov solvers: CG, PCG, GMRES, FGMRES, BiCGStab, CGS, MINRES, QMR, TFQMR,
  TCQMR, CGNR, LSQR, LSMR, Richardson, Chebyshev, CR, GCR, pipelined GCR, and
  PCA-GMRES.
- Direct-solver interfaces for feature-gated backends through
  `SolverType::Preonly` and direct preconditioner implementations.
- Preconditioners: none, Jacobi, block Jacobi, SOR/SSOR, ILU variants, ILUT,
  ILUTP, ILUP, Chebyshev, AMG, ASM/RAS, approximate inverse, FieldSplit, shell
  callbacks, nested KSP-as-PC, MG, BDDC, GAMG, LU, QR, and feature-gated
  SuperLU_DIST.
- Matrix types and adapters for dense matrices, CSR, CSC, shell/matrix-free
  operators, distributed CSR, backend conversions, and Matrix Market loading.
- Runtime option parsing for solver and preconditioner configuration, including
  `-ksp_type`, `-pc_type`, tolerances, restart lengths, side selection, AMG/ILU
  tuning, FieldSplit, shell PC, nested KSP, and distributed routing options.
- Setup reuse through operator structure/value identifiers and preconditioner
  numeric-update hooks.
- Optional complex scalars, Rayon kernels, MPI communicators, diagnostics,
  monitoring callbacks, and reproducible reduction modes.

## Installation

```toml
[dependencies]
kryst = "4.1.5"
```

Default features enable Rayon and the `faer` backend:

```toml
kryst = { version = "4.1.5", default-features = true }
```

For a smaller build, opt into only what you need:

```toml
kryst = { version = "4.1.5", default-features = false, features = ["backend-faer"] }
```

Common feature flags:

| Feature | Purpose |
| --- | --- |
| `backend-faer` | Dense/CSR backend used by most examples and many preconditioners. Enabled by default. |
| `rayon` | Shared-memory parallel kernels and thread controls. Enabled by default. |
| `mpi` | MPI communicator support and distributed operators/preconditioners. |
| `mpi_examples` | Convenience feature for MPI examples. |
| `complex` | Uses `num_complex::Complex64` as the library scalar type `S`. |
| `dense-direct` | Opt-in dense direct solver and preconditioner modules. Requires `backend-faer`. |
| `simd` | Enables SIMD-oriented sparse matvec planning through `wide`. |
| `transpose-cache` | Enables cached transpose support with `parking_lot`. |
| `logging` | Enables log-oriented instrumentation paths. |
| `superlu_dist` | Enables SuperLU_DIST-facing code paths where available. |

## Quick Start

This example builds a matrix-free one-dimensional Laplacian and solves it with
GMRES:

```rust
use kryst::matrix::MatShell;
use kryst::prelude::*;
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n = 8;
    let op = MatShell::<S>::new(n, n, move |x, y| {
        for i in 0..n {
            let mut sum = S::from_real(2.0) * x[i];
            if i > 0 {
                sum -= x[i - 1];
            }
            if i + 1 < n {
                sum -= x[i + 1];
            }
            y[i] = sum;
        }
    });

    let a = Arc::new(op) as Arc<dyn LinOp<S = S>>;
    let b = vec![S::from_real(1.0); n];
    let mut x = vec![S::zero(); n];

    let mut ksp = KspContext::new();
    ksp.set_type(SolverType::Gmres)?;
    ksp.set_pc_type(PcType::None, None)?;
    ksp.set_operators(a, None);
    ksp.setup()?;

    let stats = ksp.solve(&b, &mut x)?;
    println!(
        "iters={} reason={:?} final_residual={:.3e}",
        stats.iterations, stats.reason, stats.final_residual
    );

    Ok(())
}
```

Run the bundled version with:

```bash
cargo run --example basic_gmres
```

## KSP Lifecycle

Typical usage follows the same sequence for dense, sparse, shell, and
distributed operators:

1. Configure the solver with `set_type`, `set_pc_type`, or parsed options.
2. Attach the system operator and optional preconditioning operator with
   `set_operators`.
3. Call `setup()` to build preconditioners and allocate reusable workspace.
4. Call `solve(&b, &mut x)` for one or more right-hand sides.

`solve()` can trigger setup when needed, but explicit setup is useful when you
want to separate setup cost from solve cost or reuse the same context:

```rust
let mut ksp = KspContext::new();
ksp.set_type(SolverType::Cg)?;
ksp.set_pc_type(PcType::Jacobi, None)?;
ksp.set_operators(a.clone(), None);
ksp.setup()?;

for b in rhs_list {
    let mut x = vec![S::zero(); b.len()];
    let stats = ksp.solve(&b, &mut x)?;
    println!("{:?}", stats.reason);
}
```

Operators can expose `structure_id` and `values_id` metadata. When those IDs
show that only numeric values changed, preconditioners that support numeric
updates can avoid a full symbolic rebuild.

## Solver and Preconditioner Selection

The typed API uses `SolverType` and `PcType`:

```rust
let mut ksp = KspContext::new();
ksp.set_type(SolverType::Fgmres)?;
ksp.set_pc_type(PcType::Amg, None)?;
```

The string parser accepts the same names used by the runtime options layer:

- Solvers: `cg`, `pcg`, `gmres`, `fgmres`, `bicgstab`, `cgs`, `minres`,
  `qmr`, `tfqmr`, `tcqmr`, `cgnr`, `lsqr`, `lsmr`, `richardson`,
  `chebyshev`, `cr`, `gcr`, `pipegcr`, `pca_gmres`, and `preonly`.
- Preconditioners: `none`, `jacobi`, `block_jacobi`, `sor`, `ilu0`, `ilu`,
  `ilut`, `ilutp`, `ilup`, `chebyshev`, `amg`, `asm`, `ras`, `approxinv`,
  `fieldsplit`, `shell`, `ksp`, `mg`, `bddc`, `gamg`, `lu`, `qr`, and
  `superludist` when built with `superlu_dist`.

Some combinations have mathematical requirements. For example, CG, PCG, MINRES,
LSQR, LSMR, Chebyshev, and CR require left preconditioning. Flexible methods
such as FGMRES and GCR use right preconditioning.

## Runtime Options

`kryst::config::options::parse_all_options` parses command-line style options
into `KspOptions` and `PcOptions`:

```rust
use kryst::config::options::parse_all_options;

let args: Vec<String> = std::env::args().collect();
let (ksp_opts, pc_opts) = parse_all_options(&args)?;

let mut ksp = KspContext::new();
ksp.set_from_all_options(&ksp_opts, &pc_opts)?;
ksp.set_operators(a, None);
let stats = ksp.solve(&b, &mut x)?;
```

Common options:

```bash
my_solver -ksp_type gmres -ksp_rtol 1e-8 -ksp_max_it 1000 -pc_type jacobi
my_solver -ksp_type fgmres -ksp_gmres_restart 80 -pc_type ilutp -pc_ilutp_drop_tol 1e-4
my_solver -ksp_type preonly -pc_type lu   # requires a direct-solver backend feature
my_solver -ksp_view -pc_view
my_solver -help
```

The options system also supports options files with `-options_file <path>`.
Precedence is command line, then options files, then environment, then defaults.

## Matrices and Operators

The central operator trait is `LinOp`. Implement it directly for custom matrix
types, or use one of the provided wrappers:

- `MatShell` for matrix-free matvec closures.
- Dense operators backed by `faer` when `backend-faer` is enabled.
- CSR and CSC matrix utilities, including sparse kernels and Matrix Market IO.
- `DistCsrOp` and related distributed CSR helpers for MPI builds.

The examples directory includes small matrix-free demos, Matrix Market demos,
distributed CSR examples, AMG demos, shell preconditioner examples, and complex
solver examples.

## Parallel Execution

Rayon support is enabled by default. You can size the worker pool through the
options layer:

```bash
my_solver -ksp_threads 4
```

MPI support is feature-gated:

```bash
cargo mpirun -n 4 --example mpi_parallel_demo --features mpi
cargo mpirun -n 4 --example fidapm05_amg_features --features mpi_examples
```

Distributed support is built around the `Comm` abstraction and distributed
matrix/preconditioner paths. Some preconditioners operate on rank-local blocks;
others have distributed routes or MPI-aware wrappers. The route can often be
controlled through options such as `-pc_global`, `-pc_local`,
`-pc_dist_route`, and AMG distributed coarse-solver options.

## Complex Builds

In real builds, `S = f64` and `R = f64`. With `--features complex`, `S` becomes
`num_complex::Complex64` and `R` remains `f64`.

```bash
cargo run --example complex_gmres --features complex
cargo run --example shell_pc_matrix_free_complex --features complex
```

Not every example is available in complex mode; real-only examples print a
message when built with `complex`.

## Diagnostics and Monitoring

`SolveStats` reports iteration count, convergence reason, final residual, and
solver counters such as global reductions where tracked. Structured diagnostics
can be emitted through:

```bash
my_solver -ksp_view -pc_view
```

`KspContext` supports monitor callbacks, and `kryst::utils` includes monitoring,
verification, convergence, matrix-screening, metrics, and tuning utilities.

## Examples

Useful starting points:

```bash
cargo run --example basic_gmres
cargo run --example options_demo -- -ksp_type gmres -pc_type jacobi
cargo run --example matrix_market_demo
cargo run --example amg_options_demo
cargo run --example shell_pc_matrix_free
cargo run --example complex_gmres --features complex
cargo mpirun -n 4 --example mpi_parallel_demo --features mpi
```

Run `cargo run --example test_help -- -help` to inspect registered runtime
options from the crate.

## Development

Common checks:

```bash
cargo test
cargo test --features complex
cargo test --features mpi
cargo bench
```

MPI tests and examples require an MPI implementation and are normally launched
with `cargo mpirun` or `mpirun`, depending on your local tooling.
