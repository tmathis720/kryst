# kryst

Krylov subspace and preconditioned iterative solvers for dense and sparse linear systems, with shared and distributed memory parallelism.

## Features
- GMRES, CG, BiCGStab, MINRES, and other Krylov solvers
- Preconditioners: Jacobi, ILU, Chebyshev, AMG, Additive Schwarz, and more
- Dense and sparse matrix support
- Shared-memory parallelism via Rayon
- Distributed-memory parallelism via MPI (optional)
- Block and pipelined communication-avoiding variants
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

## Example
```rust
use kryst::solver::GmresSolver;
// ... set up your matrix A and vectors b, x ...
let mut solver = GmresSolver::new(30, 1e-8, 200);
let stats = solver.solve(&A, None, &b, &mut x).unwrap();
println!("Converged: {} in {} iterations", stats.converged, stats.iterations);
```

## Documentation
- [API Docs (docs.rs)](https://docs.rs/kryst)
- [Repository](https://github.com/yourusername/kryst)

## License
MIT

## Contributing
Contributions, bug reports, and feature requests are welcome! Open an issue or pull request on GitHub.
