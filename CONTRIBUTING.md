# Contributing

## Smoke tests

These commands reproduce the new smoke coverage locally:

```bash
# Serial
RUST_LOG=kryst=debug cargo test --test context_smoke -- --nocapture

# MPI (2 ranks)
RUST_LOG=kryst=debug mpirun -n 2 cargo test --features mpi --test mpi_smoke -- --nocapture --test-threads=1
```
