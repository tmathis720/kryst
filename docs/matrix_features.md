# Matrix module feature matrix

This document summarizes which `matrix` module functionality is enabled under
each combination of crate features, and which invariants apply regardless of
the backend.

## Features

- `backend-faer`: Enables Faer-based dense and sparse interop (`faer::Mat`,
  `SparseColMat`, etc.) and the SIMD/parallel wrappers.
- `simd`: Builds SIMD-accelerated CSR SpMV kernels for real scalars when
  `backend-faer` is active and `complex` is off.
- `rayon`: Enables parallel SpMV / matvec paths and Rayon-backed communicators.
- `complex`: Switches the active scalar domain to complex numbers, disabling
  SIMD kernels and altering transpose semantics.
- `transpose-cache`: Caches CSR→CSC conversions inside `CsrOp` keyed by
  [`ValuesId`].
- `mat-values-fingerprint`: Strengthens `faer::Mat<f64>::values_id()` by
  hashing entries so caches can detect value changes without wrappers.

## Support matrix

| Features                              | Dense backend           | CSR SpMV            | CSC / transpose        | DistCsrOp / ParCsr | Notes |
|--------------------------------------|--------------------------|---------------------|------------------------|--------------------|-------|
| *(none)*                             | traits-only stub         | ✅ scalar           | ✅ via CSR gather      | ✅ (local-only)    | No Faer backend; format helpers fall back to pointer identity. |
| `backend-nalgebra`                   | `nalgebra::DMatrix`      | ❌                  | ❌                     | ❌                | Dense-only materialization; sparse formats unsupported. |
| `backend-naive`                      | stub                     | ❌                  | ❌                     | ❌                | Feature-gated backend stub with no materialization support. |
| `backend-faer`                       | `faer::Mat` + `DenseOp`  | ✅ scalar           | ✅ CSC + gather        | ✅                 | SIMD/rayon disabled; caches rely on change IDs. |
| `backend-faer,rayon`                 | same as above            | ✅ parallel          | ✅ parallel            | ✅                 | Rayon thresholds govern local parallelism. |
| `backend-faer,simd`                  | same as above            | ✅ SIMD + scalar     | ✅ scalar transpose    | ✅                 | SIMD path chosen for large CSR, real scalars only. |
| `backend-faer,simd,rayon`            | same as above            | ✅ SIMD + Rayon      | ✅ scalar transpose    | ✅                 | Highest-performance real configuration. |
| `backend-faer,complex`               | `faer::Mat<Complex64>`   | ✅ scalar only       | ✅ scalar transpose    | ✅                 | SIMD disabled; complex transpose uses scalar gather. |
| `backend-faer,transpose-cache`       | same as above             | ✅ scalar           | ✅ cached CSC          | ✅                 | CSC cache keyed on `ValuesId` for repeated transposes. |
| `backend-faer,mpi`                   | same as above            | ✅ scalar           | ✅ scalar transpose    | ✅ distributed      | MPI communicator required for halo exchange and reductions. |
| `backend-faer,mpi,rayon`             | same as above            | ✅ parallel          | ✅ parallel            | ✅ distributed      | MPI collectives + Rayon local kernels; pin thread count for reproducibility. |
| `backend-faer,mpi,rayon,complex`     | same as above            | ✅ parallel          | ✅ scalar transpose    | ✅ distributed      | Complex transpose remains scalar; MPI paths support complex scalars. |

## MPI/Rayon capability checklist

Use this checklist when validating MPI/Rayon configurations against the
documented behavior in the README and matrix module:

- **Features and runtime flags**
  - `backend-faer` required for CSR/Dense kernels; MPI and Rayon are optional.
  - `mpi` enables distributed communicators and `DistCsrOp` halo exchange.
  - `rayon` enables shared-memory parallel kernels and Rayon communicators.
  - `-ksp_threads <N>` and/or `RAYON_NUM_THREADS=<N>` controls Rayon worker count.
  - `-ksp_reproducible` forces rank-ordered MPI reductions and fixed-order local kernels.

- **MPI reductions and communicator paths**
  - `src/parallel/mod.rs` selects MPI vs. Rayon communicator implementations.
  - `src/parallel/mpi_comm.rs` provides rank-ordered and non-deterministic reductions.
  - `src/parallel/rayon_comm.rs` mirrors the reduction API for local-only runs.

- **Distributed SpMV and halo exchange**
  - `DistCsrOp` uses halo exchange in `src/matrix/dist/halo.rs`.
  - Distributed CSR application in `src/matrix/dist_csr.rs` requires MPI for neighbor exchange.

- **Local SpMV/vector kernels**
  - Rayon-backed CSR paths live in `src/matrix/spmv/mod.rs`.
  - Vector kernels used by solvers live in `src/algebra/parallel.rs`.

## Preconditioner local vs distributed behavior

The matrix below summarizes which preconditioners operate rank-locally (no MPI
communication) versus those that provide distributed/global behavior when the
operator is a `DistCsrOp` and `mpi` is enabled.

| Preconditioner | Rank-local only | Distributed-capable | Notes |
|---|---|---|---|
| Jacobi / Block Jacobi | ✅ | ✅ | `pc_type block_jacobi` uses local blocks; `pc_global=block_jacobi` wraps MPI with `DistPcAdapter`. |
| ILU / ILUT / ILUTP / ILUP | ✅ | ⚠️ | Operate on rank-local blocks; wrap with `pc_global=block_jacobi` for distributed apply. |
| Chebyshev / SOR / SSOR | ✅ | ⚠️ | Local-only; select `pc_global` if a global wrapper is desired. |
| Approximate Inverse | ✅ | ⚠️ | Rank-local only under MPI. |
| ASM / RAS | ⚠️ | ✅ | Distributed ASM supports RAS mode with overlap; ASM mode is local only. |
| AMG / GAMG | ⚠️ | ✅ | Distributed coarse execution is policy-controlled (`pc_amg_dist_apply_mode`); diagnostics include grid/operator complexity and per-level nnz/smoothing-work estimates. |
| LU / QR (dense-direct) | ✅ | ❌ | Local-only under MPI; use `SuperLU_DIST` for distributed direct solves. |
| SuperLU_DIST | ❌ | ✅ | Distributed direct solve when `superlu_dist` feature is enabled. |
| PC Chain | ⚠️ | ⚠️ | Each stage retains its own local vs distributed behavior. |

## MPI/Rayon test matrix plan

Use this plan to validate MPI/Rayon coverage with minimal, targeted tests:

1. **Communicator reductions (MPI vs. Rayon)**
   - Exercise `Comm::allreduce_sum` and `Comm::allreduce_max` on small vectors.
   - Compare results between `mpi` builds (`src/parallel/mpi_comm.rs`) and
     `rayon` builds (`src/parallel/rayon_comm.rs`).
   - Include a `-ksp_reproducible` run to confirm rank-ordered reductions.

2. **Distributed SpMV (DistCsrOp + halo exchange)**
   - Build a 2-rank halo case and verify `DistCsrOp::apply` results.
   - Validate halo buffers populated in `src/matrix/dist/halo.rs` and the
     distributed CSR apply path in `src/matrix/dist_csr.rs`.

3. **Rayon SpMV + vector kernels**
   - Run a local CSR SpMV on a moderate-size matrix with `rayon` enabled and
     verify correctness against the scalar baseline (`src/matrix/spmv/mod.rs`).
   - Validate parallel dot/axpy paths in `src/algebra/parallel.rs` with
     multiple thread counts.

## Expected feature combinations

- `backend-faer,rayon`: local parallel kernels only.
- `backend-faer,mpi`: distributed kernels with scalar local work.
- `backend-faer,mpi,rayon`: distributed kernels with parallel local work.
- `backend-faer,mpi,rayon,complex`: distributed complex kernels; transpose
  gather remains scalar.

## Complex scalar guidance

### Complex-safe matrix utilities

The CSR SpMV helpers are scalar-generic and safe to use in complex builds:

- `CsrMatrix::try_spmv`
- `CsrMatrix::spmv`
- `CsrMatrix::spmv_scaled`
- `CsrMatrix::spmv_transpose_scaled` (uses the conjugate transpose in complex builds)

### Opting into complex builds

- Enable the feature at build time: `cargo build --features complex`.
- When disabling default features, keep the matrix backend: `cargo build --no-default-features --features "backend-faer complex"`.
- In a `Cargo.toml`, add `features = ["complex"]` under the `kryst` dependency to opt in.

For end-to-end guidance on complex I/O, see `docs/howto/complex-scalars.md`.

### Expected errors for real-only AMG utilities

AMG-oriented helpers in `src/matrix/utils.rs` are real-only. In complex builds they
return `KError::Unsupported` with a "real-only; complex scalars are unsupported" message:

- `spgemm_with_drop_tol_generic`
- `spgemm_generic`
- `spgemm_btree_generic`
- `sparse_galerkin_product_generic`
- `rap_btree_generic`
- `rap_opt_generic`

### Complex-safe usage examples

```rust
use kryst::algebra::prelude::*;
use kryst::matrix::sparse::CsrMatrix;

let a = CsrMatrix::from_csr(
    2,
    2,
    vec![0, 2, 4],
    vec![0, 1, 0, 1],
    vec![S::from_parts(1.0, 0.0), S::from_parts(2.0, 1.0), S::from_parts(0.0, -1.0), S::from_parts(3.0, 0.0)],
);

let x = vec![S::from_parts(1.0, 0.5), S::from_parts(-2.0, 0.0)];
let mut y = vec![S::zero(); 2];

a.spmv_transpose_scaled(S::one(), &x, S::zero(), &mut y).unwrap();
```


### AMG/GAMG feature-behavior notes by build

| Feature set | AMG/GAMG hierarchy behavior | Scaling tradeoff |
| --- | --- | --- |
| `backend-faer` | Full AMG hierarchy build, symbolic+numeric phases with cached numeric refresh when structure is unchanged. | Best single-rank baseline; coarsening quality dominates setup time. |
| `backend-faer,rayon` | Same hierarchy semantics; smoother and local SpMV paths can parallelize per-rank work. | Better throughput on medium/large ranks; thread oversubscription can hurt coarse-level latency. |
| `backend-faer,mpi` | Distributed apply supports coarse policies via `pc_amg_dist_apply_mode` (`root_gather`/`local_prototype`). | `root_gather` lowers implementation risk but centralizes coarse solve; `local_prototype` avoids gather bottleneck but may reduce coarse-grid quality. |
| `backend-faer,mpi,rayon` | MPI coarse policy + threaded local smoothers/matvec; instrumentation via `pc_amg_dist_instrumentation`. | Highest ceiling for strong scaling if coarse policy is tuned; requires balancing MPI ranks vs threads. |
| `backend-faer,complex` | AMG/GAMG remain available but core hierarchy operators are still real-valued in MG paths. | Use AMG/GAMG where real-operator assumptions are valid; complex workloads may need additional projection cost. |

## CSR invariants

These invariants hold in every configuration and are enforced via
`debug_assert!` in [`CsrMatrix::from_csr`]:

- `row_ptr.len() == nrows + 1`.
- `row_ptr` is non-decreasing and satisfies `row_ptr[i] <= row_ptr[i + 1]`.
- `col_idx.len() == values.len()`.
- Within each row, `col_idx[row_ptr[i]..row_ptr[i + 1]]` is sorted ascending.
- `col_idx[k] < ncols` for all entries.
- No duplicates per row are produced by helper builders such as Poisson or
  SpGEMM helpers.

Violating these invariants is undefined behavior for the matrix module and may
trip the `debug_assert!` checks during development.

## Maintenance checklist

When introducing a new feature or backend option:

1. Add a row (or extend an existing row) in this table describing the new
   combination and whether SIMD, Rayon, or Faer interop is available.
2. Document any new invariants or limitations in:
   * `src/matrix/sparse.rs` and `src/matrix/csr.rs` (CSR doc comments).
   * `src/matrix/op.rs` and `src/matrix/format.rs` (change IDs / cache docs).
   * `src/matrix/dist/halo.rs` / `src/matrix/dist_csr.rs` if distributed behavior changes.
3. Update CI/test matrix to run `cargo test` with the new feature set.
4. If invariants or `unsafe` assumptions change, add debug assertions or
   `SAFETY` notes alongside the affected code.
