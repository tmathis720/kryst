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
