# Complex scalars

- Enable the `complex` Cargo feature when building (`cargo build --features complex`) to switch Kryst's internal scalar alias `S` to `num_complex::Complex64`.
- Read Matrix Market matrices with `read_matrix_market` and convert them into scalar-aware structures via `MatrixMarketData::to_csr_matrix_scalar()`, `MatrixMarketData::to_dense_matrix_scalar()`, or `MatrixMarketData::to_vector_scalar()` when the file is complex-valued.
- Write scalar vectors and matrices back to Matrix Market using `write_vector_market_scalar`, `write_matrix_market_coordinate_scalar`, and `write_matrix_market_array_scalar`; these helpers automatically emit complex headers when any imaginary parts are present.
- Real-only consumers can continue calling the legacy real helpers. When the `complex` feature is active they reuse the scalar writers and drop the imaginary data during conversion so the public `f64` API stays stable.
- Always pair complex right-hand sides with matching matrix files; the dense and coordinate writers validate symmetry (Hermitian/skew) before emitting, preventing silent structural mistakes.

## Method-level complex support notes

The `complex` feature is still evolving. Current high-value methods now behave as follows:

- `GMRES` `SStep` variant no longer hard-fails under complex builds; it executes through the same `solve_sstep` path used by real scalars (with existing fallback to classical GMRES when `s > 1`).
- `SOR` (`SorPc`) accepts complex vectors and uses a real-valued sweep operator assembled from the operator real part when setup is provided as CSR. This is functionally equivalent for real-valued matrices with complex RHS/iterates.
- `DeflationPC` now applies in complex mode by combining real coarse-space operators (`Z`, `AZ`, `E`) with complex vectors; coarse factors remain real.
- `ApproxInv` complex setup/apply now runs through native complex kernels (`Complex64`) for CSR operators, preserving coupling between real/imaginary components.
- `IluCsr` complex setup/apply uses native complex factorization and triangular-solve kernels over `Complex64`, while still retaining explicit fallback controls for regression coverage.

### Remaining exclusions / tradeoffs

- `SOR`, `ApproxInv`, and `IluCsr` complex setup paths currently require CSR operators in complex mode.
- `IluCsr` and `ApproxInv` expose setup-kernel diagnostics so you can confirm whether native complex kernels were active (and if not, inspect fallback reasons).
- `MG` (`MgPc`) level/operator storage is scalar-aware (`S`) across hierarchy operators and transfer operators; method parity still depends on the configured smoothers/coarse solvers.
