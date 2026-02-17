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
- `ApproxInv` in complex mode currently runs in a guarded degraded mode: CSR setup is required and initialization uses diagonal real-part projection. Complex apply remains available, but setup is not a full complex SPAI/FSAI build.
- `IluCsr` in complex mode currently runs in a guarded degraded mode: setup projects CSR values to their real part and apply executes split real/imag triangular solves. This preserves CSR fast paths but is not a native complex ILU factorization.

### Remaining exclusions / tradeoffs

- `SOR`, `ApproxInv`, and `IluCsr` complex setup paths currently require CSR operators in complex mode.
- `IluCsr` and `ApproxInv` emit explicit degraded-mode diagnostics in complex setup paths, and both currently rely on real-part projection kernels where mathematically unavoidable.
- `MG` (`MgPc`) level/operator storage is scalar-aware (`S`) across hierarchy operators and transfer operators; method parity still depends on the configured smoothers/coarse solvers.
