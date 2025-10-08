# Complex scalars

- Enable the `complex` Cargo feature when building (`cargo build --features complex`) to switch Kryst's internal scalar alias `S` to `num_complex::Complex64`.
- Read Matrix Market matrices with `read_matrix_market` and convert them into scalar-aware structures via `MatrixMarketData::to_csr_matrix_scalar()`, `MatrixMarketData::to_dense_matrix_scalar()`, or `MatrixMarketData::to_vector_scalar()` when the file is complex-valued.
- Write scalar vectors and matrices back to Matrix Market using `write_vector_market_scalar`, `write_matrix_market_coordinate_scalar`, and `write_matrix_market_array_scalar`; these helpers automatically emit complex headers when any imaginary parts are present.
- Real-only consumers can continue calling the legacy real helpers. When the `complex` feature is active they reuse the scalar writers and drop the imaginary data during conversion so the public `f64` API stays stable.
- Always pair complex right-hand sides with matching matrix files; the dense and coordinate writers validate symmetry (Hermitian/skew) before emitting, preventing silent structural mistakes.
