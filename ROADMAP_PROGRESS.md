# Complex Number Support Roadmap - Progress Report

## Summary
This document tracks the implementation of generic scalar support (f64 + Complex64) throughout the kryst library to fix compilation errors in the `complex_matrix_market_demo` example.

## Completed Tasks

### 1. ✅ Core Scalar Trait Stabilization
- `KrystScalar` trait already exists in `algebra/scalar.rs` with all necessary bounds
- Supports both `f64` and `Complex<f64>` implementations
- Global type alias `S` switches between them based on `--features=complex` flag

### 2. ✅ Make LinOp Scalar-Generic  
**File:** `src/matrix/op.rs` (lines 557-635)
- Changed `impl LinOp for Mat<f64>` → `impl<Scalar: KrystScalar> LinOp for Mat<Scalar>`
- Now `Mat<S>` automatically implements `LinOp<S = S>` for any `S: KrystScalar`
- This enables `MatVec<Vec<S>>` via blanket impl in `core/traits.rs`

### 3. ✅ Make MatVec/DenseMatrix Work with Generic Scalar
**File:** `src/matrix/dense.rs` (lines 16-28)
- `DenseMatrix` trait inherits from `MatVec<Vec<S>>` and `Indexing`
- Generic `impl DenseMatrix for Mat<S>` works because:
  - `Mat<S>` implements `LinOp<S = S>` (from task 2)
  - Blanket `impl MatVec for all LinOp` automatically applies
- Result: `Mat<Complex64>` now satisfies all dense matrix traits

### 4. ✅ Generalize CsrOp Operator
**File:** `src/matrix/op.rs` (lines 415-530)
- Changed from `pub struct CsrOp { csr: Arc<CsrMatrix<f64>>, ... }`
- To: `pub struct CsrOp<Scalar = S> { csr: Arc<CsrMatrix<Scalar>>, ... }`
- Implemented `LinOp for CsrOp<Scalar>` with proper bounds
- Updated `ensure_csc_view()` for generic scalar support
- Fixed deref issues: `self.csr.as_ref()` → `&*self.csr` for method calls

### 5. ✅ BlockJacobi Preconditioner Adjustment  
**File:** `src/preconditioner/block_jacobi.rs` (lines 155-239)
- Separated `apply()` (takes generic `S`) from `apply_real()` (takes `f64`)
- For complex case: `apply_s` converts Complex→Real, calls `apply_real()`, converts back
- Avoids type confusion between complex and real apply methods

### 6. 🔄 Restrict ILU to Real-Only (Pragmatic Approach)
**File:** `src/preconditioner/ilu.rs` (struct Ilu, lines 370-415)
- Changed all generic `S` references to hard-coded `f64`
- Rationale: ILU algorithm uses comparisons, tolerances, and reordering that don't cleanly extend to complex
- For now, complex problems use simpler PCs (Jacobi, diagonal scaling)
- Updated apply method signature: `Vec<S>` → `Vec<f64>`

## Remaining Issues (Next Steps)

### Issue 1: CsrOp::spmv Type Bounds
**Location:** `src/matrix/op.rs:462`
**Problem:** Generic Scalar doesn't satisfy `CsrMatrix<Scalar>::spmv` bounds
**Solution:** May need explicit `KrystScalar` bound in the inherent impl block

### Issue 2: BlockJacobi apply_real Call
**Location:** `src/preconditioner/block_jacobi.rs:156`
**Problem:** apply_real(r, z) mismatch when apply takes different types  
**Solution:** Ensure apply() and apply_real() signatures align for dispatch

### Issue 3: ILU Matrix Operations with f64
**Location:** `src/preconditioner/ilu.rs:783-794`
**Problem:** ILU methods still reference generic `matrix: &Mat<S>` but struct uses f64
**Solution:** Update all ILU implementation methods to use `&Mat<f64>` explicitly

### Issue 4: AdditiveSchwarz Requires Real Scalars
**Error:** `AdditiveSchwarz<Mat<faer::mat::Own<f64>>, Vec<f64>, f64>::new()` unsatisfied
**Reason:** Submatrix extraction and decomposition need real-only ops
**Fix:** Similar to ILU - keep ASM real-only for now

## Architecture Notes

### Type Parameter Shadowing
The crate uses a global type alias `S` (see `algebra/prelude.rs`):
```rust
#[cfg(feature = "complex")]
pub type S = Complex<f64>;
#[cfg(not(feature = "complex"))]
pub type S = f64;
```

When adding generic implementations, **use different names** (e.g., `Scalar`) for type parameters to avoid shadowing and confusing the compiler about which `S` is which.

### Feature-Gated Implementations
For complex-only code paths:
- Use `#[cfg(feature = "complex")]` guards
- Implement `KPreconditioner` trait for complex scalar support  
- Fall back to simpler PC implementations that don't require deep matrix introspection

## Testing Recommendations

1. **Build with complex feature:**
   ```bash
   cargo build --example complex_matrix_market_demo --features=complex,mpi,mpi_examples
   ```

2. **Check that real case still works:**
   ```bash
   cargo build --example complex_matrix_market_demo --features=mpi,mpi_examples
   ```

3. **Unit test complex operators:**
   - `Mat<Complex64>` LinOp implementation
   - `CsrOp<Complex64>` spmv and t_matvec
   - Jacobi preconditioner with complex vectors

## Code Quality Checklist

- [ ] All `Mat<S>` operations maintain generic scalar support
- [ ] CsrOp and CsrMatrix operations compile for `S = Complex64`
- [ ] ILU explicitly documents real-only limitation
- [ ] BlockJacobi properly bridges complex ↔ real for apply path
- [ ] Example builds without errors
- [ ] All existing tests still pass
