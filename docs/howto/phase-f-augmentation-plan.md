# Phase F: GMRES/FGMRES Augmentation & Deflation Plan

This plan extends the existing Phase C GMRES/FGMRES implementation with production-ready augmentation and deflation features. It preserves the current public API, works with the async reduction and block primitives already in place, and outlines tests required for validation.

## 1. Public API & Wiring

### 1.1 Solver options

```rust
#[derive(Clone, Debug)]
pub enum AugmentationPolicy {
    None,
    /// GMRES-DR(m,k): carry k harmonic Ritz vectors across restarts
    GmresDR { k: usize },
    /// LGMRES(m,ℓ): carry ℓ recent solution-difference (or preconditioned) vectors
    Lgmres { ell: usize },
}

pub struct GmresOptions {
    pub restart: usize,
    pub augment: AugmentationPolicy,
    pub reorth: ReorthPolicy,
    // existing fields remain unchanged
}
```

* Add identical plumbing for FGMRES (right-preconditioned variant).
* Introduce configuration flags:
  * `-ksp_gmres_augment none|dr|lgmres`
  * `-ksp_gmres_dr_k <int>`
  * `-ksp_lgmres_ell <int>`

## 2. RecyclingSpace in the workspace

Embed a `RecyclingSpace` object inside the GMRES/FGMRES cycle workspace.

```rust
pub struct RecyclingSpace {
    /// Orthonormal columns (size n × r), r ≤ rmax.
    /// GMRES stores physical-space vectors; right-PC FGMRES stores Z-space vectors.
    pub U: Vec<Vec<f64>>,
    /// A*U in physical space (needed to seed Arnoldi and build the Hessenberg block).
    pub AU: Vec<Vec<f64>>,
    pub rmax: usize,
    pub kind: AugmentationPolicy,
}
```

* Provide `new`, `clear`, and `r()` helpers.
* Thread `RecyclingSpace` into `GmresCycleWs` and `FgmresCycleWs`.

## 3. Where augmentation enters the algorithm

1. At restart (or the first cycle) refresh `recycle.U` and `recycle.AU` according to the active policy.
2. Prepend the block `U` as the initial columns of the new Arnoldi basis and seed the block with `AU`.
3. Continue the usual Arnoldi process from the deflated residual; the least-squares problem remains unchanged.

## 4. GMRES-DR implementation

1. Build harmonic Ritz vectors from the completed cycle:
   * Form `C = [H_m; h_{m+1,m} e_m^T]`.
   * Compute its small SVD via `faer` and select the `k` right singular vectors associated with the smallest singular values (`Y_k`).
   * Compute `U = V_m Y_k` and orthonormalize with two-pass MGS.
   * Compute `AU = V_{m+1} \bar H_m Y_k`; fall back to explicit matvecs if the cached data are unavailable.
2. Install the block into `ws.recycle` and call `prepend_block` before starting the next Arnoldi cycle.
3. Reduce `k` if the SVD is ill-conditioned or `k == 0`.

## 5. LGMRES implementation

1. During each cycle collect up to `ℓ` augmentation vectors:
   * GMRES: solution differences `d_j = x_{j+1} - x_j`.
   * FGMRES (right-PC): the preconditioned search directions `z_j`.
2. At restart:
   * Orthonormalize the collected vectors (two-pass MGS, drop tiny norms).
   * Compute `AU = A U` with a batched SpMM/SpMV.
   * Install the block into `ws.recycle` and call `prepend_block`.
3. This approach matches classical LGMRES while reusing the standard Arnoldi and least-squares path.

## 6. Mechanics to add

### 6.1 Block prepend helper

Implement `prepend_block` to consume `(U, AU)` and produce the first `r` basis vectors and the leading block of the Hessenberg matrix.

* Reuse the existing block GMRES routines: compute Gram matrices with a batched reduction, perform Cholesky-QR, and fall back to Householder QR when conditioning is poor.
* Populate `V[0..r-1]` with the orthonormalized columns and write the upper block of `H` using the resulting `R` factor.
* Continue the Arnoldi loop starting at column `r`.

### 6.2 Orthogonalization

No special handling is required after the prepend; subsequent vectors already orthogonalize against the augmented basis via the standard code paths.

## 7. Right-preconditioned FGMRES specifics

* GMRES-DR: compute harmonic Ritz vectors in the unpreconditioned space but store augmentation vectors in `Z`-space (`U_Z = Z_m Y_k`) while `AU = A U_Z`.
* LGMRES: capture and recycle the preconditioned search directions (`z_j`).
* Ensure solution updates use the `Z` basis exactly as in Phase C.

## 8. Multi-solve recycling (RecyclingKsp)

Provide an optional wrapper that persists augmentation across solves.

```rust
pub struct RecyclingKsp<S: LinearSolver> {
    inner: S,
    space: RecyclingSpace,
    policy: AugmentationPolicy,
}
```

* Add `install_recycling` and `export_recycling` hooks on GMRES/FGMRES.
* Merge exported vectors into the persistent space with eviction when exceeding `rmax`.

## 9. Numerics & fallbacks

* Two-pass MGS for all augmentation columns; drop vectors with norm `< 1e-12` of their initial norm.
* Guard Cholesky-QR with a condition-number test (`cond(R) ≤ 1e8`); otherwise fall back to Householder QR.
* Clamp `k` when the GMRES-DR SVD is rank deficient or singular values cluster.
* Keep `U` in `Z`-space for right-preconditioned FGMRES while maintaining `AU = A U` in physical space.
* Fix column ordering (ascending singular values, then `‖AU_i‖`) for deterministic behaviour.

## 10. Tests & benchmarks

1. **Augmentation efficacy**: classical GMRES(m) stagnation problems; compare iterations and residuals versus DR/LGMRES and ensure reduction counts only increase by the prepend reduction.
2. **Right-PC parity**: validate DR/LGMRES with AMG (left PC) and ILU (right PC); confirm solution updates use the `Z` basis in small-dimension checks.
3. **Recycling across solves**: multiple RHS or Newton steps; verify monotonic iteration reductions and enforcement of `rmax`.
4. **Stability & guards**: trigger near-dependent augmentation columns to exercise QR fallbacks and SVD conditioning logic.

---

Delivering these steps yields a low-risk, production-ready augmentation phase with dramatic improvements on difficult restarted problems and optional multi-solve recycling without altering the existing external API.
