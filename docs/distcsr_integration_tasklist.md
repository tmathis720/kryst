# DistCsr Integration Task List

This task list targets stronger use of fast distributed paths in Krylov solves and preconditioner application when `DistCsrOp` is available.

## 1) Baseline and observability hardening

- [ ] Add a reproducible MPI benchmark matrix set (Poisson 2D/3D, power-law, block systems) with fixed partition seeds and documented expected convergence profiles.
- [ ] Promote distributed route telemetry to first-class benchmark output:
  - `pc_dist_route_policy`
  - `pc_dist_selected_route`
  - `pc_dist_fallback_chain`
  - `pc_dist_fallback_reason`
  - `pc_dist_fallback_counters`
- [ ] Add per-iteration timing split for `matvec`, `halo`, `pc_apply`, `global_reduction`, and `other` in distributed runs.
- [ ] Gate regressions with CI thresholds for native-route usage and fallback frequency.

## 2) DistCsr route selection and setup improvements

- [ ] Centralize distributed route policy resolution (`native`, `adapted`, `root_gather`) into one audited decision function with explicit reason codes.
- [ ] Add “native feasibility preflight” checks before setup (layout completeness, communicator size, halo readiness, PC compatibility).
- [ ] Cache preflight outcome in setup metadata to avoid repeated dynamic probing in iterative solves.
- [ ] Surface actionable setup diagnostics when native setup is rejected (which condition failed and nearest valid configuration).

## 3) Fast-path implementation expansion in preconditioners

- [ ] Expand DistCsr-native apply kernels for local-PC families currently relying on wrapped-local behavior under MPI.
- [ ] Implement shared distributed local-apply utilities used by Block-Jacobi + ILU/SOR/Chebyshev variants to remove duplicated control logic.
- [ ] Add overlap-aware apply options (e.g., additive correction + one-step neighbor refresh) for latency-sensitive preconditioners.
- [ ] Ensure strict mode (`pc_dist_local_apply=strict`) produces deterministic errors and zero silent adaptation.

## 4) Halo exchange and communication optimization

- [ ] Introduce persistent communication plans for halo exchange (reused send/recv schedules and buffers across iterations).
- [ ] Add nonblocking overlap mode: launch halo exchange, compute interior work, then finish boundary work.
- [ ] Add message coalescing and neighbor ordering heuristics to reduce small-message overhead.
- [ ] Benchmark and tune eager/rendezvous cutovers for common problem sizes.

## 5) Solver-level distributed efficiency

- [ ] Reduce global synchronizations in distributed Krylov variants where mathematically valid (pipeline-safe paths).
- [ ] Add optional residual replacement / periodic recomputation policy for long runs to control drift in reduced-sync modes.
- [ ] Revisit stopping criteria handling so distributed norms and local diagnostics stay consistent and cheap.
- [ ] Validate reproducible mode interaction with optimized reductions and document expected performance delta.

## 6) Data layout and kernel performance

- [ ] Audit DistCsr local storage and access patterns for cache locality and SIMD-friendliness.
- [ ] Add a layout-aware kernel selector (scalar / SIMD gather / SELL-C-σ-like local kernels) specific to distributed local blocks.
- [ ] Introduce prepacked local row blocks for repeated apply-heavy workloads.
- [ ] Add counters for interior-vs-boundary row cost and nnz skew to guide kernel selection.

## 7) Robustness and correctness matrix

- [ ] Build a distributed correctness matrix crossing:
  - solver type (`cg`, `gmres`, `fgmres`, `bicgstab`, etc.)
  - global PC (`block_jacobi`, `asm`, `ras`)
  - local PC (`ilu*`, `sor`, `chebyshev`, `jacobi`)
  - route policy (`native`, `adapted`)
- [ ] Add stress tests for pathological partitions (empty ranks, highly imbalanced nnz, disconnected subdomains).
- [ ] Add fault-injection-style tests for failed native setup and verify fallback chain/reporting semantics.
- [ ] Add deterministic replay hooks for distributed setup/apply failures.

## 8) API and configuration clarity

- [ ] Document “how to force native path” and “how to debug fallback” playbooks in one concise guide.
- [ ] Add option validation rules that warn early on contradictory distributed settings.
- [ ] Add a machine-readable “effective distributed config” section in solver view output.
- [ ] Standardize terminology (`native distributed`, `adapted local wrapper`, `gather route`) across docs and logs.

## 9) Rollout plan

- [ ] Phase 1: instrumentation + route diagnostics + benchmark baselines.
- [ ] Phase 2: halo and apply fast-path performance work.
- [ ] Phase 3: solver reduction optimizations + robustness matrix completion.
- [ ] Phase 4: defaults tuning (prefer native aggressively where safe) and deprecate weak adapter-only patterns.

## 10) Success criteria (exit checks)

- [ ] Native distributed route selected by default in >95% of supported DistCsr benchmark cases.
- [ ] ≥20% end-to-end speedup on medium/large MPI cases versus current baseline for representative solver+PC combos.
- [ ] No regressions in convergence quality versus baseline within defined tolerance bands.
- [ ] Fallbacks are explicit, rare, and diagnosable from one solver-view snapshot.
