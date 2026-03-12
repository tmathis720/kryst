# MPI × Rayon × Solver-Variant Tuning

This note summarizes practical settings when running Krylov methods with MPI + Rayon and the newer pipelined/fused reduction paths.

## 1) Rank/thread topology

- Prefer **1 MPI rank per NUMA domain** (or socket) and set Rayon threads to local cores.
- Kryst thread-pool sizing now honors local-rank hints (`OMPI_COMM_WORLD_LOCAL_SIZE`, `MPI_LOCALNRANKS`, `SLURM_NTASKS_PER_NODE`) and computes per-rank threads from `KRYST_THREADS` / `RAYON_NUM_THREADS` / `SLURM_CPUS_PER_TASK`.
- Override explicitly when needed:
  - `KRYST_THREADS=<total cores per node>`
  - `KRYST_MPI_LOCAL_SIZE=<ranks per node>`

## 2) Reduction-heavy solver guidance

`SolveStats.counters.num_global_reductions` gives measured counts, and `SolveStats.reduction_model` gives expected startup/per-iteration behavior.

Recommended defaults:

- `PcgVariant::Classic`: best when latency is low and arithmetic intensity dominates.
- `PcgVariant::Pipelined`: best for high-latency MPI; ~1 global reduction/iter model.
- `GmresVariant::Classical`: robust baseline.
- `GmresVariant::Pipelined`: prefer when restart is moderate and collectives are expensive.
- `FgmresVariant::Pipelined`: preferred with variable/nonlinear preconditioners under MPI latency pressure.

## 3) Suggested matrix

| MPI regime | Rayon threads/rank | Solver variant |
|---|---:|---|
| Intra-node only (`size=1`) | all local cores | classical or pipelined (usually close) |
| Multi-node, low latency fabric | 4–16 | classical GMRES/PCG unless sync stalls visible |
| Multi-node, higher latency / congested | 2–8 | pipelined PCG/GMRES/FGMRES |
| Strong-scaling tail (tiny local subproblems) | 1–4 | pipelined (minimize syncs), or switch to block methods |

## 3.5) Distributed Krylov variants with safe reduction pipelining

Use reduced-sync variants only where the recurrences are mathematically valid for the operator/PC pairing:

- `CgVariant::Pipelined` (CG): SPD operator, left HPD preconditioner.
- `PcgVariant::Pipelined { replace_every }` (PCG): same SPD/left-HPD assumptions.
- `GmresVariant::Pipelined` and `FgmresVariant::Pipelined`: general nonsymmetric systems with Arnoldi-based fused reductions.
- `BiCgStabVariant::LowSync`: nonsymmetric systems when lower synchronization is desired over strict classical recurrence behavior.

Practical drift control for long runs:

- `-ksp_cg_replace_every <k>` now controls periodic residual refresh/replacement in pipelined CG-family paths (PCG replacement and CG residual recomputation).
- Smaller `k` improves robustness and stopping consistency; larger `k` minimizes extra matvec/reduction overhead.

Stopping-criteria/reproducibility notes:

- Iterative checks continue to use each solver's reported norm semantics (preconditioned/unpreconditioned/natural), while end-of-solve diagnostics use true residual norms where implemented.
- In distributed mode, prefer `-ksp_reproducible true` when regression stability is required; expect slower reductions due to deterministic ordering.
- For throughput runs, keep reproducibility off and use periodic refresh only when residual drift appears in monitor histories.

## 4) Benchmarking

Use:

- `cargo bench --bench pcg_pipelined_scaling`
- `cargo bench --bench solver_reduction_scaling`

Compare both runtime and `num_global_reductions` trends across strong/weak scaling points.


## 5) Nested KSP + FieldSplit tuning quick picks

- For mixed outer/inner solvers under MPI, use **outer FGMRES + inner GMRES/Jacobi** first:
  - `-ksp_type fgmres -pc_type ksp -pc_ksp_ksp_type gmres -pc_ksp_pc_type jacobi`
  - Set inner monitor behavior with `-pc_ksp_monitor_policy rank0` (or compatibility flag `-pc_ksp_monitor_rank0 true`) to reduce log fanout.
  - Tight coupling control: `-pc_ksp_allow_maxits false` forces inner max-its to fail the nested PC, and `-pc_ksp_propagate_converged_reason true` keeps the exact inner reason in `SolveStats.nested_pc_failure`.
- For block-coupled operators in distributed runs:
  - `-pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_fact_type full`
  - Prefer `-pc_fieldsplit_schur_precondition self|self_p|full|matfree` for complex builds.
- When outer/inner sides differ, set inner explicitly with `-pc_ksp_pc_side` rather than relying on outer-side inheritance.

These settings pair well with pipelined outer Krylov variants on higher-latency fabrics.


## 6) MG/GAMG + distributed PC route playbook (recommended defaults for large MPI jobs)

For large runs (`O(10^2+)` ranks), prefer DistCsr-native kernels first and keep adapter paths as explicit fallback-only routes.

Recommended defaults:

- Keep native distributed route preference:
  - `-pc_dist_route native`
  - `-pc_dist_local_apply distributed_native` (or `hybrid` if coarse coupling helps)
- Let distributed Block-Jacobi auto-promote when `DistCsrOp` + communicator constraints are satisfied:
  - keep `-pc_global none` unless you need to force `asm|ras|block_jacobi`
- Coarse correction route defaults:
  - `-pc_amg_dist_coarse_solver_route auto`
  - `-pc_amg_dist_coarse_policy root` (switch to `local` after validation)
  - `-pc_mg_dist_coarse_repartition keep` initially, then test `uniform`

Diagnostics to verify native routing:

- `KSPView` / PC diagnostics now include route-selection fields:
  - `pc_dist_selected_route`
  - `pc_dist_fallback_chain`
  - `pc_dist_fallback_reason` (when fallback is used)
- AMG/MG diagnostics should show coarse-route selection and fallback chain that match your policy.

Fallback policy guidance:

1. Start native-first (`pc_dist_route=native`) and inspect diagnostics for `distcsr_native_block_jacobi:*`.
2. If native setup fails on a path, keep fallback explicit (`pc_dist_route=adapted`) only for that workload.
3. Track fallback frequency in production runs; repeated fallback means tune local PC/coarse settings before scaling out.
