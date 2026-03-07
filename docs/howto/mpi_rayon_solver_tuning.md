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


## 6) MG/GAMG distributed coarse route playbook

When tuning multigrid in distributed mode, set route controls explicitly and verify them in hierarchy diagnostics:

- Primary route selection:
  - `-pc_amg_dist_coarse_solver_route auto|root|local|superlu_dist`
  - MG aliases also map: `-pc_mg_coarse_solver_route ...`
- Strategy policy:
  - `-pc_amg_dist_coarse_policy root|local|superlu_dist|none`
- Repartition policy:
  - `-pc_mg_dist_coarse_repartition keep|uniform|root`

Suggested sequence:

1. Start with `route=auto`, `policy=root` for robust setup.
2. Move to `policy=local` and `route=local,root` once communication dominates.
3. Keep a fallback route list (`local,root` or `root,local`) so setup can recover predictably.
4. Use AMG/MG stats to confirm selected route, fallback chain, per-level nnz, and smoothing work.
