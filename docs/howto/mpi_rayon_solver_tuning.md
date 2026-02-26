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
