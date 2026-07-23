# Milestone 0 SpMV baseline

`artifacts/latest.json` is a reproducible snapshot produced by:

```bash
KRYST_M0_SIZE=4096 KRYST_M0_ITERATIONS=10 scripts/run_spmv_baseline.sh
```

The checked-in snapshot is intentionally modest so it can be regenerated on a
developer machine. For a stable performance comparison, increase both values,
keep the host otherwise idle, and preserve the generated environment metadata.
Raw JSONL files retain each feature configuration; `baseline.json` combines
them and calculates MPI strong-scaling efficiency whenever matching one-rank
and multi-rank runs exist.

The runner automatically covers one, two, four, and the maximum available
Rayon thread count. It covers MPI ranks 1, 2, 4, and 8 when `mpirun` and enough
ranks are available; set `KRYST_M0_MAX_MPI_RANKS` to match the host or CI quota.
The source inventory, call-path audit, metric definitions, caveats, and larger
benchmark commands are in [`../../docs/milestone0-source-audit.md`](../../docs/milestone0-source-audit.md).
