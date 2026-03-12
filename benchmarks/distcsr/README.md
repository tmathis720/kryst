# DistCSR benchmark fixtures

Replayable distributed benchmark assets for CI comparisons.

- `fixtures.json`: matrix + solver definitions with fixed process counts and partition seeds.
- `expectations.json`: expected iteration/residual bands and route-selection expectations.
- `../../ci/distcsr_thresholds.json`: CI guardrail targets for rollout gates (native default rate, fallback rate, speedup floor, convergence bands, fallback diagnostics completeness).
- `artifacts/latest.json`: normalized runner output for baseline comparison.

Run:

```bash
cargo run --bin distcsr_bench_runner --features mpi -- \
  --fixtures benchmarks/distcsr/fixtures.json \
  --expectations benchmarks/distcsr/expectations.json \
  --output benchmarks/distcsr/artifacts/latest.json
```

Optional timing instrumentation modes:

- `--timing-detail off`: disables timing collection output.
- `--timing-detail basic` (default): emits solve wall-time and per-category totals/averages (`matvec`, `halo`, `pc_apply`, `global_reduction`, `other`) with minimal overhead suitable for CI.
- `--timing-detail high`: reserved for higher-detail timing output; currently same overhead profile as `basic` unless additional profiling is enabled in a metrics build.

When built without `metrics`, category timings other than `other` are zero-filled and `other` captures wall-time remainder; this keeps default CI runs cheap while preserving a stable output schema.

CI rollout scoreboard gate:

```bash
python3 scripts/check_distcsr_thresholds.py \
  --artifact benchmarks/distcsr/artifacts/latest.json \
  --thresholds ci/distcsr_thresholds.json
```

The checker exits non-zero with explicit messages for regressions in:

- native-route default rate,
- fallback frequency,
- speedup target vs baseline timings,
- convergence non-regression bands, and
- fallback diagnostic snapshot completeness.

To enable the speedup/convergence gates, include baseline values in each case's `details` object in the artifact (`baseline_solve_ms`, `baseline_iterations`, `baseline_final_residual`, etc.).
