# DistCSR benchmark fixtures

Replayable distributed benchmark assets for CI comparisons.

- `fixtures.json`: matrix + solver definitions with fixed process counts and partition seeds.
- `expectations.json`: expected iteration/residual bands and route-selection expectations.
- `artifacts/latest.json`: normalized runner output for baseline comparison.

Run:

```bash
cargo run --bin distcsr_bench_runner --features mpi -- \
  --fixtures benchmarks/distcsr/fixtures.json \
  --expectations benchmarks/distcsr/expectations.json \
  --output benchmarks/distcsr/artifacts/latest.json
```
