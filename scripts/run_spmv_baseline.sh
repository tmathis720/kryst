#!/usr/bin/env bash
set -euo pipefail

SIZE="${KRYST_M0_SIZE:-16384}"
ITERATIONS="${KRYST_M0_ITERATIONS:-30}"
MAX_MPI_RANKS="${KRYST_M0_MAX_MPI_RANKS:-8}"
OUTPUT_DIR="${KRYST_M0_OUTPUT_DIR:-benchmarks/milestone0/artifacts}"
RAW_DIR="${OUTPUT_DIR}/raw"

mkdir -p "${RAW_DIR}"

cargo build --release --features "simd,transpose-cache" --bin spmv_baseline
target/release/spmv_baseline --size "${SIZE}" --iterations "${ITERATIONS}" \
  > "${RAW_DIR}/shared-real.jsonl"

cargo build --release --features "complex,simd,transpose-cache" --bin spmv_baseline
target/release/spmv_baseline --size "${SIZE}" --iterations "${ITERATIONS}" \
  > "${RAW_DIR}/shared-complex.jsonl"

cargo build --release --no-default-features --features "backend-sprs" --bin spmv_baseline
target/release/spmv_baseline --size "${SIZE}" --iterations "${ITERATIONS}" \
  > "${RAW_DIR}/shared-sprs.jsonl"

if command -v mpirun >/dev/null 2>&1; then
  cargo build --release --no-default-features --features "backend-faer,mpi,simd" --bin spmv_baseline
  for ranks in 1 2 4 8; do
    if (( ranks <= MAX_MPI_RANKS )); then
      mpirun -n "${ranks}" --oversubscribe target/release/spmv_baseline \
        --size "${SIZE}" --iterations "${ITERATIONS}" --distributed-only \
        > "${RAW_DIR}/mpi-real-r${ranks}.jsonl"
    fi
  done

  cargo build --release --no-default-features --features "backend-faer,mpi,rayon,simd" --bin spmv_baseline
  for ranks in 1 2 4 8; do
    if (( ranks <= MAX_MPI_RANKS )); then
      mpirun -n "${ranks}" --oversubscribe target/release/spmv_baseline \
        --size "${SIZE}" --iterations "${ITERATIONS}" --distributed-only \
        > "${RAW_DIR}/hybrid-real-r${ranks}.jsonl"
    fi
  done

  cargo build --release --no-default-features --features "backend-faer,mpi,complex" --bin spmv_baseline
  for ranks in 1 2 4 8; do
    if (( ranks <= MAX_MPI_RANKS )); then
      mpirun -n "${ranks}" --oversubscribe target/release/spmv_baseline \
        --size "${SIZE}" --iterations "${ITERATIONS}" --distributed-only \
        > "${RAW_DIR}/mpi-complex-r${ranks}.jsonl"
    fi
  done

  cargo build --release --no-default-features --features "backend-faer,mpi,rayon,complex" --bin spmv_baseline
  for ranks in 1 2 4 8; do
    if (( ranks <= MAX_MPI_RANKS )); then
      mpirun -n "${ranks}" --oversubscribe target/release/spmv_baseline \
        --size "${SIZE}" --iterations "${ITERATIONS}" --distributed-only \
        > "${RAW_DIR}/hybrid-complex-r${ranks}.jsonl"
    fi
  done
else
  echo "mpirun is unavailable; MPI rank baselines were skipped" >&2
fi

python3 scripts/aggregate_spmv_baseline.py "${RAW_DIR}"/*.jsonl \
  --output "${OUTPUT_DIR}/latest.json"
echo "wrote ${OUTPUT_DIR}/latest.json"
