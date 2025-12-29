#!/usr/bin/env bash
set -euo pipefail

MPI_RANKS="${1:-2}"
THREADS="${2:-4}"

echo "==> Rayon-only (${THREADS} threads)"
KRYST_BENCH_THREADS="${THREADS}" \
  cargo bench --no-default-features --features "rayon backend-faer" --bench mpi_rayon_suite

echo "==> MPI-only (${MPI_RANKS} ranks, 1 thread)"
mpirun -n "${MPI_RANKS}" env KRYST_BENCH_THREADS=1 \
  cargo bench --no-default-features --features "mpi backend-faer" --bench mpi_rayon_suite

echo "==> Hybrid (${MPI_RANKS} ranks, ${THREADS} threads)"
mpirun -n "${MPI_RANKS}" env KRYST_BENCH_THREADS="${THREADS}" \
  cargo bench --no-default-features --features "mpi rayon backend-faer" --bench mpi_rayon_suite
