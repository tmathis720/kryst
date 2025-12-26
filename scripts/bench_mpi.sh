#!/usr/bin/env bash
set -euo pipefail

MPI_RANKS="${1:-2}"

mpirun -n "${MPI_RANKS}" cargo bench --features mpi --bench gmres_variants
