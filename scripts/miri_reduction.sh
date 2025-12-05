#!/usr/bin/env bash
set -euo pipefail

# Run Miri on the critical units to catch UB in the buffer pool and reduction kernels.
rustup toolchain install nightly >/dev/null
rustup +nightly component add miri >/dev/null

cargo +nightly miri test buffer_pool dot_engine
