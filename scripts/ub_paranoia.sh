#!/usr/bin/env bash
set -euo pipefail

# Run a focused ASan-enabled test harness for the crates we worry about.
export RUSTFLAGS="-Zsanitizer=address"
export RUSTDOCFLAGS="-Zsanitizer=address"
export ASAN_OPTIONS="detect_leaks=0"

# Nightly toolchain is required because ASan is unstable.
rustup toolchain install nightly >/dev/null

cargo +nightly test buffer_pool dot_engine
