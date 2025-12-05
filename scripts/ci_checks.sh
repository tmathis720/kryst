#!/usr/bin/env bash
set -euo pipefail

# Run the basic lint/test suite that should be fast enough to run before pushing.
cargo fmt --all -- --check
cargo clippy --all-targets --all-features
cargo test --all-features
