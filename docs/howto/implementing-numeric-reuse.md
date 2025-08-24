# Implementing numeric reuse

- Expose `supports_numeric_update` returning `true`.
- Keep structure stable; update values in `update_numeric`.
- Fall back to `update_symbolic` when structure changes.
- Callers check `supports_numeric_update` before invoking reuse.

