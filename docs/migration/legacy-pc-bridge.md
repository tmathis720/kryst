# Legacy PC bridge

The `legacy-pc-bridge` feature exposes `LegacyOpPreconditioner` for adapting
old matrix/vector APIs. It incurs per-apply copies; prefer native PCs when
possible.

