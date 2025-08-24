# Adding a preconditioner

- Implement `Preconditioner` (respect `apply` contract).
- Declare builder in `preconditioner/builders.rs`.
- Map options in `PcConfig::from_type_and_options`.
- Add unit tests + a doctest example.
- If distributed: read comm from `a.comm()` in `setup`.
- If supports reuse: implement `supports_numeric_update`/`update_numeric`.

