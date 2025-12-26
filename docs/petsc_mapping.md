# PETSc to kryst mapping

The table below maps common PETSc KSP/PC calls to their kryst equivalents.

| PETSc concept | kryst equivalent |
| --- | --- |
| `KSPSetOperators` | `KspContext::set_operators` / `try_set_operators` |
| `KSPSetUp` | `KspContext::setup` |
| `KSPSolve` | `KspContext::solve` |
| `KSPGetConvergedReason` | `SolveStats::reason` |
| `KSPSetType` | `KspContext::set_type` (`SolverType`) |
| `PCSetType` | `KspContext::set_pc_type` (`PcType`) |
| `PCSetFromOptions` | `KspContext::set_from_options` / `set_from_all_options` |
| `PCSide` | `PcSide` |
