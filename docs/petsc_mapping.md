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

| `PCFIELDSPLIT` | `PcType::FieldSplit` + `PcOptions.pc_fieldsplit_*` |
| `PCSHELL` | `PcType::Shell` + `PcOptions.pc_shell_name` + `register_shell_callback` |
| `PCKSP` | `PcType::Ksp` + `PcOptions.pc_ksp_*` |
| `PCMG` | `PcType::Mg` + `PcOptions.pc_mg_*` |
| `PCBDDC` | `PcType::Bddc` (placeholder; explicit unsupported error) |
