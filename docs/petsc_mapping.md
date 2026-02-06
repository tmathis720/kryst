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

| PETSc `-ksp_type` | kryst `SolverType` | Notes |
| --- | --- | --- |
| `richardson` | `SolverType::Richardson` | Supports optional `-ksp_richardson_omega` |
| `chebyshev` | `SolverType::Chebyshev` | Exposed as solver mode; requires left PC side |
| `cr` | `SolverType::Cr` | Current implementation adapts the CGNR kernel |
| `tcqmr` | `SolverType::Tcqmr` | Current implementation adapts TFQMR |
| `gcr` | `SolverType::Gcr` | Backed by flexible GMRES |
| `pipegcr` / `gcr_pipe` | `SolverType::PipeGcr` | Parsed but returns explicit unsupported error |

| `PCFIELDSPLIT` | `PcType::FieldSplit` + `PcOptions.pc_fieldsplit_*` |
| `PCSHELL` | `PcType::Shell` + `PcOptions.pc_shell_name` + `register_shell_callback` |
| `PCKSP` | `PcType::Ksp` + `PcOptions.pc_ksp_*` |
| `PCMG` | `PcType::Mg` + `PcOptions.pc_mg_*` |
| `PCBDDC` | `PcType::Bddc` (placeholder; explicit unsupported error) |
