# PETSc to kryst mapping

This page expands the PETSc → kryst mapping into a full compatibility matrix for KSP/PC
features, options, and monitoring/convergence hooks.

## Status labels
- **Supported**: Implemented with PETSc-equivalent behavior.
- **Partial**: Implemented with reduced scope or a simplified algorithm.
- **Unsupported**: Not yet implemented; see linked tracking issue.

## Core lifecycle and setup

| PETSc concept | kryst equivalent | Status | Notes/Tracking |
| --- | --- | --- | --- |
| `KSPSetOperators` | [`KspContext::set_operators`](https://docs.rs/kryst/latest/kryst/context/struct.KspContext.html#method.set_operators) / [`KspContext::try_set_operators`](https://docs.rs/kryst/latest/kryst/context/struct.KspContext.html#method.try_set_operators) | Supported | `try_set_operators` enforces communicator congruence. |
| `KSPSetUp` | [`KspContext::setup`](https://docs.rs/kryst/latest/kryst/context/struct.KspContext.html#method.setup) | Supported | Idempotent; respects structure/value reuse. |
| `KSPSolve` | [`KspContext::solve`](https://docs.rs/kryst/latest/kryst/context/struct.KspContext.html#method.solve) | Supported | Returns [`SolveStats`](https://docs.rs/kryst/latest/kryst/utils/convergence/struct.SolveStats.html). |
| `KSPGetConvergedReason` | [`SolveStats::reason`](https://docs.rs/kryst/latest/kryst/utils/convergence/struct.SolveStats.html#structfield.reason) | Supported | See convergence table below. |
| `KSPSetType` | [`KspContext::set_type`](https://docs.rs/kryst/latest/kryst/context/struct.KspContext.html#method.set_type) + [`SolverType`](https://docs.rs/kryst/latest/kryst/context/ksp_context/enum.SolverType.html) | Supported | See KSP method matrix below. |
| `PCSetType` | [`KspContext::set_pc_type`](https://docs.rs/kryst/latest/kryst/context/struct.KspContext.html#method.set_pc_type) + [`PcType`](https://docs.rs/kryst/latest/kryst/context/pc_context/enum.PcType.html) | Supported | See PC method matrix below. |
| `KSP/PCSetFromOptions` | [`KspContext::set_from_options`](https://docs.rs/kryst/latest/kryst/context/struct.KspContext.html#method.set_from_options) / [`set_from_all_options`](https://docs.rs/kryst/latest/kryst/context/struct.KspContext.html#method.set_from_all_options) | Supported | Uses [`KspOptions`](https://docs.rs/kryst/latest/kryst/config/options/struct.KspOptions.html) + [`PcOptions`](https://docs.rs/kryst/latest/kryst/config/options/struct.PcOptions.html). |
| `PCSide` | [`PcSide`](https://docs.rs/kryst/latest/kryst/preconditioner/enum.PcSide.html) | Supported | `pc_side` is normalized per solver rules. |

## KSP method matrix

| PETSc `-ksp_type` | kryst API | Status | Notes/Tracking |
| --- | --- | --- | --- |
| `cg` | [`SolverType::Cg`](https://docs.rs/kryst/latest/kryst/context/ksp_context/enum.SolverType.html#variant.Cg) | Supported | Left preconditioning only. |
| `cgnr` | [`SolverType::Cgnr`](https://docs.rs/kryst/latest/kryst/context/ksp_context/enum.SolverType.html#variant.Cgnr) | Supported | Uses CGNR kernel. |
| `gmres` | [`SolverType::Gmres`](https://docs.rs/kryst/latest/kryst/context/ksp_context/enum.SolverType.html#variant.Gmres) | Supported | Supports restart, orthog, reorthog, variants. |
| `fgmres` | [`SolverType::Fgmres`](https://docs.rs/kryst/latest/kryst/context/ksp_context/enum.SolverType.html#variant.Fgmres) | Supported | Supports restart and orthog options. |
| `bicgstab` | [`SolverType::BiCgStab`](https://docs.rs/kryst/latest/kryst/context/ksp_context/enum.SolverType.html#variant.BiCgStab) | Supported | Standard BiCGStab implementation. |
| `cgs` | [`SolverType::Cgs`](https://docs.rs/kryst/latest/kryst/context/ksp_context/enum.SolverType.html#variant.Cgs) | Supported | Monitors report true residual. |
| `pcg` | [`SolverType::Pcg`](https://docs.rs/kryst/latest/kryst/context/ksp_context/enum.SolverType.html#variant.Pcg) | Supported | Pipelined variant available via `-ksp_cg_pipelined`. |
| `minres` | [`SolverType::Minres`](https://docs.rs/kryst/latest/kryst/context/ksp_context/enum.SolverType.html#variant.Minres) | Supported | Left preconditioning only. |
| `lsqr` | [`SolverType::Lsqr`](https://docs.rs/kryst/latest/kryst/context/ksp_context/enum.SolverType.html#variant.Lsqr) | Supported | Left preconditioning only. |
| `lsmr` | [`SolverType::Lsmr`](https://docs.rs/kryst/latest/kryst/context/ksp_context/enum.SolverType.html#variant.Lsmr) | Supported | Left preconditioning only. |
| `pca_gmres` | [`SolverType::PcaGmres`](https://docs.rs/kryst/latest/kryst/context/ksp_context/enum.SolverType.html#variant.PcaGmres) | Supported | Maps PETSc PCA-PC modes. |
| `qmr` | [`SolverType::Qmr`](https://docs.rs/kryst/latest/kryst/context/ksp_context/enum.SolverType.html#variant.Qmr) | Supported | Complex builds only. |
| `tfqmr` | [`SolverType::Tfqmr`](https://docs.rs/kryst/latest/kryst/context/ksp_context/enum.SolverType.html#variant.Tfqmr) | Supported | Complex builds only. |
| `tcqmr` | [`SolverType::Tcqmr`](https://docs.rs/kryst/latest/kryst/context/ksp_context/enum.SolverType.html#variant.Tcqmr) | Partial | Adapted TFQMR kernel. |
| `richardson` | [`SolverType::Richardson`](https://docs.rs/kryst/latest/kryst/context/ksp_context/enum.SolverType.html#variant.Richardson) | Supported | `-ksp_richardson_omega` supported. |
| `chebyshev` | [`SolverType::Chebyshev`](https://docs.rs/kryst/latest/kryst/context/ksp_context/enum.SolverType.html#variant.Chebyshev) | Partial | Chebyshev-as-KSP mode with fixed omega. |
| `cr` | [`SolverType::Cr`](https://docs.rs/kryst/latest/kryst/context/ksp_context/enum.SolverType.html#variant.Cr) | Partial | Backed by CGNR kernel. |
| `gcr` | [`SolverType::Gcr`](https://docs.rs/kryst/latest/kryst/context/ksp_context/enum.SolverType.html#variant.Gcr) | Partial | Implemented via flexible GMRES. |
| `pipegcr` / `gcr_pipe` | [`SolverType::PipeGcr`](https://docs.rs/kryst/latest/kryst/context/ksp_context/enum.SolverType.html#variant.PipeGcr) | Supported | Pipelined GCR via reduced-reduction FGMRES kernel. |
| `preonly` | [`SolverType::Preonly`](https://docs.rs/kryst/latest/kryst/context/ksp_context/enum.SolverType.html#variant.Preonly) | Supported | Uses `Preconditioner::direct_solve` when available. |

## PC method matrix

| PETSc `-pc_type` | kryst API | Status | Notes/Tracking |
| --- | --- | --- | --- |
| `none` | [`PcType::None`](https://docs.rs/kryst/latest/kryst/context/pc_context/enum.PcType.html#variant.None) | Supported | No-op preconditioner. |
| `jacobi` | [`PcType::Jacobi`](https://docs.rs/kryst/latest/kryst/context/pc_context/enum.PcType.html#variant.Jacobi) | Supported | Dense/CSR backends. |
| `block_jacobi` | [`PcType::BlockJacobi`](https://docs.rs/kryst/latest/kryst/context/pc_context/enum.PcType.html#variant.BlockJacobi) | Supported | Block size via `PcOptions::jacobi_block_size`. |
| `ilu0` | [`PcType::Ilu0`](https://docs.rs/kryst/latest/kryst/context/pc_context/enum.PcType.html#variant.Ilu0) | Supported | ILU(0). |
| `ilu` | [`PcType::Ilu`](https://docs.rs/kryst/latest/kryst/context/pc_context/enum.PcType.html#variant.Ilu) | Supported | ILU(k). |
| `ilut` | [`PcType::Ilut`](https://docs.rs/kryst/latest/kryst/context/pc_context/enum.PcType.html#variant.Ilut) | Supported | Drop tol + max fill. |
| `ilutp` | [`PcType::Ilutp`](https://docs.rs/kryst/latest/kryst/context/pc_context/enum.PcType.html#variant.Ilutp) | Supported | Drop tol + max fill + pivoting. |
| `ilup` | [`PcType::Ilup`](https://docs.rs/kryst/latest/kryst/context/pc_context/enum.PcType.html#variant.Ilup) | Supported | Polynomial ILU. |
| `sor` | [`PcType::Sor`](https://docs.rs/kryst/latest/kryst/context/pc_context/enum.PcType.html#variant.Sor) | Supported | `PcOptions::sor_*` knobs. |
| `asm` / `ras` | [`PcType::Asm`](https://docs.rs/kryst/latest/kryst/context/pc_context/enum.PcType.html#variant.Asm) | Supported | `ras` maps to ASM with RAS mode. |
| `chebyshev` | [`PcType::Chebyshev`](https://docs.rs/kryst/latest/kryst/context/pc_context/enum.PcType.html#variant.Chebyshev) | Supported | Chebyshev smoothing PC. |
| `amg` | [`PcType::Amg`](https://docs.rs/kryst/latest/kryst/context/pc_context/enum.PcType.html#variant.Amg) | Supported | Algebraic multigrid with `PcOptions::amg_*`. |
| `approxinv` | [`PcType::ApproxInverse`](https://docs.rs/kryst/latest/kryst/context/pc_context/enum.PcType.html#variant.ApproxInverse) | Supported | CSR approximate inverse (FSAI/SPAI). |
| `fieldsplit` | [`PcType::FieldSplit`](https://docs.rs/kryst/latest/kryst/context/pc_context/enum.PcType.html#variant.FieldSplit) | Partial | Block-diagonal split only; Tracking: [FieldSplit advanced](#tracking-fieldsplit-advanced). |
| `shell` | [`PcType::Shell`](https://docs.rs/kryst/latest/kryst/context/pc_context/enum.PcType.html#variant.Shell) + [`register_shell_callback`](https://docs.rs/kryst/latest/kryst/preconditioner/shell/fn.register_shell_callback.html) | Partial | Supports apply/setup/destroy hooks with context bindings; Tracking: [Shell PC parity](#tracking-shell-pc). |
| `ksp` | [`PcType::Ksp`](https://docs.rs/kryst/latest/kryst/context/pc_context/enum.PcType.html#variant.Ksp) | Partial | Uses simple inner-PC loop; Tracking: [KSP-as-PC parity](#tracking-ksp-as-pc). |
| `mg` | [`PcType::Mg`](https://docs.rs/kryst/latest/kryst/context/pc_context/enum.PcType.html#variant.Mg) | Partial | Injection hierarchy with Galerkin coarse operators and V/W/F cycles; Tracking: [Multigrid parity](#tracking-mg-parity). |
| `bddc` | [`PcType::Bddc`](https://docs.rs/kryst/latest/kryst/context/pc_context/enum.PcType.html#variant.Bddc) | Partial | Prototype coarse space/constraints + interface metadata; Tracking: [BDDC support](#tracking-bddc). |
| `gamg` | [`PcType::Gamg`](https://docs.rs/kryst/latest/kryst/context/pc_context/enum.PcType.html#variant.Gamg) | Partial | Backed by AMG with PETSc GAMG defaults; supports core GAMG options. |
| `lu` | [`PcType::Lu`](https://docs.rs/kryst/latest/kryst/context/pc_context/enum.PcType.html#variant.Lu) | Supported | Direct solve (PREONLY recommended). |
| `qr` | [`PcType::Qr`](https://docs.rs/kryst/latest/kryst/context/pc_context/enum.PcType.html#variant.Qr) | Supported | Direct solve (PREONLY recommended). |
| `superludist` | [`PcType::SuperLuDist`](https://docs.rs/kryst/latest/kryst/context/pc_context/enum.PcType.html#variant.SuperLuDist) | Partial | Requires `superlu_dist` feature. |

## Key options matrix

### KSP options

| PETSc option | kryst option | Status | Notes |
| --- | --- | --- | --- |
| `-ksp_type` | [`KspOptions::ksp_type`](https://docs.rs/kryst/latest/kryst/config/options/struct.KspOptions.html#structfield.ksp_type) | Supported | Mirrors `SolverType`. |
| `-ksp_rtol` | [`KspOptions::rtol`](https://docs.rs/kryst/latest/kryst/config/options/struct.KspOptions.html#structfield.rtol) | Supported | Relative tolerance. |
| `-ksp_atol` | [`KspOptions::atol`](https://docs.rs/kryst/latest/kryst/config/options/struct.KspOptions.html#structfield.atol) | Supported | Absolute tolerance. |
| `-ksp_dtol` | [`KspOptions::dtol`](https://docs.rs/kryst/latest/kryst/config/options/struct.KspOptions.html#structfield.dtol) | Supported | Divergence tolerance. |
| `-ksp_max_it` | [`KspOptions::maxits`](https://docs.rs/kryst/latest/kryst/config/options/struct.KspOptions.html#structfield.maxits) | Supported | Max iterations. |
| `-ksp_restart` | [`KspOptions::restart`](https://docs.rs/kryst/latest/kryst/config/options/struct.KspOptions.html#structfield.restart) | Supported | GMRES/GCR restart length. |
| `-ksp_gmres_*` | [`KspOptions::gmres_*`](https://docs.rs/kryst/latest/kryst/config/options/struct.KspOptions.html) | Supported | Orthog/reorthog/variant/s-step. |
| `-ksp_fgmres_*` | [`KspOptions::fgmres_*`](https://docs.rs/kryst/latest/kryst/config/options/struct.KspOptions.html) | Supported | Orthog/reorthog/variant. |
| `-ksp_gcr_restart` | [`KspOptions::gcr_restart`](https://docs.rs/kryst/latest/kryst/config/options/struct.KspOptions.html#structfield.gcr_restart) | Supported | GCR/PipeGCR restart length. |
| `-ksp_richardson_omega` | [`KspOptions::richardson_omega`](https://docs.rs/kryst/latest/kryst/config/options/struct.KspOptions.html#structfield.richardson_omega) | Supported | Richardson step size. |
| `-ksp_chebyshev_omega` | [`KspOptions::chebyshev_omega`](https://docs.rs/kryst/latest/kryst/config/options/struct.KspOptions.html#structfield.chebyshev_omega) | Partial | Used for Chebyshev-as-KSP. |
| `-ksp_pc_side` | [`KspOptions::pc_side`](https://docs.rs/kryst/latest/kryst/config/options/struct.KspOptions.html#structfield.pc_side) | Supported | `left`/`right`/`symmetric`. |
| `-ksp_monitor_rank0` | [`KspOptions::ksp_monitor_rank0`](https://docs.rs/kryst/latest/kryst/config/options/struct.KspOptions.html#structfield.ksp_monitor_rank0) | Supported | Rank-0 monitor policy. |
| `-ksp_reduction` | [`KspOptions::reduction`](https://docs.rs/kryst/latest/kryst/config/options/struct.KspOptions.html#structfield.reduction) | Supported | Reduction mode (`fast`/`deterministic`). |

### PC options

| PETSc option | kryst option | Status | Notes |
| --- | --- | --- | --- |
| `-pc_type` | [`PcOptions::pc_type`](https://docs.rs/kryst/latest/kryst/config/options/struct.PcOptions.html#structfield.pc_type) | Supported | Mirrors `PcType`. |
| `-pc_chain` | [`PcOptions::pc_chain`](https://docs.rs/kryst/latest/kryst/config/options/struct.PcOptions.html#structfield.pc_chain) | Supported | Composite chain (`jacobi->ilu->amg`). |
| `-pc_composite_type` | [`PcOptions::pc_composite_type`](https://docs.rs/kryst/latest/kryst/config/options/struct.PcOptions.html#structfield.pc_composite_type) | Supported | `multiplicative` or `additive`. |
| `-pc_composite_prefixes` | [`PcOptions::pc_composite_prefixes`](https://docs.rs/kryst/latest/kryst/config/options/struct.PcOptions.html#structfield.pc_composite_prefixes) | Supported | Per-stage options scoping. |
| `-pc_fieldsplit_block_sizes` | [`PcOptions::pc_fieldsplit_block_sizes`](https://docs.rs/kryst/latest/kryst/config/options/struct.PcOptions.html#structfield.pc_fieldsplit_block_sizes) | Supported | Splits local or distributed operators into per-field blocks. |
| `-pc_fieldsplit_type` | [`PcOptions::pc_fieldsplit_type`](https://docs.rs/kryst/latest/kryst/config/options/struct.PcOptions.html#structfield.pc_fieldsplit_type) | Supported | Supports `additive`, `multiplicative`, `symmetric`, `schur`, and composite aliases (`composite_*`, `basic`, `gs`). |
| `-pc_fieldsplit_prefixes` | [`PcOptions::pc_fieldsplit_prefixes`](https://docs.rs/kryst/latest/kryst/config/options/struct.PcOptions.html#structfield.pc_fieldsplit_prefixes) | Supported | Per-field scoping for nested sub-KSP/PC options. |
| `-pc_ksp_type` | [`PcOptions::pc_ksp_ksp_type`](https://docs.rs/kryst/latest/kryst/config/options/struct.PcOptions.html#structfield.pc_ksp_ksp_type) | Partial | Used by `PcType::Ksp`. |
| `-pc_ksp_pc_type` | [`PcOptions::pc_ksp_pc_type`](https://docs.rs/kryst/latest/kryst/config/options/struct.PcOptions.html#structfield.pc_ksp_pc_type) | Partial | Used by `PcType::Ksp`. |
| `-pc_ksp_maxits` / `-pc_ksp_rtol` / `-pc_ksp_atol` / `-pc_ksp_dtol` | `PcOptions::pc_ksp_*tol/maxits` | Supported | Inner KSP tolerance/iteration controls for `PcType::Ksp`; scoped `pc_ksp_ksp_options` still takes precedence field-by-field. |
| `-pc_ksp_gmres_restart` / `-pc_ksp_fgmres_restart` / `-pc_ksp_restart` | `PcOptions::pc_ksp_*restart` | Supported | Restart precedence is solver-specific then generic: GMRES/FGMRES-specific key first, then `pc_ksp_restart`. |
| `-pc_ksp_monitor_policy` / `-pc_ksp_monitor_rank0` | `PcOptions::pc_ksp_monitor_policy` / `pc_ksp_monitor_rank0` | Supported | Explicit policy key (`all`/`rank0`) overrides boolean compatibility flag. |
| `-pc_ksp_allow_maxits` | [`PcOptions::pc_ksp_allow_maxits`](https://docs.rs/kryst/latest/kryst/config/options/struct.PcOptions.html#structfield.pc_ksp_allow_maxits) | Supported | Controls whether inner `KSP_DIVERGED_ITS` is treated as acceptable (`true`, default) or nested failure (`false`). |
| `-pc_ksp_propagate_converged_reason` | [`PcOptions::pc_ksp_propagate_converged_reason`](https://docs.rs/kryst/latest/kryst/config/options/struct.PcOptions.html#structfield.pc_ksp_propagate_converged_reason) | Supported | If true (default), `SolveStats::nested_pc_failure.reason` preserves the inner reason; if false, normalizes to `DivergedPcFailed`. |
| `-pc_mg_levels` | [`PcOptions::pc_mg_levels`](https://docs.rs/kryst/latest/kryst/config/options/struct.PcOptions.html#structfield.pc_mg_levels) | Partial | Controls number of MG levels (injection coarsening). |
| `-pc_mg_cycle_type` | [`PcOptions::pc_mg_cycle_type`](https://docs.rs/kryst/latest/kryst/config/options/struct.PcOptions.html#structfield.pc_mg_cycle_type) | Partial | `v`/`w`/`f` cycles supported. |
| `-pc_mg_smoother` | [`PcOptions::pc_mg_smoother`](https://docs.rs/kryst/latest/kryst/config/options/struct.PcOptions.html#structfield.pc_mg_smoother) | Partial | Smoother applied per-level; direct smoothers (`lu`/`qr`) become the coarse solve. |
| `-pc_mg_smoother_steps` | [`PcOptions::pc_mg_smoother_steps`](https://docs.rs/kryst/latest/kryst/config/options/struct.PcOptions.html#structfield.pc_mg_smoother_steps) | Partial | Number of pre/post smoothing sweeps per level. |
| `-pc_shell_name` | [`PcOptions::pc_shell_name`](https://docs.rs/kryst/latest/kryst/config/options/struct.PcOptions.html#structfield.pc_shell_name) | Partial | Names callback for `PcType::Shell`. |
| `-pc_shell_apply` | [`PcOptions::pc_shell_apply`](https://docs.rs/kryst/latest/kryst/config/options/struct.PcOptions.html#structfield.pc_shell_apply) | Partial | Names the primary shell apply hook. |
| `-pc_shell_apply_transpose` | [`PcOptions::pc_shell_apply_transpose`](https://docs.rs/kryst/latest/kryst/config/options/struct.PcOptions.html#structfield.pc_shell_apply_transpose) | Partial | Optional transpose apply hook. |
| `-pc_shell_apply_conjugate_transpose` | [`PcOptions::pc_shell_apply_conjugate_transpose`](https://docs.rs/kryst/latest/kryst/config/options/struct.PcOptions.html#structfield.pc_shell_apply_conjugate_transpose) | Partial | Optional conjugate-transpose apply hook. |
| `-pc_shell_apply_symmetric` | [`PcOptions::pc_shell_apply_symmetric`](https://docs.rs/kryst/latest/kryst/config/options/struct.PcOptions.html#structfield.pc_shell_apply_symmetric) | Partial | Optional combined symmetric apply hook. |
| `-pc_shell_apply_symmetric_left` | [`PcOptions::pc_shell_apply_symmetric_left`](https://docs.rs/kryst/latest/kryst/config/options/struct.PcOptions.html#structfield.pc_shell_apply_symmetric_left) | Partial | Optional left factor in split symmetric apply. |
| `-pc_shell_apply_symmetric_right` | [`PcOptions::pc_shell_apply_symmetric_right`](https://docs.rs/kryst/latest/kryst/config/options/struct.PcOptions.html#structfield.pc_shell_apply_symmetric_right) | Partial | Optional right factor in split symmetric apply. |
| `-pc_shell_setup` | [`PcOptions::pc_shell_setup`](https://docs.rs/kryst/latest/kryst/config/options/struct.PcOptions.html#structfield.pc_shell_setup) | Partial | Names shell setup hook. |
| `-pc_shell_destroy` | [`PcOptions::pc_shell_destroy`](https://docs.rs/kryst/latest/kryst/config/options/struct.PcOptions.html#structfield.pc_shell_destroy) | Partial | Names shell destroy hook. |
| `-pc_shell_context` | [`PcOptions::pc_shell_context`](https://docs.rs/kryst/latest/kryst/config/options/struct.PcOptions.html#structfield.pc_shell_context) | Partial | Names shell context binding. |
| `-pc_bddc_coarse_ksp_type` | [`PcOptions::pc_bddc_coarse_ksp_type`](https://docs.rs/kryst/latest/kryst/config/options/struct.PcOptions.html#structfield.pc_bddc_coarse_ksp_type) | Partial | Wired to BDDC config/diagnostics; coarse solver placeholder. |
| `-pc_bddc_coarse_pc_type` | [`PcOptions::pc_bddc_coarse_pc_type`](https://docs.rs/kryst/latest/kryst/config/options/struct.PcOptions.html#structfield.pc_bddc_coarse_pc_type) | Partial | Wired to BDDC config/diagnostics; coarse solver placeholder. |
| `-pc_bddc_use_vertices` | [`PcOptions::pc_bddc_use_vertices`](https://docs.rs/kryst/latest/kryst/config/options/struct.PcOptions.html#structfield.pc_bddc_use_vertices) | Partial | Enables vertex constraint metadata. |
| `-pc_gamg_type` | [`PcOptions::pc_gamg_type`](https://docs.rs/kryst/latest/kryst/config/options/struct.PcOptions.html#structfield.pc_gamg_type) | Partial | Supported values: `agg`, `classical`. |
| `-pc_gamg_threshold` | [`PcOptions::pc_gamg_threshold`](https://docs.rs/kryst/latest/kryst/config/options/struct.PcOptions.html#structfield.pc_gamg_threshold) | Supported | Maps to GAMG strength threshold. |
| `-pc_gamg_levels` | [`PcOptions::pc_gamg_levels`](https://docs.rs/kryst/latest/kryst/config/options/struct.PcOptions.html#structfield.pc_gamg_levels) | Supported | Maps to GAMG hierarchy depth. |
| `-pc_gamg_coarsen_type` | [`PcOptions::pc_gamg_coarsen_type`](https://docs.rs/kryst/latest/kryst/config/options/struct.PcOptions.html#structfield.pc_gamg_coarsen_type) | Supported | Coarsening override (`rs`, `hmis`, `pmis`, `falgout`). |
| `-pc_gamg_interp_type` | [`PcOptions::pc_gamg_interp_type`](https://docs.rs/kryst/latest/kryst/config/options/struct.PcOptions.html#structfield.pc_gamg_interp_type) | Supported | Interpolation override (`classical`, `direct`, `multipass`, `extended`, `standard`, `he`). |
| `-pc_gamg_aggressive_levels` | [`PcOptions::pc_gamg_aggressive_levels`](https://docs.rs/kryst/latest/kryst/config/options/struct.PcOptions.html#structfield.pc_gamg_aggressive_levels) | Supported | Aggressive coarsening level count (>=1). |
| `-pc_gamg_aggressive_mis_k` | [`PcOptions::pc_gamg_aggressive_mis_k`](https://docs.rs/kryst/latest/kryst/config/options/struct.PcOptions.html#structfield.pc_gamg_aggressive_mis_k) | Supported | Aggressive PMIS/HMIS neighborhood depth (>=2). |

## Monitor hooks

| PETSc hook | kryst API | Status | Notes/Tracking |
| --- | --- | --- | --- |
| `KSPMonitorSet` | [`KspContext::add_monitor`](https://docs.rs/kryst/latest/kryst/context/struct.KspContext.html#method.add_monitor) | Supported | Callback signature is [`MonitorCallback`](https://docs.rs/kryst/latest/kryst/solver/type.MonitorCallback.html). |
| `KSPMonitorSet` (rank-0 only) | [`KspContext::add_monitor_rank0`](https://docs.rs/kryst/latest/kryst/context/struct.KspContext.html#method.add_monitor_rank0) | Supported | Mirrors `-ksp_monitor_rank0`. |
| `KSPMonitorCancel` | [`KspContext::clear_monitors`](https://docs.rs/kryst/latest/kryst/context/struct.KspContext.html#method.clear_monitors) | Supported | Removes all registered monitors. |
| `KSPMonitorSet` stop | [`MonitorAction::Stop`](https://docs.rs/kryst/latest/kryst/solver/enum.MonitorAction.html#variant.Stop) | Supported | Return `Stop` to terminate. |
| `KSPMonitorSet` policy | [`MonitorPolicy`](https://docs.rs/kryst/latest/kryst/context/ksp_context/enum.MonitorPolicy.html) | Supported | `Rank0Only` or `AllRanks`. |

## Convergence reasons

| PETSc reason | kryst reason | Status | Notes/Tracking |
| --- | --- | --- | --- |
| `KSP_CONVERGED_RTOL` | [`ConvergedReason::ConvergedRtol`](https://docs.rs/kryst/latest/kryst/utils/convergence/enum.ConvergedReason.html#variant.ConvergedRtol) | Supported | Relative tolerance met. |
| `KSP_CONVERGED_ATOL` | [`ConvergedReason::ConvergedAtol`](https://docs.rs/kryst/latest/kryst/utils/convergence/enum.ConvergedReason.html#variant.ConvergedAtol) | Supported | Absolute tolerance met. |
| `KSP_CONVERGED_TRUST_REGION` | [`ConvergedReason::ConvergedTrustRegion`](https://docs.rs/kryst/latest/kryst/utils/convergence/enum.ConvergedReason.html#variant.ConvergedTrustRegion) | Partial | Trust-region style stopping. |
| `KSP_CONVERGED_HAPPY_BREAKDOWN` | [`ConvergedReason::ConvergedHappyBreakdown`](https://docs.rs/kryst/latest/kryst/utils/convergence/enum.ConvergedReason.html#variant.ConvergedHappyBreakdown) | Supported | Happy breakdown in GMRES/TFQMR variants. |
| `KSP_DIVERGED_DTOL` | [`ConvergedReason::DivergedDtol`](https://docs.rs/kryst/latest/kryst/utils/convergence/enum.ConvergedReason.html#variant.DivergedDtol) | Supported | Divergence tolerance exceeded. |
| `KSP_DIVERGED_ITS` | [`ConvergedReason::DivergedMaxIts`](https://docs.rs/kryst/latest/kryst/utils/convergence/enum.ConvergedReason.html#variant.DivergedMaxIts) | Supported | Max iterations. |
| `KSP_DIVERGED_BREAKDOWN` | [`ConvergedReason::DivergedBreakdown`](https://docs.rs/kryst/latest/kryst/utils/convergence/enum.ConvergedReason.html#variant.DivergedBreakdown) | Partial | Generic breakdown/indefinite failures are currently emitted by IDRS through `KError::BreakdownOrIndefinite` remapping. |
| `KSP_DIVERGED_BREAKDOWN_BICG` | [`ConvergedReason::DivergedBreakdownBiCG`](https://docs.rs/kryst/latest/kryst/utils/convergence/enum.ConvergedReason.html#variant.DivergedBreakdownBiCG) | Supported | BiCG-family breakdown reason used by BiCGStab, CGS, QMR, and TFQMR kernels. |
| `KSP_DIVERGED_INDEFINITE_MAT` | [`ConvergedReason::DivergedIndefiniteMatrix`](https://docs.rs/kryst/latest/kryst/utils/convergence/enum.ConvergedReason.html#variant.DivergedIndefiniteMatrix) | Supported | Matrix indefiniteness checks are active in CG/PCG paths. |
| `KSP_DIVERGED_INDEFINITE_PC` | [`ConvergedReason::DivergedIndefinitePC`](https://docs.rs/kryst/latest/kryst/utils/convergence/enum.ConvergedReason.html#variant.DivergedIndefinitePC) | Supported | Emitted when solver math or setup detects an indefinite preconditioner. |
| `KSP_DIVERGED_PCSETUP_FAILED` | [`ConvergedReason::DivergedPcSetupFailed`](https://docs.rs/kryst/latest/kryst/utils/convergence/enum.ConvergedReason.html#variant.DivergedPcSetupFailed) | Supported | `KspContext::setup` and setup-on-solve remap factorization/PC setup failures into a converged-reason result. |
| `KSP_DIVERGED_PC_FAILED` | [`ConvergedReason::DivergedPcFailed`](https://docs.rs/kryst/latest/kryst/utils/convergence/enum.ConvergedReason.html#variant.DivergedPcFailed) | Supported | Preconditioner apply failures (including shell apply/setup hooks) propagate as solve divergence. |
| `KSP_DIVERGED_NANORINF` | [`ConvergedReason::DivergedNan`](https://docs.rs/kryst/latest/kryst/utils/convergence/enum.ConvergedReason.html#variant.DivergedNan) / [`ConvergedReason::DivergedInf`](https://docs.rs/kryst/latest/kryst/utils/convergence/enum.ConvergedReason.html#variant.DivergedInf) | Supported | Non-finite residual checks map both NaN and Inf to PETSc's shared NaN-or-Inf reason code. |
| `KSP_CONVERGED_ITERATING` | [`ConvergedReason::Continued`](https://docs.rs/kryst/latest/kryst/utils/convergence/enum.ConvergedReason.html#variant.Continued) | Supported | Still iterating. |
| `KSP_DIVERGED_USER` | [`ConvergedReason::StoppedByMonitor`](https://docs.rs/kryst/latest/kryst/utils/convergence/enum.ConvergedReason.html#variant.StoppedByMonitor) | Supported | Monitor requested stop (PETSc "user requested" divergence code). |

### Divergence reason emitters by method

- `KSP_DIVERGED_BREAKDOWN`: currently emitted by `IDRS` through `KError::BreakdownOrIndefinite` remapping in `KspContext`.
- `KSP_DIVERGED_BREAKDOWN_BICG`: emitted by BiCG-family kernels (`BiCGStab`, `CGS`, `QMR`, `TFQMR`).
- `KSP_DIVERGED_NANORINF`: emitted by any solver calling `Convergence::check` (`CG`, `PCG`, `GMRES`, `FGMRES`, `MINRES`, `LSQR`, `PCA-GMRES`, `QMR`) and by explicit non-finite guards in `BiCGStab`/`TFQMR`.
- `KSP_DIVERGED_INDEFINITE_MAT` / `KSP_DIVERGED_INDEFINITE_PC`: emitted by `CG`/`PCG` indefiniteness checks and preserved through nested KSP-PC propagation.
- `KSP_DIVERGED_PCSETUP_FAILED`: emitted during `KspContext::setup` (or implicit setup within `solve`) when PC setup/factorization fails.
- `KSP_DIVERGED_PC_FAILED`: emitted when preconditioner apply paths fail (including shell PC hooks and nested KSP-as-PC inner failures).
- For nested preconditioners, `SolveStats::nested_pc_failure` captures the inner reason/iterations/component metadata so monitor output can explain the outer failure.
- Outer `SolveStats.reason` remains `DivergedPcFailed` for nested apply failures, while `SolveStats::nested_pc_failure.reason` carries the inner reason (or normalized `DivergedPcFailed` when `-pc_ksp_propagate_converged_reason false`).


## MG/GAMG hierarchy and coarse-policy mapping

| PETSc option/workflow | kryst option/API | Status | Notes |
| --- | --- | --- | --- |
| `-pc_mg_coarse_pc_type` / `-pc_mg_coarse_ksp_type` | `PcOptions::pc_mg_coarse_*` | Supported | Global coarse solver selection for built-in MG. |
| Per-level MG coarse strategy overrides | `MgPc::set_level_coarse_solver_type(level, "...")` | Partial | Deepest matching level override is used for the active coarse solve path. |
| User-supplied MG transfers (`P`,`R`) | `MgPc::set_level_transfer_operators` / `set_level_transfer_from_linops` | Supported | Overrides generated transfers level-by-level. |
| GAMG distributed coarse policy | `-pc_amg_dist_apply_mode {root_gather,local_prototype,...}` | Supported | `GamgConfig::try_from_opts` forwards these controls into AMG. |
| GAMG hierarchy transfer/coarse overrides | `Gamg::set_level_transfer_operators`, `Gamg::set_level_coarse_solver` | Partial | Uses AMG hierarchy override hooks; useful for prototypes and tuning. |
| AMG diagnostics (complexity + level nnz) | `AmgStats::{grid_complexity, operator_complexity, levels[]}` | Supported | Includes per-level `nnz_a/nnz_p/nnz_r`, effective nnz, smoothing-work estimates, and selected distributed coarse-route diagnostics. |

### Per-level override precedence

MG/GAMG option resolution follows an explicit precedence chain:

1. Global defaults (for example `-pc_mg_smoother`, `-pc_mg_coarse_*`, `-pc_amg_sweeps_*`).
2. Family-level policy entries (`-pc_mg_levels_policy` / `-pc_gamg_levels_policy`).
3. Exact scoped level overrides (`-pc_mg_levels_<i>_*` / `-pc_gamg_levels_<i>_*`).

When multiple policy entries target the same level, later entries are merged deterministically and exact scoped level options win field-by-field.



### New MG/GAMG per-level precedence keys

`kryst` supports precedence-resolved policy layering for per-level smoother/KSP/PC stacks:

1. Global defaults (`-pc_mg_smoother`, `-pc_mg_coarse_*`, `-pc_ksp_*`).
2. Family-level policy keys via `level_key` in `-pc_mg_level_policies` / `-pc_gamg_level_policies`:
   - `level_key=all|fine|intermediate|coarse`.
3. Exact level entries from scoped options (`-pc_mg_levels_<i>_*`, `-pc_gamg_levels_<i>_*`) and explicit `level=<i>` policies.

Exact level entries always override family/global values field-by-field.

### Distributed coarse diagnostics surfaced in stats

AMG hierarchy stats report:

- `grid_complexity`, `operator_complexity`
- total hierarchy `nnz`
- per-level `nnz(A/P/R)` and smoothing work estimate
- selected distributed coarse route and ordered fallback list

MG diagnostics expose the chosen coarse route and resolved fallback chain at the coarse level.

## Parity estimate and prioritized roadmap

### Current estimate

- **Surface parity (KSP/PC type and option presence): ~84%**
  - Most commonly used PETSc Krylov and PC categories are present, but several entries are still marked partial because they route through simplified kernels or compatibility adapters.
- **Behavioral parity (runtime semantics + diagnostics): ~72%**
  - Core lifecycle, converged-reason mapping, and nested diagnostics are strong, while advanced nested composition and some method-specific divergence semantics remain uneven.
- **Performance-oriented parity (native distributed + complex-quality paths): ~61%**
  - This is the gating score for production parity: the biggest gaps are native distributed preconditioners, scalable MG/GAMG policy depth, and mathematically strong complex-scalar preconditioning paths.

> **Overall estimate for full PETSc KSP/PC parity in practical HPC use:** **~69% complete**.

### Prioritized roadmap (efficiency/customization first)

1. **Close complex-preconditioner quality gaps in existing high-value PCs (highest ROI)**
   - Prioritize **ILU_CSR**, **ApproxInv**, and **SOR/Deflation-adjacent** complex paths so they avoid real-part projection whenever mathematically valid.
   - Add per-PC quality diagnostics (residual reduction vs. setup/apply time, stability flags) to keep implementations performance-driven instead of parity-driven.
   - Ensure every complex upgrade keeps matrix reuse (`symbolic`/`numeric`) and no-allocation apply paths intact.
2. **Promote native MPI distributed preconditioners over compatibility adapters**
   - Expand `DistCsrOp`-native implementations for Block-Jacobi/ILU family, coarse corrections, and selected AMG/GAMG components.
   - Keep adapter/global modes as migration fallbacks, but make native distributed the default recommendation for large runs.
   - Add scaling-focused acceptance criteria (strong/weak scaling + reduction count budgets).
3. **Strengthen MPI + rayon co-design in Krylov and PC internals**
   - Push overlap-friendly reductions and thread-parallel local kernels into more solver/PC paths, not just flagship variants.
   - Add tuning hooks that expose reproducibility/performance tradeoffs (e.g., overlap strategies, thread policy, reduction engine choices) at KSP/PC boundaries.
4. **Deepen MG/GAMG policy control where it impacts scalability**
   - Extend per-level solver/smoother/coarse controls and distributed coarse routing, but prioritize options with measurable complexity and time-to-solution impact.
   - Improve hierarchy diagnostics so users can tune operator/grid complexity with predictable outcomes.
5. **Complete nested composition controls (KSP-as-PC + FieldSplit) for real coupled systems**
   - Fill high-impact lifecycle/monitor/tolerance routing gaps for nested KSP, including inner/outer convergence coordination.
   - Expand split strategy flexibility that affects throughput and robustness for block multiphysics problems.
6. **Finish shell/transposed-apply and reason consistency work as an observability multiplier**
   - Add missing shell hooks and align divergence reason emission across methods so custom operator/preconditioner stacks are easier to debug at scale.

### Scope guardrails

- Favor changes that improve **time-to-solution**, **customizability**, and **numerical robustness** over low-impact option-count parity.
- Treat **real + complex scalar correctness** and **MPI + rayon execution quality** as first-class acceptance criteria for any new solver/PC capability.
- Require evidence for major parity work: benchmark deltas, reduction counts, and convergence-quality diagnostics on representative sparse systems.


## Recommended nested + split templates

For PETSc-like nested workflows, these templates are the most robust defaults in kryst:

- **Outer GMRES/FGMRES + inner KSP-as-PC (Jacobi)**
  - `-pc_type ksp -pc_ksp_ksp_type gmres -pc_ksp_pc_type jacobi`
  - Prefer explicit inner side (`-pc_ksp_pc_side left|right`) when outer side differs.
  - Inner restart precedence follows PETSc-style solver-specific keys: `pc_ksp_gmres_restart` / `pc_ksp_fgmres_restart` then `pc_ksp_restart`.
  - Inner failure coordination knobs: `-pc_ksp_allow_maxits {true|false}`, `-pc_ksp_propagate_converged_reason {true|false}`, and `-pc_ksp_monitor_policy {all|rank0}`.
- **FieldSplit Schur for coupled systems**
  - Start with `-pc_fieldsplit_type schur -pc_fieldsplit_schur_fact_type full -pc_fieldsplit_schur_precondition full`.
  - For complex scalars, use `self`, `selfp`, `full`, `full_matfree`, or `user` Schur preconditioning paths (avoid `diag`).
- **Composite aliases**
  - PETSc-style `composite_additive`, `composite_multiplicative`, and `composite_symmetric_multiplicative` are accepted.
  - kryst additionally accepts `basic`, `gs`, and `symmetric_multiplicative` for migration convenience.

All templates preserve structured nested diagnostics through `SolveStats.nested_pc_failure` and PETSc-style converged reason mapping.

## Known parity gaps

High-impact PETSc APIs or workflows that are not yet equivalent in kryst:

1. **Multigrid hierarchy management** (remaining parity for full PETSc per-level KSP stacks and advanced adaptive policies). Tracking: [Multigrid parity](#tracking-mg-parity).
2. **KSP-as-PC parity** (nested KSP choices, full inner KSP lifecycle). Tracking: [KSP-as-PC parity](#tracking-ksp-as-pc).
3. **Shell PC parity** (remaining PETSc `PCSHELL` hooks like transpose/symmetric apply, richer context helpers). Tracking: [Shell PC parity](#tracking-shell-pc).
4. **Convergence-reason method coverage** (some reasons are implemented but remain method-specific). Tracking: [Convergence reason parity](#tracking-breakdown-reason).
5. **BDDC advanced features** (`-pc_type bddc`). Tracking: [BDDC support](#tracking-bddc).
6. **GAMG advanced options** (`-pc_type gamg`), especially full PETSc-level smoother and repartition controls. Tracking: [GAMG support](#tracking-gamg).

## Tracking issues

<a id="tracking-mg-parity"></a>
### Tracking issue: Multigrid parity
Scope: remaining advanced cycle orchestration and full PETSc-equivalent per-level KSP/PC policy controls.

<a id="tracking-ksp-as-pc"></a>
### Tracking issue: KSP-as-PC parity
Scope: full nested KSP configuration (inner tolerances, monitors, solver selection) vs. current fixed-loop implementation.

<a id="tracking-shell-pc"></a>
### Tracking issue: Shell PC parity
Scope: remaining `PCSHELL` hooks (transpose/symmetric apply), helper APIs for typed context binding, and broader option parity beyond setup/destroy/context.

<a id="tracking-bddc"></a>
### Tracking issue: BDDC support
Scope: coarse spaces, constraints, subdomain interface coupling, and full coarse solve integration.

<a id="tracking-gamg"></a>
### Tracking issue: GAMG support
Scope: remaining PETSc GAMG parity (e.g., repartitioning and full smoother stacks) beyond supported type/threshold/levels/coarsen/interpolation/aggressive/distributed-coarse controls.

<a id="tracking-breakdown-reason"></a>
### Tracking issue: Breakdown reason parity
Scope: explicit divergence reasons (breakdown, NaN/Inf, PC failure) matching PETSc enums.

<a id="tracking-pc-failure-reason"></a>
### Tracking issue: PC failure reasons
Scope: extend PC failure propagation beyond shell hooks (e.g., factorization/solver failures).

## Complex-scalar exclusions by method

When building with `--features complex`, support is currently method-specific:

- `KSP GMRES (s-step)`: available; no complex hard-fail path.
- `PC SOR`: available with CSR-only setup and real-part operator projection.
- `PC Deflation`: available with real coarse operators applied to complex vectors.
- `PC ILU (CSR)`: partial; CSR-only complex path is available with explicit degraded-mode diagnostics, using real-part setup projection and split real/imag solves.
- `PC ApproxInv`: partial; CSR-only complex path is available with explicit degraded-mode diagnostics, using diagonal real-part initialization and complex apply.
- `PC MG`: hierarchy/operator storage is scalar-aware (S-based levels/transfers). Complex parity still depends on smoother/coarse preconditioner implementations.
