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
| `pipegcr` / `gcr_pipe` | [`SolverType::PipeGcr`](https://docs.rs/kryst/latest/kryst/context/ksp_context/enum.SolverType.html#variant.PipeGcr) | Unsupported | Tracking: [PipeGCR support](#tracking-pipegcr). |
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
| `shell` | [`PcType::Shell`](https://docs.rs/kryst/latest/kryst/context/pc_context/enum.PcType.html#variant.Shell) + [`register_shell_callback`](https://docs.rs/kryst/latest/kryst/preconditioner/shell/fn.register_shell_callback.html) | Partial | Supports named callback only; Tracking: [Shell PC parity](#tracking-shell-pc). |
| `ksp` | [`PcType::Ksp`](https://docs.rs/kryst/latest/kryst/context/pc_context/enum.PcType.html#variant.Ksp) | Partial | Uses simple inner-PC loop; Tracking: [KSP-as-PC parity](#tracking-ksp-as-pc). |
| `mg` | [`PcType::Mg`](https://docs.rs/kryst/latest/kryst/context/pc_context/enum.PcType.html#variant.Mg) | Partial | Placeholder V-cycle; Tracking: [Multigrid parity](#tracking-mg-parity). |
| `bddc` | [`PcType::Bddc`](https://docs.rs/kryst/latest/kryst/context/pc_context/enum.PcType.html#variant.Bddc) | Unsupported | Tracking: [BDDC support](#tracking-bddc). |
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
| `-ksp_gcr_restart` | [`KspOptions::gcr_restart`](https://docs.rs/kryst/latest/kryst/config/options/struct.KspOptions.html#structfield.gcr_restart) | Supported | GCR restart length. |
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
| `-pc_fieldsplit_block_sizes` | [`PcOptions::pc_fieldsplit_block_sizes`](https://docs.rs/kryst/latest/kryst/config/options/struct.PcOptions.html#structfield.pc_fieldsplit_block_sizes) | Partial | Block-diagonal split only. |
| `-pc_fieldsplit_prefixes` | [`PcOptions::pc_fieldsplit_prefixes`](https://docs.rs/kryst/latest/kryst/config/options/struct.PcOptions.html#structfield.pc_fieldsplit_prefixes) | Partial | FieldSplit scope only. |
| `-pc_ksp_type` | [`PcOptions::pc_ksp_ksp_type`](https://docs.rs/kryst/latest/kryst/config/options/struct.PcOptions.html#structfield.pc_ksp_ksp_type) | Partial | Used by `PcType::Ksp`. |
| `-pc_ksp_pc_type` | [`PcOptions::pc_ksp_pc_type`](https://docs.rs/kryst/latest/kryst/config/options/struct.PcOptions.html#structfield.pc_ksp_pc_type) | Partial | Used by `PcType::Ksp`. |
| `-pc_mg_levels` | [`PcOptions::pc_mg_levels`](https://docs.rs/kryst/latest/kryst/config/options/struct.PcOptions.html#structfield.pc_mg_levels) | Partial | Placeholder MG. |
| `-pc_mg_cycle_type` | [`PcOptions::pc_mg_cycle_type`](https://docs.rs/kryst/latest/kryst/config/options/struct.PcOptions.html#structfield.pc_mg_cycle_type) | Partial | Placeholder MG. |
| `-pc_shell_name` | [`PcOptions::pc_shell_name`](https://docs.rs/kryst/latest/kryst/config/options/struct.PcOptions.html#structfield.pc_shell_name) | Partial | Names callback for `PcType::Shell`. |

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
| `KSP_CONVERGED_STEP_LENGTH` | [`ConvergedReason::ConvergedTrustRegion`](https://docs.rs/kryst/latest/kryst/utils/convergence/enum.ConvergedReason.html#variant.ConvergedTrustRegion) | Partial | Trust-region style stopping. |
| `KSP_CONVERGED_HAPPY_BREAKDOWN` | [`ConvergedReason::ConvergedHappyBreakdown`](https://docs.rs/kryst/latest/kryst/utils/convergence/enum.ConvergedReason.html#variant.ConvergedHappyBreakdown) | Supported | Happy breakdown in GMRES/TFQMR variants. |
| `KSP_DIVERGED_DTOL` | [`ConvergedReason::DivergedDtol`](https://docs.rs/kryst/latest/kryst/utils/convergence/enum.ConvergedReason.html#variant.DivergedDtol) | Supported | Divergence tolerance exceeded. |
| `KSP_DIVERGED_ITS` | [`ConvergedReason::DivergedMaxIts`](https://docs.rs/kryst/latest/kryst/utils/convergence/enum.ConvergedReason.html#variant.DivergedMaxIts) | Supported | Max iterations. |
| `KSP_DIVERGED_BREAKDOWN` | — | Unsupported | Tracking: [Breakdown reason parity](#tracking-breakdown-reason). |
| `KSP_DIVERGED_BREAKDOWN_BICG` | — | Unsupported | Tracking: [Breakdown reason parity](#tracking-breakdown-reason). |
| `KSP_DIVERGED_PC_FAILED` | — | Unsupported | Tracking: [PC failure reasons](#tracking-pc-failure-reason). |
| `KSP_DIVERGED_NANORINF` | — | Unsupported | Tracking: [NaN/Inf reasons](#tracking-nan-inf-reason). |
| `KSP_CONVERGED_ITERATING` | [`ConvergedReason::Continued`](https://docs.rs/kryst/latest/kryst/utils/convergence/enum.ConvergedReason.html#variant.Continued) | Supported | Still iterating. |
| `KSP_DIVERGED_MONITOR` | [`ConvergedReason::StoppedByMonitor`](https://docs.rs/kryst/latest/kryst/utils/convergence/enum.ConvergedReason.html#variant.StoppedByMonitor) | Supported | Monitor requested stop. |

## Known parity gaps

High-impact PETSc APIs or workflows that are not yet equivalent in kryst:

1. **FieldSplit advanced modes** (Schur, multiplicative, symmetric, and custom split options). Tracking: [FieldSplit advanced](#tracking-fieldsplit-advanced).
2. **Multigrid hierarchy management** (custom smoothers/coarse solvers, interpolation/restriction). Tracking: [Multigrid parity](#tracking-mg-parity).
3. **KSP-as-PC parity** (nested KSP choices, full inner KSP lifecycle). Tracking: [KSP-as-PC parity](#tracking-ksp-as-pc).
4. **Shell PC parity** (setup hooks, stateful callbacks, richer PETSc `PCSHELL` options). Tracking: [Shell PC parity](#tracking-shell-pc).
5. **Explicit breakdown/divergence reasons** (NaN/Inf, PC failure, BiCG breakdown distinctions). Tracking: [Convergence reason parity](#tracking-breakdown-reason).
6. **PipeGCR solver** (`-ksp_type pipegcr`). Tracking: [PipeGCR support](#tracking-pipegcr).
7. **BDDC preconditioner** (`-pc_type bddc`). Tracking: [BDDC support](#tracking-bddc).

## Tracking issues

<a id="tracking-fieldsplit-advanced"></a>
### Tracking issue: FieldSplit advanced
Scope: Schur complement splits, additive/multiplicative modes, per-field matrices, and PETSc-style split types.

<a id="tracking-mg-parity"></a>
### Tracking issue: Multigrid parity
Scope: coarse-grid operators, smoothers, prolongation/restriction, and full V/W/F-cycle semantics.

<a id="tracking-ksp-as-pc"></a>
### Tracking issue: KSP-as-PC parity
Scope: full nested KSP configuration (inner tolerances, monitors, solver selection) vs. current fixed-loop implementation.

<a id="tracking-shell-pc"></a>
### Tracking issue: Shell PC parity
Scope: richer `PCSHELL` hooks (setup/destroy), mutable context, and error propagation matching PETSc.

<a id="tracking-pipegcr"></a>
### Tracking issue: PipeGCR support
Scope: pipelined GCR implementation and option parity (`-ksp_gcr_restart`, monitoring).

<a id="tracking-bddc"></a>
### Tracking issue: BDDC support
Scope: coarse spaces, constraints, and subdomain interface handling.

<a id="tracking-breakdown-reason"></a>
### Tracking issue: Breakdown reason parity
Scope: explicit divergence reasons (breakdown, NaN/Inf, PC failure) matching PETSc enums.

<a id="tracking-pc-failure-reason"></a>
### Tracking issue: PC failure reasons
Scope: propagate PC setup/apply failures into convergence reasons.

<a id="tracking-nan-inf-reason"></a>
### Tracking issue: NaN/Inf reasons
Scope: detect NaN/Inf residuals and map to PETSc divergence reasons.
