# Communicator rules

kryst follows PETSc-style communicator ownership: the operator decides the communicator and the
solver/preconditioner follows it. The rules below describe how `KspContext` binds to communicators
and what is considered valid.

## Binding and congruence

- `KspContext` is bound to a communicator once operators are set.
- `try_set_operators(Amat, Pmat)` requires `Amat.comm()` and `Pmat.comm()` to be congruent; a mismatch is an error.
- Dimensions must match: `Amat.dims() == Pmat.dims()`.
- Communicators are considered compatible only if they are IDENT or CONGRUENT
  (MPI_Comm_compare semantics).

## Explicit communicator wrapping

`try_set_operators_with_comm(Amat, Pmat, comm)` wraps the operators with an explicit communicator
and then applies the same congruence checks. Overriding is only allowed when the base operator
communicator is trivial (`UniverseComm::NoComm` or size==1) or already congruent with `comm`.
This is intended for cases where the operator is replicated but the solver should run on a
specific MPI communicator.

## Recommended usage patterns

- Create the `KspContext` on the communicator you intend to solve on.
- Build operators on the same communicator and call `try_set_operators`.
- For subcommunicator solves, build the subcomm first, then create the KSP and operators on it.
- Avoid mixing `WORLD` operators with subcomm KSPs.

## MPI-global preconditioning (`-pc_global`)

MPI-global preconditioners are selected via `PcOptions::mpi_pc_options()` and only activate when
`Amat`/`Pmat` is a `DistCsrOp` on a communicator with size > 1. When those conditions are met,
`pc_global` overrides the local `pc_type` path and builds a distributed preconditioner:

- `pc_global=block_jacobi` → `preconditioner::dist::build_block_jacobi_pc` with `pc_local` selecting
  the local ILU/ILUT/ILUTP block solver.
- `pc_global=asm` → `preconditioner::asm::AsmPc`, which dispatches to the distributed ASM path
  when a distributed layout is present.
- `pc_global=ras` → `preconditioner::asm::AsmPc` with restricted additive Schwarz mode enabled.
  It reuses the existing `pc_asm_*` options (overlap, weighting, subdomain sizing, etc.) rather
  than introducing separate `pc_global`-specific flags.

Supported combinations:

- Distributed operators: `DistCsrOp` with `UniverseComm` size > 1.
- Local operators or size-1 communicators: ignore `pc_global` and use the standard `pc_type` path.

### Example: FGMRES + RAS + ILUTP (MPI)

```bash
cargo mpirun -n 4 --example matrix_market_demo --features backend-faer,mpi -- \
  -ksp_type fgmres \
  -pc_global ras \
  -pc_asm_overlap 1 \
  -pc_asm_weighting uniform \
  -pc_local ilutp \
  -pc_ilutp_max_fill 20 \
  -pc_ilutp_drop_tol 1e-4 \
  -pc_ilutp_perm_tol 0.1
```

## Distributed route selection policy (`pc_dist_route`)

Standard terms used in diagnostics and docs:

- **native distributed**: DistCsr-native route and apply kernels.
- **adapted local wrapper**: compatibility/local-wrapper route (`wrapped_local`).
- **gather route**: root-gather heavy coarse/global route (`root_gather`).

Selection policy is native-first by default:

- `pc_dist_route=native` (default) prefers the **native distributed** route whenever possible.
- `pc_dist_route=adapted` forces the **adapted local wrapper** route and should be treated as fallback-only.
- `pc_dist_route=root_gather` selects the **gather route** for coarse/global workflows.

When running with `DistCsrOp` and communicator size > 1, Kryst attempts native distributed Block-Jacobi/ILU routes first (including auto-promotion from local PC settings when safe), then records fallback transitions when native setup fails.

Kryst now emits early option-preflight warnings for contradictory distributed settings (for example, `pc_dist_route=native` combined with `pc_dist_local_apply=wrapped_local`).

### Playbook: force native distributed path

- Prefer:
  - `-pc_dist_route native`
  - `-pc_dist_local_apply distributed_native` (or `strict` when you want setup to fail instead of fallback)
  - leave `-pc_global` unset unless you explicitly need `asm|ras|block_jacobi`
- Validate with `-ksp_view` or `-pc_view`:
  - `pc_dist_selected_route` should start with `distcsr_native_block_jacobi`
  - `pc_dist_option_warnings` should be empty

### Playbook: debug fallback behavior

- Turn on `-ksp_view` (or `-pc_view`) and inspect:
  - `pc_dist_effective_config` (machine-readable effective distributed config)
  - `pc_dist_option_warnings` (early contradictory-setting warnings)
  - `pc_dist_fallback_chain`, `pc_dist_fallback_reason`, `pc_dist_fallback_counters`
  - `pc_dist_preflight_*` fields for native-readiness probes
- For benchmarking/replay artifacts (`distcsr_bench_runner`), check:
  - `pc_dist_effective_config`
  - `pc_dist_option_warnings`
  - `pc_dist_selected_route` + fallback fields

## Common errors

- Communicator mismatch: `Amat.comm()` and `Pmat.comm()` are not congruent.
- Unsafe override: trying to relabel a nontrivial operator to an incompatible communicator.
- Dimension mismatch between `Amat` and `Pmat`.
- Using a replicated operator with a nontrivial communicator without wrapping.
