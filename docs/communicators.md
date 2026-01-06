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

## Common errors

- Communicator mismatch: `Amat.comm()` and `Pmat.comm()` are not congruent.
- Unsafe override: trying to relabel a nontrivial operator to an incompatible communicator.
- Dimension mismatch between `Amat` and `Pmat`.
- Using a replicated operator with a nontrivial communicator without wrapping.
