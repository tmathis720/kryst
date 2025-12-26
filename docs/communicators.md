# Communicator rules

kryst follows PETSc-style communicator ownership: the operator decides the communicator and the
solver/preconditioner follows it. The rules below describe how `KspContext` binds to communicators
and what is considered valid.

## Binding and congruence

- `KspContext` is bound to a communicator once operators are set.
- `try_set_operators(Amat, Pmat)` requires `Amat.comm() == Pmat.comm()`; a mismatch is an error.
- Dimensions must match: `Amat.dims() == Pmat.dims()`.
- Communicators are considered compatible only if they are IDENT or CONGRUENT
  (MPI_Comm_compare semantics).

## Explicit communicator wrapping

`try_set_operators_with_comm(Amat, Pmat, comm)` wraps the operators with an explicit communicator
and then applies the same congruence checks. This is intended for cases where the operator is
replicated or trivial (`UniverseComm::NoComm`) but the solver should run on a specific MPI
communicator.

## Recommended usage patterns

- Create the `KspContext` on the communicator you intend to solve on.
- Build operators on the same communicator and call `try_set_operators`.
- For subcommunicator solves, build the subcomm first, then create the KSP and operators on it.
- Avoid mixing `WORLD` operators with subcomm KSPs.

## Common errors

- Communicator mismatch: `Amat.comm() != Pmat.comm()`.
- Dimension mismatch between `Amat` and `Pmat`.
- Using a replicated operator with a nontrivial communicator without wrapping.
