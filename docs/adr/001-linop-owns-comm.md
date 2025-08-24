# ADR-001: `LinOp::comm()` is the source of truth

## Status
Accepted

## Context
We integrate distributed libs (SuperLU_DIST, ASM/AMG). Passing comm via KSP/PC led to drift and bugs.

## Decision
Expose `fn comm(&self) -> UniverseComm` on `LinOp`; enforce `A.comm()==P.comm()` in `KspContext::set_operators`.

## Consequences
+ One invariant to check, less wiring.
+ Deferred/chain PCs can be built from `P` without extra context.
- Breaking change for `direct_solve` (no `comm` param).

