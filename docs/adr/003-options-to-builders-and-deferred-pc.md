# ADR-003: Options to builders and deferred PC construction

## Status
Accepted

## Context
Passing raw option maps into solvers tangled configuration and initialization.

## Decision
Translate `PcOptions` into typed builders (`PcConfig`) and defer preconditioner
construction until a matrix is available.

## Consequences
+ Clear separation between option parsing and PC setup.
+ Enables deferred PCs and chaining without solver churn.
- Slightly more boilerplate when adding new PCs.

