# Milestone 0: CSR source audit and performance baseline

This document describes the source tree at Kryst 4.6.0. It is an audit, not a
claim that similarly named types share storage. In particular, the repository
currently has two independent owned host CSR types.

## Representation inventory

| Representation | Storage and ownership | Operator route | Copies / notes |
|---|---|---|---|
| `matrix::sparse::CsrMatrix<T>` | Primary public owned CSR: `Vec<usize>` row pointers and columns, `Vec<T>` values, plus `diag_pos`. | Implements `LinOp`, `SparseMatrix`, and `CsrMatRef`. | This is `matrix::Csr`, `matrix::CsrMatrix`, and Faer's `DefaultCsrMat`; those names are aliases, not extra storage. With `simd`, its optional `spmv_plan` contains a second CSR copy, but its `spmv` method does not consult that field. |
| `matrix::csr::CsrMatrix<S>` | Second owned CSR with public `rowptr`, `colind`, and `values`. | Implements `CsrMatRef`; normally wrapped by `GenericCsrOp`. | Used as SIMD-plan input and in scalar-generic bridges. Conversion from the primary CSR clones both index arrays and values. It has no `diag_pos`. |
| `CsrOp<S>` | `Arc<primary CSR>` plus change IDs, communicator/layout, and optional transpose cache. | `LinOp::try_matvec` selects the canonical Rayon-aware CSR entry point. | Shares the primary arrays. The wrapper itself adds no CSR copy. |
| `GenericCsrOp<S>` | `Arc<second CSR>` plus an owned `SpmvPlan<S>`. | `LinOp::try_matvec` calls `plan.apply_scaled`. | `SpmvPlan::build` clones the second CSR, so the operator retains the matrix arrays and a second plan copy. SELL-C, when selected, adds padded/permuted storage. |
| `CsrMatRef` / `CsrMatMut` | Borrowed interfaces returning CSR slices. | Canonical serial and Rayon kernels are generic over `CsrAccess`, a blanket extension of `CsrMatRef`. | These are traits, not concrete view structs. There is no lifetime-bearing `CsrView` type. `SparseMatrix` is an older read-only slice interface implemented only by the primary CSR. |
| `DistRowCsr<S>` | Owns a primary CSR containing local rows and global columns, plus row offset/global width. | Extraction/helper type; not a `LinOp`. | `DistCsrOp::local_rows_csr` clones the distributed arrays before wrapping them. |
| `LocalSquareCsr<S>` | Owns a primary local square CSR. | Local factorization/helper type; not a `LinOp`. | `DistCsrOp::local_square_block` filters and clones the diagonal block. |
| `DistCsrOp` | Canonical distributed operator. Retains local rows with global columns and also owns row masks/ranges, translated local-diagonal CSR plan, translated ghost columns, duplicated ghost values, halo index maps, and reusable communication buffers. | Implements both `LinOp` and `KLinOp`. | Construction clones the input local CSR and builds all translations/plans once. It intentionally duplicates subsets for local/ghost kernels. |
| `ParCsrMatrix` / `ParCsrOp` | Deprecated split diagonal/off-process primary CSR blocks, owned/ghost maps, legacy halo, and lazy `OnceLock<Arc<DistCsrOp>>`. | `ParCsrOp -> ParCsrMatrix::spmv -> canonical DistCsrOp`. | First use merges and sorts both blocks into a new global-column CSR, then `DistCsrOp` clones/plans it. `spmv_scaled` allocates a local temporary vector on every call. |
| `sprs::CsMat<f64>` | Backend-owned compressed sparse storage. | Direct `LinOp` implementation when `backend-sprs` is enabled. | It does not share arrays with Kryst CSR. A CSC `CsMat` is converted to a new CSR on every `matvec`; a CSR `CsMat` is traversed directly. |
| `CudaCsrOp` | cuSPARSE CSR descriptor, device indices/values, cached SpMV descriptors/workspace, and host copies of row pointers, columns, values, and diagonal. | Separate `CudaLinOp` API, not host-slice `LinOp`. | `from_host` converts `usize` indices to i32/i64 and uploads them. Host and device data are deliberately duplicated. `host_csr_parts` clones the host cache back out. |
| `CudaDistCsrOp` | Device-resident diagonal/off-diagonal `CudaCsrOp`s, translated ghost indices, device work vectors, and staged pinned or device-direct MPI buffers. | Separate distributed `CudaLinOp` path. | Construction splits and translates once. It exchanges only referenced ghost values. |

`backend-faer` does not wrap a Faer sparse matrix. Its CSR associated type is
the primary Kryst CSR. `backend-nalgebra` and `backend-naive` declare no sparse
storage (`Csr = ()`). Preconditioner-specific types such as `IluCsr`,
`FsaiCsr`, and `SpaiCsr` contain factor/approximate-inverse CSR data but are not
general matrix representations or primary SpMV entry points.

## Construction and conversion paths

### Matrix Market

`read_matrix_market` parses coordinate or array input into
`MatrixMarketData` triplet arrays. `to_csr_matrix_scalar` converts values to the
active scalar, sorts triplets by `(row, column)`, expands symmetric,
skew-symmetric, or Hermitian entries, constructs row pointers, and moves the
columns/values into the primary CSR. Symmetry expansion and sorting allocate;
duplicate coordinates are retained rather than coalesced. `to_csr_matrix` then
converts the scalar result back to real arrays and constructs another primary
CSR, so the real convenience route has an additional full copy.

The Matrix Market examples either use that primary CSR directly or slice its
rows into another primary CSR before calling `DistCsrOp::from_local_rows`.

### Backend and format conversions

- Faer dense to CSR/CSC scans the dense matrix and allocates new compressed
  arrays. CSR to CSC and CSC to CSR count entries and allocate complete new
  pointer/index/value arrays without densifying.
- `AsFormat::as_csr` is the zero-copy borrowed route. Calling
  `to_csr_cached` on an already-CSR primary matrix still returns an `Arc` around
  a clone. Dense conversions use global weak caches keyed by pointer, change
  IDs, and drop tolerance.
- Complex conversions in `matrix::convert` are implemented separately and
  allocate new primary CSR/CSC storage.
- The sprs backend materializes `sprs::CsMat` and its own dense wrapper; there
  is no zero-copy bridge between `CsMat` and either Kryst CSR.
- `matrix::csr::CsrMatrix::from_real_csr` and
  `GenericCsrOp::from_real_csr` clone indices and lift every value.
- `CudaCsrOp::from_host` uploads converted index/value buffers and retains host
  copies. The reverse helper returns cloned parts rather than a shared view.

### Transpose cache

With `transpose-cache`, `CsrOp` owns
`RwLock<Option<(ValuesId, Arc<CscMatrix<S>>)>>`. Real transpose apply calls
`ensure_csc_view`, reuses the CSC when `ValuesId` matches, or converts the full
CSR and replaces the cache. Complex builds deliberately use the scalar CSR
conjugate-gather path and do not read this cache. Direct primary-CSR transpose
also gathers from CSR and does not use `CsrOp`'s cache.

### SIMD plan

`SpmvPlan::build` delegates to `build_owned(matrix.clone(), tuning)`. The plan
therefore owns a complete second CSR. In real `simd` builds and above
`min_nnz_for_simd`, construction microbenchmarks gather CSR and optionally
SELL-C, selecting `CsrSimdGather` or `SellC`; otherwise it selects `Scalar`.
Complex builds always select `Scalar`.

The plan is reached by `GenericCsrOp` and by `DistCsrOp`'s local-diagonal
strategy. Although the primary CSR exposes `build_spmv_plan`, no primary-CSR
apply method reads its `spmv_plan` field. Building that cache currently
duplicates storage without changing `CsrMatrix::spmv` performance.

## Distributed vector and communicator

`DistCsrOp` does not require a distributed vector object. Its `LinOp`
dimensions are local-by-local and it consumes the owned local `x`/`y` slices;
the operator's `DistLayout` supplies global dimensions and ownership ranges.
`DistVecS` is a separate preconditioner adapter representation with owned or
borrowed local storage and an owned or borrowed reusable scratch vector.

Distributed operators communicate through `UniverseComm`, the concrete enum
over `NoComm`, `MpiComm`, and (when enabled) `RayonComm`. It implements the
`Comm` abstraction: rank/size/barrier, scatter/gather, reductions,
nonblocking f64/u64 point-to-point calls, and `wait_all`. Complex halo values
are exposed to that real-valued point-to-point interface as interleaved real
components. Rayon is local execution, not a distributed transport.

## Exact `y = A x` paths

The following paths describe the normal forward apply after construction.

| Configuration / entry object | Call path | Output and checks | Per-call work |
|---|---|---|---|
| Real or complex, primary CSR as `LinOp` | dynamic `LinOp::try_matvec` -> primary `try_spmv` -> `csr_matvec` -> `spmv_csr_serial` -> monomorphized `csr_row_dot` | Checked at `spmv_csr_serial`; every `y[row]` is overwritten. | No temporary, translation, repartition, or row-loop dynamic dispatch. The primary `LinOp` route stays serial even in a Rayon build. |
| Real or complex, `CsrOp` without Rayon | dynamic `LinOp::try_matvec` -> `CsrOp::try_matvec` -> `csr_matvec_par` -> non-Rayon `spmv_csr_parallel` -> serial path | Checked in the wrapper and again in the kernel; overwrite. | No steady-state temporary or conversion. |
| Real or complex, `CsrOp` with Rayon (or explicit `try_spmv_parallel`) | `csr_matvec_par` -> `spmv_csr_parallel`; below `min_rows_spmv` it calls serial, otherwise `y.par_chunks_mut(max(64, chunk_rows_spmv))` -> `csr_row_dot` | Checked before threshold selection; overwrite. | Rayon tasks are formed each call but rows/entries are not copied or repartitioned. Work is row-chunked, not nonzero-balanced. No dynamic dispatch in a row. |
| Planned real/SIMD | dynamic `LinOp::try_matvec` -> `GenericCsrOp::try_matvec` -> `SpmvPlan::apply_scaled(1, 0)` -> one plan match -> scalar, gather-SIMD, or SELL-C kernel | Checked in `GenericCsrOp`; beta zero overwrites `y`. | No apply-time conversion/repartition. The scalar and gather kernels allocate no temporary. The current SELL-C kernel allocates one accumulator `Vec` per nonempty slice on every call. Kernel dispatch occurs once per apply, outside row loops. This path is not Rayon-parallel. |
| Planned complex, with or without `simd` | same planned route -> `SpmvKernel::Scalar` -> `spmv_scaled_csr` | Checked in wrapper; overwrite. | No temporary. Complex SIMD is unsupported by construction. This planned route is serial even when Rayon is enabled. |
| MPI without Rayon, `DistCsrOp` | dynamic `LinOp::try_matvec` -> `DistCsrOp::matvec` -> `KLinOp::matvec_s` -> clear `y` -> post halo -> local-diagonal plan or local row spans -> wait/unpack -> ghost contribution | Checked in `try_matvec`; `matvec_s` also asserts. Local planned apply overwrites, then ghost contribution accumulates; externally `y` is overwritten. | Only requested unique ghosts are exchanged. Packing uses prebuilt schedules. `complete_halo` clones the flat ghost vector, allocating when ghosts exist. No indices are translated or rows repartitioned per call. |
| Hybrid MPI + Rayon | same DistCSR route; `RowSplitScalar` uses Rayon `par_iter_mut` for local and border rows, while `LocalDiagSpmvPlan` remains a serial planned kernel | Same overwrite/accumulate behavior and checks. | MPI posting/wait may overlap local work in `Interior` mode. Rayon is used only for the row-split strategy; planner selection can make a nominal hybrid run locally serial. No row-loop dynamic dispatch. |
| Legacy `ParCsrOp` | `ParCsrOp::try_matvec` -> `ParCsrMatrix::spmv_scaled` -> lazy canonical conversion -> allocate `tmp` -> `DistCsrOp::try_matvec` -> scale/copy into `y` | Checked by legacy wrapper and DistCSR; public result is overwritten for beta zero or accumulated for nonzero beta. | Canonical conversion occurs once, but `tmp = vec![0; local_n]` is allocated every apply. |
| sprs CSR backend | dynamic `LinOp::matvec` -> `CsMat` storage check -> `outer_iterator` row loop | No explicit dimension error; fills `y` then overwrites rows. | CSR storage has no conversion/temp. CSC storage calls `to_csr` and allocates every apply. |
| CUDA CSR backend | `CudaLinOp::apply` -> device/dimension checks -> lazy cached descriptors/workspace -> cuSPARSE `SpMV(alpha=1,beta=0)` | Device and dimensions checked; output overwritten. | usize-to-i32/i64 translation and upload occur at construction. First apply can allocate cached descriptors/workspace; prepared steady state reuses them. Dispatch is inside cuSPARSE, not a Rust row loop. |
| Distributed CUDA | `CudaDistCsrOp::apply` -> pack requested device entries -> staged or device-direct MPI exchange while diagonal cuSPARSE apply runs -> off-diagonal cuSPARSE apply -> device axpby | Device/local dimensions checked; final output overwritten. | Only ghosts are exchanged; all split/translation/workspace allocation is done at construction. |

`DistCsrOp` chooses `Disabled` or `Interior` overlap and
`RowSplitScalar` or `LocalDiagSpmvPlan` once during construction from fixed
metrics. Neither partition ownership, row masks, row ranges, column
translations, neighbor lists, nor packing schedules are rebuilt on an apply.

## Allocation and dispatch summary

- The canonical serial and warmed Rayon host kernels require no matrix/vector
  temporary allocation.
- `GenericCsrOp` scalar and gather-SIMD plans allocate at setup and are
  allocation-free when warmed. The SELL-C apply allocates one accumulator per
  nonempty slice; the checked-in baseline exposes this rather than hiding it.
- A halo-bearing `DistCsrOp` currently allocates the returned ghost clone once
  per apply. Reentrant concurrent calls can also grow its halo-context pool.
- Legacy `ParCsrMatrix::spmv_scaled` always allocates its local temporary.
- Dynamic `LinOp` dispatch occurs at the operator boundary. CSR row loops are
  statically dispatched. Plan kernel dispatch is one enum match per apply.
- No CPU path gathers an entire distributed vector for DistCSR SpMV.

## Reproducible baselines

`src/bin/spmv_baseline.rs` contains deterministic generators for diagonal,
5-point, 7-point, irregular, short-row, long-row, empty-row, structurally
symmetric, and nonsymmetric matrices. Every timed implementation is checked
against an independent CSR reference first. The runner emits JSON Lines with:

- nanoseconds per SpMV and per nonzero;
- modeled effective bandwidth;
- allocations per SpMV;
- local compute, packing, request-posting communication, wait, and unpack time;
- shared-memory or distributed strong-scaling efficiency;
- static contiguous thread-partition imbalance and observed rank work/time
  imbalance;
- DistCSR plan, kernel, overlap, and halo-volume diagnostics.

The bandwidth model counts values, column indices, one indirect `x` scalar per
nonzero, row pointers, and output writes. It is a stable effective-bandwidth
model, not a hardware-counter measurement. `communication` is CPU time to post
requests; incomplete transfer time is intentionally reported as `wait`.
Thread imbalance is a reproducible contiguous-row work proxy because the
production Rayon kernel does not expose worker-level counters.

Run the full real/complex and 1/2/4/8-rank suite (rank counts above available
resources can be capped):

```bash
KRYST_M0_SIZE=16384 KRYST_M0_ITERATIONS=30 \
KRYST_M0_MAX_MPI_RANKS=8 scripts/run_spmv_baseline.sh
```

For a quick correctness/performance smoke run:

```bash
cargo run --release --bin spmv_baseline -- \
  --size 1024 --iterations 3 --cases diagonal,stencil5,empty_rows
```

The suite writes raw JSONL and an aggregated artifact under
`benchmarks/milestone0/artifacts/`. The aggregator calculates distributed
strong-scaling efficiency only when a matching one-rank record exists; it
never substitutes made-up rank timings. Baseline artifacts must record the
host, compiler, feature set, matrix size, and iteration count and must be
regenerated before any optimization comparison.
