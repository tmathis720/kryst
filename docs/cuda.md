# CUDA backend

The `cuda` feature is non-default and uses `cudarc` with dynamic loading. A
CUDA toolkit is not needed to compile `kryst`; creating `CudaRuntime` requires
the CUDA driver, cuBLAS, and cuSPARSE shared libraries at runtime. CUDA 12.0 is
the binding baseline. CUDA 12 and CUDA 13 GPU jobs exercise the same binary API
surface.

## Supported surface

| Area | Supported |
| --- | --- |
| Scalars | `f64`; `complex64` with `cuda,complex` |
| Storage | `CudaVector`, explicit upload/download |
| Operators | 32/64-bit CSR SpMV and dense GEMV |
| Operations | normal, transpose, conjugate-transpose |
| Solvers | classical/pipelined CG and PCG, CGNR, CR-via-CGNR, LSQR, LSMR and unpreconditioned QMR/TFQMR/TCQMR, classical/reduced-collective pipelined restarted GMRES and FGMRES, GCR-via-FGMRES, device-resident PipeGCR, right-preconditioned BiCGStab and CGS, Richardson, Chebyshev KSP |
| Preconditioners | None, CSR Jacobi, block Jacobi, polynomial Chebyshev, host-factorized ILU(0), host-built/device-resident aggregation AMG |
| Reuse | solver workspace, SpMV descriptors/workspace, numeric value uploads |
| Diagnostics | allocation and H2D/D2H/D2D bytes, library calls, launches, synchronizations, setup/solve counts and wall time |
| Distributed | one device per rank, GPU halo packing, pinned-host staging or explicit CUDA-aware device-pointer MPI, global scalar reductions |

Unsupported solver/preconditioner combinations fail during configuration or
setup. They never fall back to CPU execution.

`CudaCsrOp::update_values` updates the existing device allocation and advances
its `ValuesId`; it does not upload row offsets or column indices. The next KSP
setup refreshes an automatically constructed Jacobi preconditioner while
retaining the solver workspace. A structure change requires constructing and
attaching a new operator.

Block Jacobi performs partial-pivoting dense block inversion during setup,
uploads a block-diagonal inverse CSR operator, and applies it with cuSPARSE.
`CudaKspContext::set_block_jacobi_size` controls the block width. Numeric-only
source updates refactor the host blocks and refresh only the inverse values.

`CudaKspContext::set_cg_variant(CudaCgVariant::Pipelined)` selects a
Chronopoulos/Gear-style PCG recurrence. Its two local inner products are
written into a reusable device payload, transferred together, and combined by
one MPI collective per ordinary iteration. True-residual validation restarts
the recurrence if its recursive residual is optimistic.

`CudaKspContext::set_gmres_variant(CudaGmresVariant::Pipelined)` selects a
classical-Gram-Schmidt Arnoldi step for GMRES and FGMRES that combines all
projection coefficients and the candidate-vector norm in one MPI collective.
The basis and candidate vector stay on the GPU. The current implementation
uses one checked-in fixed-tree multi-dot PTX launch for the entire payload,
with compensated per-thread accumulation and no iteration-time allocation.

Chebyshev KSP mirrors kryst's current host semantics: a damped stationary
iteration configured with `set_chebyshev_omega` (default `0.8`). It is separate
from the polynomial Chebyshev preconditioner. The latter is selected with
`PcType::Chebyshev` and configured using
`set_chebyshev_pc(degree, eig_lo, eig_hi)`. Its three recurrence vectors are
allocated during setup, and every application remains device-resident.

CUDA ILU(0) factorization runs on the host during setup, preserving the input
CSR sparsity pattern. The lower and upper factors are uploaded once, analyzed
with cuSPARSE SpSV, and applied with two device triangular solves through a
reused intermediate vector. Numeric-only matrix updates refactor on the host
and replace factor values without reallocating their device structure.

CUDA AMG builds pairwise aggregates, injection transfer operators, and
Galerkin coarse matrices on the host during setup. It uploads every level and
retains all V-cycle vectors on the device; damped Jacobi performs pre/post
smoothing and the coarse solve. `CudaAmgOptions` controls hierarchy depth,
coarse size, smoothing counts, coarse iterations, and damping. Numeric matrix
updates rebuild the hierarchy during setup; iteration loops never allocate or
transfer full vectors.

CUDA CGNR accepts rectangular operators and uses `A^T` for real scalars or
`A^H` for complex scalars. CR matches kryst's host contract by delegating to
the same normal-equations recurrence. None preconditioning works directly;
Jacobi and block Jacobi require a separate square `ncols x ncols`
preconditioning operator. The row- and column-sized workspaces are allocated
during setup and reused across solves.

LSQR and LSMR share the rectangular/adjoint storage foundation and follow the
host Golub-Kahan recurrences. As on the host, both currently support only
`PcType::None`; selecting or attaching a preconditioner fails during setup.

GCR likewise matches the host compatibility surface by using the flexible
GMRES recurrence. It is right-preconditioned and shares the preallocated CUDA
Krylov basis and restart control used by FGMRES.

PipeGCR uses dedicated device-resident `p` and `A p` bases. Its classical
projection payload is combined in one MPI collective regardless of restart
depth, and its step projection is a second collective; the recursive norm is
verified with a true residual at convergence and restart boundaries. QMR uses
CSR transpose (or conjugate-transpose for complex scalars) and currently
requires `PcType::None`; TFQMR is transpose-free and TCQMR follows the host
compatibility mapping to that kernel. Unsupported QMR-family preconditioning
is rejected during setup.

## Checked-in kernels

The `src/cuda/ptx` modules target PTX ISA 7.0 / `sm_50` as a portable virtual
architecture baseline. CUDA JIT-loads them through the driver. They implement
real and complex AXPBY, the fused CG `x/r` update, indexed halo gathering, and
a fixed-tree compensated multi-dot reduction for pipelined Arnoldi.
The crate enables cudarc's PTX container API but never invokes NVRTC; neither
NVRTC nor `nvcc` is required at build or runtime.

## Determinism

Set `CudaOptions::deterministic` to select CSR ALG2, disable cuBLAS atomics, and
use pedantic cuBLAS math. Repeated results are intended to be stable only when
GPU model, driver, CUDA libraries, algorithms, and launch configuration are
unchanged. CUDA library upgrades or different GPU architectures are outside
the bitwise reproducibility contract.

## Diagnostics and transfers

`CudaRuntime::diagnostics()` returns a cumulative snapshot. After `setup()`, a
device-vector solve reuses its work vectors and Krylov basis. Full-vector H2D
or D2H transfers occur only through vector upload/download methods and
`solve_host`; convergence reductions return scalar payloads to the host.

## Validation

CPU-only feature checks do not need CUDA installed:

```bash
cargo test --no-default-features --features cuda --test cuda_feature
cargo test --features "cuda,complex" --test cuda_feature
```

GPU tests are ignored in ordinary test runs and enabled by the dedicated CUDA
workflow:

```bash
cargo test --features cuda --test cuda_gpu -- --ignored --test-threads=1
cargo test --features "cuda,complex" --test cuda_gpu -- --ignored --test-threads=1
```

Generate the synchronized CPU/CUDA setup, first-solve, repeated-solve,
transfer-volume, and allocation report used to locate the crossover point:

```bash
KRYST_CUDA_BENCH_REPEATS=10 cargo run --release \
  --example cuda_crossover --features cuda -- 16 32 64 96
```

The command emits CSV and deliberately reports measured crossover rather than
asserting a fixed speedup. Run it on each supported GPU/driver stack and retain
the output with the CUDA CI artifacts.

The scheduled workflow expects self-hosted runner labels `cuda-12` and
`cuda-13`.

Distributed GPU validation additionally uses:

```bash
mpirun -n 2 cargo test --features "cuda,mpi" --test cuda_mpi \
  -- --ignored --test-threads=1
```

`CudaRuntime::for_local_rank` recognizes Open MPI, Intel/MPICH, MVAPICH, and
Slurm local-rank variables. It maps local rank to device and rejects
oversubscription unless `CudaOptions::allow_device_oversubscription` is set.

`CudaDistCsrOp::from_local_rows` accepts the same global-column local CSR and
partition representation as the host distributed operator. It splits local
and ghost blocks, retains both on-device, packs requested local entries with a
CUDA kernel, exchanges only compact halos through preallocated page-locked
buffers, overlaps the independent diagonal-block SpMV with the in-flight halo,
reuses a preallocated raw-MPI request slab without iteration-time host
allocation, and uses the existing MPI collectives for Krylov scalar reductions.
`CudaMpiTransport::DeviceDirect` passes the packed send and ghost receive
device pointers directly to nonblocking MPI and therefore requires a
CUDA-aware MPI implementation. Selecting it is an explicit capability
assertion; MPI failures are returned as structured CUDA library errors.
`Auto` and `Staged` use the portable pinned-host path because MPI exposes no
portable, side-effect-free CUDA-awareness query.

To exercise device-direct transport in the distributed test, set the opt-in
test flag only on a CUDA-aware MPI installation:

```bash
KRYST_TEST_CUDA_AWARE_MPI=1 mpirun -n 2 cargo test \
  --features "cuda,mpi" --test cuda_mpi -- --ignored --test-threads=1
```

## Deferred scope

Fill-level/threshold ILU variants, automatic CUDA-aware MPI discovery, and
deeper communication/computation scheduling remain follow-on work.
Unsupported selections fail explicitly rather than performing an implicit
host solve.
