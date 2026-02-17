#[path = "infra/datasets.rs"]
mod datasets;

use criterion::{criterion_group, criterion_main, Criterion};
use kryst::config::options::KspOptions;
use kryst::context::ksp_context::{KspContext, SolverType};
use kryst::matrix::{CsrMatrix, DistCsrOp};
use kryst::parallel::{NoComm, UniverseComm};
use kryst::preconditioner::PcSide;
use kryst::Comm;
use std::sync::Arc;

#[cfg(feature = "mpi")]
use kryst::parallel::MpiComm;

fn bench_comm() -> UniverseComm {
    #[cfg(feature = "mpi")]
    {
        UniverseComm::Mpi(Arc::new(MpiComm::new()))
    }
    #[cfg(not(feature = "mpi"))]
    {
        UniverseComm::NoComm(NoComm)
    }
}

fn build_part_prefix(n_global: usize, size: usize) -> Vec<usize> {
    let base = n_global / size;
    let rem = n_global % size;
    let mut prefix = Vec::with_capacity(size + 1);
    prefix.push(0);
    let mut acc = 0usize;
    for rank in 0..size {
        let n_local = base + usize::from(rank < rem);
        acc += n_local;
        prefix.push(acc);
    }
    prefix
}

fn slice_rows(a: &CsrMatrix<f64>, row_start: usize, n_local: usize) -> CsrMatrix<f64> {
    let row_end = row_start + n_local;
    let mut row_ptr = Vec::with_capacity(n_local + 1);
    row_ptr.push(0);
    let start_nnz = a.row_ptr()[row_start];
    let end_nnz = a.row_ptr()[row_end];
    let mut col_idx = Vec::with_capacity(end_nnz - start_nnz);
    let mut values = Vec::with_capacity(end_nnz - start_nnz);
    for row in row_start..row_end {
        let start = a.row_ptr()[row];
        let end = a.row_ptr()[row + 1];
        col_idx.extend_from_slice(&a.col_idx()[start..end]);
        values.extend_from_slice(&a.values()[start..end]);
        row_ptr.push(col_idx.len());
    }
    CsrMatrix::from_csr(n_local, a.ncols(), row_ptr, col_idx, values)
}

fn build_case(
    grid: usize,
    comm: &UniverseComm,
    pc_local: &str,
    apply_mode: &str,
) -> (KspContext, Vec<f64>, Vec<f64>) {
    let a_global = datasets::poisson2d_csr(grid);
    let n_global = a_global.nrows();
    let part_prefix = build_part_prefix(n_global, comm.size());
    let row_start = part_prefix[comm.rank()];
    let n_local = part_prefix[comm.rank() + 1] - row_start;

    let local = slice_rows(&a_global, row_start, n_local);
    let dist = DistCsrOp::from_local_rows(n_global, row_start, &local, &part_prefix, comm.clone())
        .expect("dist csr build");

    let opts = KspOptions::from_args(&[
        "-pc_global",
        "block_jacobi",
        "-pc_local",
        pc_local,
        "-pc_dist_local_apply",
        apply_mode,
    ])
    .expect("ksp options parse");

    let mut ksp = KspContext::new();
    ksp.set_type(SolverType::Gmres).unwrap();
    ksp.set_pc_type_from_str("ilu").unwrap();
    ksp.set_pc_side(PcSide::Left);
    ksp.set_restart(30);
    ksp.set_tolerances(1e-8, 0.0, 1e6, 200);
    ksp.set_from_options(&opts).unwrap();
    ksp.set_operators(Arc::new(dist), None);

    let b = vec![1.0f64; n_local];
    let x = vec![0.0f64; n_local];
    (ksp, b, x)
}

fn bench_suite(c: &mut Criterion) {
    let comm = bench_comm();
    let strong_sizes = [("strong_64", 64usize), ("strong_96", 96usize)];
    let weak_per_rank = [40usize, 64usize];

    let mut strong = c.benchmark_group("mpi_pc_dist_native_strong");
    for (label, grid) in strong_sizes {
        for pc_local in ["ilutp", "sor", "chebyshev"] {
            for apply_mode in ["wrapped_local", "distributed_native"] {
                let bench_mode = if apply_mode == "distributed_native" {
                    "native"
                } else {
                    "wrapper"
                };
                let (mut ksp, b, mut x) = build_case(grid, &comm, pc_local, apply_mode);
                strong.bench_function(format!("{label}_{pc_local}_{bench_mode}"), |ben| {
                    ben.iter(|| {
                        x.fill(0.0);
                        let _ = ksp.solve(&b, &mut x).unwrap();
                    });
                });
            }
        }
    }
    strong.finish();

    let mut weak = c.benchmark_group("mpi_pc_dist_native_weak");
    for base in weak_per_rank {
        let global_grid = base * comm.size().max(1);
        for pc_local in ["ilutp", "sor", "chebyshev"] {
            for apply_mode in ["wrapped_local", "distributed_native"] {
                let bench_mode = if apply_mode == "distributed_native" {
                    "native"
                } else {
                    "wrapper"
                };
                let (mut ksp, b, mut x) = build_case(global_grid, &comm, pc_local, apply_mode);
                weak.bench_function(format!("weak_{base}_{pc_local}_{bench_mode}"), |ben| {
                    ben.iter(|| {
                        x.fill(0.0);
                        let _ = ksp.solve(&b, &mut x).unwrap();
                    });
                });
            }
        }
    }
    weak.finish();
}

criterion_group!(benches, bench_suite);
criterion_main!(benches);
