#[path = "infra/datasets.rs"]
mod datasets;

use criterion::{Criterion, criterion_group, criterion_main};
use kryst::config::options::KspOptions;
use kryst::context::ksp_context::{KspContext, SolverType};
use kryst::matrix::{CsrMatrix, DistCsrOp};
use kryst::parallel::{NoComm, UniverseComm};
use kryst::preconditioner::PcSide;
use std::sync::Arc;

#[cfg(feature = "mpi")]
use kryst::parallel::MpiComm;

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

fn build_case(
    grid: usize,
    pc_type: &str,
    comm: &UniverseComm,
    threads: Option<usize>,
) -> (KspContext, Vec<f64>, Vec<f64>) {
    let a_global = datasets::poisson2d_csr(grid);
    let n_global = a_global.nrows();
    let part_prefix = build_part_prefix(n_global, comm.size());
    let row_start = part_prefix[comm.rank()];
    let n_local = part_prefix[comm.rank() + 1] - row_start;

    let local = slice_rows(&a_global, row_start, n_local);
    let dist = DistCsrOp::from_local_rows(n_global, row_start, &local, &part_prefix, comm.clone())
        .expect("dist csr build");

    let mut ksp = KspContext::new();
    ksp.set_type(SolverType::Gmres).unwrap();
    ksp.set_pc_type_from_str(pc_type).unwrap();
    ksp.set_pc_side(PcSide::Left);
    ksp.set_restart(30);
    ksp.set_tolerances(1e-8, 0.0, 1e6, 200);

    if let Some(n) = threads {
        let mut opts = KspOptions::default();
        opts.threads = Some(n);
        ksp.set_from_options(&opts).unwrap();
    }

    ksp.set_operators(Arc::new(dist), None);

    let b = vec![1.0f64; n_local];
    let x = vec![0.0f64; n_local];
    (ksp, b, x)
}

fn bench_suite(c: &mut Criterion) {
    let comm = bench_comm();
    let sizes = [("small", 16usize), ("medium", 48usize), ("large", 96usize)];
    let threads = std::env::var("KRYST_BENCH_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok());

    for pc_type in ["ilu", "asm"] {
        let mut group = c.benchmark_group(format!("dist_{pc_type}"));
        for (label, grid) in sizes {
            let (mut ksp, b, mut x) = build_case(grid, pc_type, &comm, threads);
            let bench_id = format!("{label}_g{grid}");
            group.bench_function(bench_id, |ben| {
                ben.iter(|| {
                    x.fill(0.0);
                    let _ = ksp.solve(&b, &mut x).unwrap();
                });
            });
        }
        group.finish();
    }
}

criterion_group!(benches, bench_suite);
criterion_main!(benches);
