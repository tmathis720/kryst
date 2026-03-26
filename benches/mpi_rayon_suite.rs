#[path = "infra/datasets.rs"]
mod datasets;

use criterion::{Criterion, criterion_group, criterion_main};
use kryst::algebra::parallel_cfg::{
    ParallelTune, ParallelTunerMode, set_parallel_tune, set_parallel_tuner_mode,
};
use kryst::config::options::KspOptions;
use kryst::context::ksp_context::{KspContext, SolverType};
use kryst::matrix::{CsrMatrix, DistCsrOp};
use kryst::parallel::{Comm, NoComm, UniverseComm};
use kryst::preconditioner::PcSide;
use serde_json::Value;
use std::sync::Arc;

#[cfg(feature = "mpi")]
use kryst::parallel::MpiComm;
#[cfg(feature = "mpi")]
use kryst::preconditioner::asm::alltoallv_u64_sparse;

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

fn dist_fallback_total(ksp: &KspContext) -> usize {
    let view = ksp.view();
    let Some(Value::Object(obj)) = view.solver_config.get("pc_dist_fallback_counters") else {
        return 0;
    };
    obj.values().filter_map(|v| v.as_u64()).sum::<u64>() as usize
}

fn run_once(
    grid: usize,
    pc_type: &str,
    comm: &UniverseComm,
    threads: Option<usize>,
) -> (std::time::Duration, usize, usize) {
    let (mut ksp, b, mut x) = build_case(grid, pc_type, comm, threads);
    x.fill(0.0);
    let start = std::time::Instant::now();
    let stats = ksp.solve(&b, &mut x).unwrap();
    (
        start.elapsed(),
        stats.counters.num_global_reductions,
        dist_fallback_total(&ksp),
    )
}

fn check_reduction_budget(tag: &str, reductions: usize, n_local: usize) {
    let enforce = std::env::var("KRYST_BENCH_ENFORCE_SCALING")
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let budget = ((4.0 * n_local.max(1) as f64).ceil() as usize).max(24);
    if reductions > budget {
        let msg = format!(
            "{tag}: reduction budget exceeded (reductions={reductions}, budget={budget}, n_local={n_local})"
        );
        if enforce {
            panic!("{msg}");
        } else {
            eprintln!("WARN: {msg}");
        }
    }
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
            let (base_time, base_red, base_fb) = run_once(grid, pc_type, &comm, Some(1));
            let (cfg_time, cfg_red, cfg_fb) = run_once(grid, pc_type, &comm, threads);
            println!(
                "{pc_type}:{label}:g{grid} delta => reductions={} runtime_ms={:.3} fallback_delta={} (threads={:?})",
                base_red as isize - cfg_red as isize,
                (base_time.as_secs_f64() - cfg_time.as_secs_f64()) * 1.0e3,
                base_fb as isize - cfg_fb as isize,
                threads
            );

            let (mut ksp, b, mut x) = build_case(grid, pc_type, &comm, threads);
            x.fill(0.0);
            let stats = ksp.solve(&b, &mut x).unwrap();
            check_reduction_budget(
                &format!("{pc_type}:{label}:g{grid}"),
                stats.counters.num_global_reductions,
                b.len(),
            );
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

fn bench_adaptive_tuner_configs(c: &mut Criterion) {
    let comm = bench_comm();
    let comm_size = comm.size();
    let cfgs = [
        ("mpi_only", Some(1usize), comm_size > 1),
        ("rayon_only", Some(4usize), comm_size <= 1),
        ("hybrid", Some(4usize), comm_size > 1),
    ];
    let grid = 64usize;
    let pc_type = "ilu";
    let baseline = ParallelTune::default();

    let mut group = c.benchmark_group("adaptive_tuner_modes");
    for (label, threads, enabled) in cfgs {
        if !enabled {
            continue;
        }

        set_parallel_tune(baseline);
        set_parallel_tuner_mode(ParallelTunerMode::Manual);
        let (manual_time, _, _) = run_once(grid, pc_type, &comm, threads);

        set_parallel_tune(baseline);
        set_parallel_tuner_mode(ParallelTunerMode::Adaptive);
        let (adaptive_time, _, _) = run_once(grid, pc_type, &comm, threads);

        let ratio = manual_time.as_secs_f64() / adaptive_time.as_secs_f64().max(1e-12);
        println!(
            "adaptive:{label}: threads={threads:?} comm_size={comm_size} speedup={ratio:.3} manual_ms={:.3} adaptive_ms={:.3}",
            manual_time.as_secs_f64() * 1e3,
            adaptive_time.as_secs_f64() * 1e3
        );
        assert!(
            ratio >= 0.9,
            "adaptive tuner regressed {label} scenario: speedup={ratio:.3}"
        );

        let bench_id = format!("{label}_g{grid}");
        group.bench_function(bench_id, |ben| {
            let (mut ksp, b, mut x) = build_case(grid, pc_type, &comm, threads);
            ben.iter(|| {
                set_parallel_tuner_mode(ParallelTunerMode::Adaptive);
                x.fill(0.0);
                let _ = ksp.solve(&b, &mut x).unwrap();
            });
        });
    }
    group.finish();
    set_parallel_tuner_mode(ParallelTunerMode::Manual);
    set_parallel_tune(baseline);
}

#[cfg(feature = "mpi")]
fn sparse_ring_peers(rank: usize, size: usize, degree: usize) -> Vec<usize> {
    if size <= 1 || degree == 0 {
        return Vec::new();
    }
    let span = degree.min(size.saturating_sub(1));
    let mut peers = Vec::with_capacity(span);
    for step in 1..=span {
        peers.push((rank + step) % size);
    }
    peers
}

#[cfg(feature = "mpi")]
fn bench_sparse_exchange(c: &mut Criterion) {
    let comm = bench_comm();
    let size = comm.size();
    if size <= 1 {
        return;
    }
    let rank = comm.rank();
    let degree = std::env::var("KRYST_BENCH_SPARSE_DEGREE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(4)
        .min(size.saturating_sub(1));
    let payload_words = std::env::var("KRYST_BENCH_SPARSE_WORDS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(16);

    let send_peers = sparse_ring_peers(rank, size, degree);
    let recv_peers = (1..=send_peers.len())
        .map(|step| (rank + size - step) % size)
        .collect::<Vec<_>>();

    let mut group = c.benchmark_group("mpi_sparse_exchange");
    group.bench_function(format!("setup_r{size}_d{degree}_w{payload_words}"), |ben| {
        ben.iter(|| {
            let payloads: Vec<Vec<u64>> = send_peers
                .iter()
                .map(|&peer| vec![peer as u64; payload_words])
                .collect();
            std::hint::black_box(payloads);
        });
    });

    let payloads: Vec<Vec<u64>> = send_peers
        .iter()
        .map(|&peer| vec![peer as u64; payload_words])
        .collect();
    group.bench_function(format!("apply_r{size}_d{degree}_w{payload_words}"), |ben| {
        ben.iter(|| {
            let recv = alltoallv_u64_sparse(&comm, &send_peers, &payloads, &recv_peers)
                .expect("sparse exchange");
            std::hint::black_box(recv);
        });
    });
    group.finish();
}

#[cfg(feature = "mpi")]
criterion_group!(
    benches,
    bench_suite,
    bench_adaptive_tuner_configs,
    bench_sparse_exchange
);
#[cfg(not(feature = "mpi"))]
criterion_group!(benches, bench_suite, bench_adaptive_tuner_configs);
criterion_main!(benches);
