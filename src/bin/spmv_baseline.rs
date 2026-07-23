//! Reproducible Milestone-0 CSR/DistCSR performance baseline runner.
//!
//! Results are emitted as JSON Lines so MPI launch scripts can collect one
//! artifact per process count without coupling measurements to Criterion's
//! statistical model.

use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::BTreeSet;
use std::hint::black_box;
use std::mem::size_of;
#[cfg(feature = "mpi")]
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::time::{Duration, Instant};

use kryst::algebra::prelude::*;
use kryst::matrix::DistCsrOp;
use kryst::matrix::dist_csr::DistSpmvProfile;
use kryst::matrix::op::LinOp;
use kryst::matrix::sparse::CsrMatrix;
use kryst::parallel::{Comm, NoComm, UniverseComm};
use serde_json::{Value, json};

#[cfg(feature = "mpi")]
use kryst::parallel::MpiComm;

struct CountingAllocator;

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL_ALLOCATOR: CountingAllocator = CountingAllocator;

#[derive(Clone, Debug)]
struct Options {
    size: usize,
    iterations: usize,
    cases: Option<BTreeSet<String>>,
    distributed_only: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            size: 16_384,
            iterations: 30,
            cases: None,
            distributed_only: false,
        }
    }
}

fn parse_options() -> Options {
    let mut options = Options::default();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--size" => {
                options.size = args
                    .next()
                    .expect("missing --size value")
                    .parse()
                    .expect("invalid --size");
            }
            "--iterations" => {
                options.iterations = args
                    .next()
                    .expect("missing --iterations value")
                    .parse()
                    .expect("invalid --iterations");
            }
            "--cases" => {
                let selected = args.next().expect("missing --cases value");
                options.cases = Some(selected.split(',').map(str::to_owned).collect());
            }
            "--distributed-only" => options.distributed_only = true,
            "--help" | "-h" => {
                println!(
                    "Usage: spmv_baseline [--size N] [--iterations N] [--cases a,b] [--distributed-only]"
                );
                std::process::exit(0);
            }
            other => panic!("unknown argument: {other}"),
        }
    }
    assert!(options.size >= 8, "--size must be at least 8");
    assert!(options.iterations > 0, "--iterations must be positive");
    options
}

fn scalar(re: f64, im: f64) -> S {
    S::from_parts(re, im)
}

fn from_rows(mut rows: Vec<Vec<(usize, S)>>, ncols: usize) -> CsrMatrix<S> {
    let mut row_ptr = Vec::with_capacity(rows.len() + 1);
    let mut col_idx = Vec::new();
    let mut values = Vec::new();
    row_ptr.push(0);
    for row in &mut rows {
        row.sort_unstable_by_key(|entry| entry.0);
        row.dedup_by(|right, left| {
            if right.0 == left.0 {
                left.1 = left.1 + right.1;
                true
            } else {
                false
            }
        });
        for &(column, value) in row.iter() {
            col_idx.push(column);
            values.push(value);
        }
        row_ptr.push(col_idx.len());
    }
    CsrMatrix::from_csr(rows.len(), ncols, row_ptr, col_idx, values)
}

fn diagonal(n: usize) -> CsrMatrix<S> {
    from_rows(
        (0..n).map(|row| vec![(row, scalar(2.0, 0.25))]).collect(),
        n,
    )
}

fn stencil_5(n_hint: usize) -> CsrMatrix<S> {
    let width = (n_hint as f64).sqrt().floor().max(2.0) as usize;
    let n = width * width;
    let mut rows = vec![Vec::new(); n];
    for i in 0..width {
        for j in 0..width {
            let row = i * width + j;
            rows[row].push((row, scalar(4.5, 0.1)));
            if i > 0 {
                rows[row].push((row - width, scalar(-1.0, 0.05)));
            }
            if i + 1 < width {
                rows[row].push((row + width, scalar(-1.0, -0.05)));
            }
            if j > 0 {
                rows[row].push((row - 1, scalar(-1.0, 0.03)));
            }
            if j + 1 < width {
                rows[row].push((row + 1, scalar(-1.0, -0.03)));
            }
        }
    }
    from_rows(rows, n)
}

fn stencil_7(n_hint: usize) -> CsrMatrix<S> {
    let width = (n_hint as f64).cbrt().floor().max(2.0) as usize;
    let plane = width * width;
    let n = plane * width;
    let mut rows = vec![Vec::new(); n];
    for z in 0..width {
        for y in 0..width {
            for x in 0..width {
                let row = z * plane + y * width + x;
                rows[row].push((row, scalar(6.5, 0.1)));
                if x > 0 {
                    rows[row].push((row - 1, scalar(-1.0, 0.01)));
                }
                if x + 1 < width {
                    rows[row].push((row + 1, scalar(-1.0, -0.01)));
                }
                if y > 0 {
                    rows[row].push((row - width, scalar(-1.0, 0.02)));
                }
                if y + 1 < width {
                    rows[row].push((row + width, scalar(-1.0, -0.02)));
                }
                if z > 0 {
                    rows[row].push((row - plane, scalar(-1.0, 0.03)));
                }
                if z + 1 < width {
                    rows[row].push((row + plane, scalar(-1.0, -0.03)));
                }
            }
        }
    }
    from_rows(rows, n)
}

fn irregular(n: usize) -> CsrMatrix<S> {
    let mut rows = vec![Vec::new(); n];
    for (row, entries) in rows.iter_mut().enumerate() {
        let degree = 1 + ((row.wrapping_mul(1_103_515_245).wrapping_add(12_345) >> 8) % 64);
        entries.push((row, scalar(8.0, 0.2)));
        for k in 0..degree.min(n) {
            let column = row
                .wrapping_mul(97)
                .wrapping_add(k.wrapping_mul(131))
                .wrapping_add(k * k + 17)
                % n;
            entries.push((
                column,
                scalar(-0.02 * (k + 1) as f64, (k % 5) as f64 * 0.003),
            ));
        }
    }
    from_rows(rows, n)
}

fn fixed_rows(n: usize, width: usize) -> CsrMatrix<S> {
    let mut rows = vec![Vec::new(); n];
    for (row, entries) in rows.iter_mut().enumerate() {
        entries.push((row, scalar(3.0, 0.1)));
        for k in 1..width.min(n) {
            entries.push((
                (row + k * 193) % n,
                scalar(-1.0 / width as f64, 0.002 * k as f64),
            ));
        }
    }
    from_rows(rows, n)
}

fn empty_rows(n: usize) -> CsrMatrix<S> {
    let rows = (0..n)
        .map(|row| {
            if row % 3 == 0 {
                Vec::new()
            } else {
                vec![(row, scalar(2.0, 0.2))]
            }
        })
        .collect();
    from_rows(rows, n)
}

fn structurally_symmetric(n: usize) -> CsrMatrix<S> {
    let mut rows = vec![Vec::new(); n];
    for row in 0..n {
        rows[row].push((row, scalar(3.0, 0.0)));
        if row + 1 < n {
            rows[row].push((row + 1, scalar(-1.0, 0.2)));
            rows[row + 1].push((row, scalar(-1.0, -0.2)));
        }
    }
    from_rows(rows, n)
}

fn nonsymmetric(n: usize) -> CsrMatrix<S> {
    let mut rows = vec![Vec::new(); n];
    for (row, entries) in rows.iter_mut().enumerate() {
        entries.push((row, scalar(2.0, 0.15)));
        if row + 1 < n {
            entries.push((row + 1, scalar(-1.25, 0.07)));
        }
        if row + 7 < n {
            entries.push((row + 7, scalar(0.3, -0.04)));
        }
    }
    from_rows(rows, n)
}

fn cases(n: usize) -> Vec<(&'static str, CsrMatrix<S>)> {
    vec![
        ("diagonal", diagonal(n)),
        ("stencil5", stencil_5(n)),
        ("stencil7", stencil_7(n)),
        ("irregular", irregular(n)),
        ("short_rows", fixed_rows(n, 3)),
        ("long_rows", fixed_rows(n, 96)),
        ("empty_rows", empty_rows(n)),
        ("structurally_symmetric", structurally_symmetric(n)),
        ("nonsymmetric", nonsymmetric(n)),
    ]
}

fn reference_spmv(a: &CsrMatrix<S>, x: &[S]) -> Vec<S> {
    let mut y = vec![S::zero(); a.nrows()];
    for (row, slot) in y.iter_mut().enumerate() {
        for entry in a.row_ptr()[row]..a.row_ptr()[row + 1] {
            *slot = *slot + a.values()[entry] * x[a.col_idx()[entry]];
        }
    }
    y
}

fn assert_close(label: &str, expected: &[S], actual: &[S]) {
    assert_eq!(expected.len(), actual.len());
    let max_error = expected
        .iter()
        .zip(actual)
        .map(|(&left, &right)| (left - right).abs())
        .fold(0.0f64, f64::max);
    assert!(max_error <= 1e-10, "{label}: max error {max_error:e}");
}

fn estimated_bytes(a: &CsrMatrix<S>) -> usize {
    a.nnz() * (size_of::<S>() * 2 + size_of::<usize>())
        + (a.nrows() + 1) * size_of::<usize>()
        + a.nrows() * size_of::<S>()
}

fn measure(mut apply: impl FnMut(), iterations: usize) -> (Duration, f64) {
    apply();
    ALLOCATIONS.store(0, Relaxed);
    let start = Instant::now();
    for _ in 0..iterations {
        apply();
        black_box(());
    }
    let elapsed = start.elapsed();
    let allocations = ALLOCATIONS.load(Relaxed) as f64 / iterations as f64;
    (elapsed / iterations as u32, allocations)
}

fn thread_partition_imbalance(a: &CsrMatrix<S>, threads: usize) -> f64 {
    let mut work = vec![0usize; threads];
    for (thread, slot) in work.iter_mut().enumerate() {
        let start = a.nrows() * thread / threads;
        let end = a.nrows() * (thread + 1) / threads;
        *slot = a.row_ptr()[end] - a.row_ptr()[start];
    }
    imbalance(work.iter().map(|&value| value as f64))
}

fn imbalance(values: impl Iterator<Item = f64>) -> f64 {
    let values: Vec<f64> = values.collect();
    if values.is_empty() {
        return 1.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    if mean == 0.0 {
        1.0
    } else {
        values.iter().copied().fold(0.0, f64::max) / mean
    }
}

fn serial_record(
    case: &str,
    implementation: &str,
    threads: usize,
    a: &CsrMatrix<S>,
    elapsed: Duration,
    allocations: f64,
    measurement_iterations: usize,
    strong_scaling_efficiency: Option<f64>,
) -> Value {
    let nanos = elapsed.as_nanos() as f64;
    json!({
        "schema_version": 1,
        "scope": "shared_memory",
        "case": case,
        "scalar": if cfg!(feature = "complex") { "complex64" } else { "f64" },
        "features": {
            "rayon": cfg!(feature = "rayon"),
            "mpi": cfg!(feature = "mpi"),
            "simd": cfg!(feature = "simd"),
            "backend_faer": cfg!(feature = "backend-faer"),
            "backend_sprs": cfg!(feature = "backend-sprs"),
        },
        "implementation": implementation,
        "threads": threads,
        "rows": a.nrows(),
        "cols": a.ncols(),
        "nnz": a.nnz(),
        "nanoseconds_per_spmv": nanos,
        "nanoseconds_per_nonzero": nanos / a.nnz().max(1) as f64,
        "effective_bandwidth_gb_s": estimated_bytes(a) as f64 / nanos,
        "allocations_per_spmv": allocations,
        "measurement_iterations": measurement_iterations,
        "local_compute_nanoseconds": nanos,
        "packing_nanoseconds": 0,
        "communication_nanoseconds": 0,
        "wait_nanoseconds": 0,
        "unpack_nanoseconds": 0,
        "strong_scaling_efficiency": strong_scaling_efficiency,
        "thread_partition_load_imbalance": thread_partition_imbalance(a, threads),
        "rank_load_imbalance": 1.0,
    })
}

fn run_shared_memory(case: &str, a: &CsrMatrix<S>, iterations: usize) {
    let x: Vec<S> = (0..a.ncols())
        .map(|index| {
            scalar(
                1.0 + (index % 17) as f64 * 0.01,
                (index % 11) as f64 * 0.005,
            )
        })
        .collect();
    let expected = reference_spmv(a, &x);
    let mut y = vec![S::zero(); a.nrows()];
    let (elapsed, allocations) = measure(
        || a.try_spmv(black_box(&x), black_box(&mut y)).unwrap(),
        iterations,
    );
    assert_close(case, &expected, &y);
    println!(
        "{}",
        serial_record(
            case,
            "canonical_serial",
            1,
            a,
            elapsed,
            allocations,
            iterations,
            Some(1.0)
        )
    );

    #[cfg(feature = "backend-faer")]
    {
        use kryst::matrix::csr::CsrMatrix as PlanCsrMatrix;
        use kryst::matrix::op::GenericCsrOp;
        use kryst::matrix::spmv::SpmvTuning;

        let plan_csr = PlanCsrMatrix::new(
            a.nrows(),
            a.ncols(),
            a.row_ptr().to_vec(),
            a.col_idx().to_vec(),
            a.values().to_vec(),
        );
        let op = GenericCsrOp::from_matrix(plan_csr, &SpmvTuning::default());
        y.fill(S::zero());
        let (elapsed, allocations) = measure(
            || op.try_matvec(black_box(&x), black_box(&mut y)).unwrap(),
            iterations,
        );
        assert_close(case, &expected, &y);
        let implementation = format!("planned_{:?}", op.plan().kernel).to_ascii_lowercase();
        println!(
            "{}",
            serial_record(
                case,
                &implementation,
                1,
                a,
                elapsed,
                allocations,
                iterations,
                Some(1.0),
            )
        );
    }

    #[cfg(feature = "rayon")]
    {
        let max_threads = std::thread::available_parallelism().map_or(1, usize::from);
        let mut thread_counts = vec![1, 2, 4, max_threads];
        thread_counts.retain(|&threads| threads <= max_threads);
        thread_counts.sort_unstable();
        thread_counts.dedup();
        let mut results = Vec::new();
        for threads in thread_counts {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            y.fill(S::zero());
            let (elapsed, allocations) = measure(
                || {
                    pool.install(|| a.try_spmv_parallel(black_box(&x), black_box(&mut y)))
                        .unwrap()
                },
                iterations,
            );
            assert_close(case, &expected, &y);
            results.push((threads, elapsed, allocations));
        }
        let one_thread_nanos = results
            .iter()
            .find(|(threads, _, _)| *threads == 1)
            .expect("one-thread Rayon baseline")
            .1
            .as_nanos() as f64;
        for (threads, elapsed, allocations) in results {
            let efficiency = one_thread_nanos / (threads as f64 * elapsed.as_nanos() as f64);
            println!(
                "{}",
                serial_record(
                    case,
                    "canonical_rayon",
                    threads,
                    a,
                    elapsed,
                    allocations,
                    iterations,
                    Some(efficiency),
                )
            );
        }
    }

    #[cfg(all(feature = "backend-sprs", not(feature = "complex")))]
    {
        let backend = sprs::CsMat::new(
            (a.nrows(), a.ncols()),
            a.row_ptr().to_vec(),
            a.col_idx().to_vec(),
            a.values().to_vec(),
        );
        y.fill(S::zero());
        let (elapsed, allocations) = measure(
            || {
                backend
                    .try_matvec(black_box(&x), black_box(&mut y))
                    .unwrap()
            },
            iterations,
        );
        assert_close(case, &expected, &y);
        println!(
            "{}",
            serial_record(
                case,
                "backend_sprs",
                1,
                a,
                elapsed,
                allocations,
                iterations,
                Some(1.0),
            )
        );
    }
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

fn partition(n: usize, size: usize) -> Vec<usize> {
    (0..=size).map(|rank| n * rank / size).collect()
}

fn slice_rows(a: &CsrMatrix<S>, start: usize, end: usize) -> CsrMatrix<S> {
    let entry_start = a.row_ptr()[start];
    let entry_end = a.row_ptr()[end];
    let row_ptr = a.row_ptr()[start..=end]
        .iter()
        .map(|&offset| offset - entry_start)
        .collect();
    CsrMatrix::from_csr(
        end - start,
        a.ncols(),
        row_ptr,
        a.col_idx()[entry_start..entry_end].to_vec(),
        a.values()[entry_start..entry_end].to_vec(),
    )
}

fn average_profile(total: DistSpmvProfile, iterations: usize) -> DistSpmvProfile {
    let divisor = iterations as u32;
    DistSpmvProfile {
        total: total.total / divisor,
        local_compute: total.local_compute / divisor,
        packing: total.packing / divisor,
        communication: total.communication / divisor,
        wait: total.wait / divisor,
        unpack: total.unpack / divisor,
    }
}

fn run_distributed(case: &str, global: &CsrMatrix<S>, iterations: usize, comm: &UniverseComm) {
    let parts = partition(global.nrows(), comm.size());
    let start = parts[comm.rank()];
    let end = parts[comm.rank() + 1];
    let local = slice_rows(global, start, end);
    let op = DistCsrOp::from_local_rows(global.nrows(), start, &local, &parts, comm.clone())
        .expect("build DistCsrOp baseline case");
    let x: Vec<S> = (start..end)
        .map(|index| {
            scalar(
                1.0 + (index % 17) as f64 * 0.01,
                (index % 11) as f64 * 0.005,
            )
        })
        .collect();
    let mut y = vec![S::zero(); local.nrows()];

    op.profile_matvec(&x, &mut y).unwrap();
    let global_x: Vec<S> = (0..global.ncols())
        .map(|index| {
            scalar(
                1.0 + (index % 17) as f64 * 0.01,
                (index % 11) as f64 * 0.005,
            )
        })
        .collect();
    let expected = reference_spmv(global, &global_x);
    assert_close(case, &expected[start..end], &y);
    comm.barrier();
    ALLOCATIONS.store(0, Relaxed);
    let mut sum = DistSpmvProfile::default();
    for _ in 0..iterations {
        let sample = op.profile_matvec(black_box(&x), black_box(&mut y)).unwrap();
        sum.total += sample.total;
        sum.local_compute += sample.local_compute;
        sum.packing += sample.packing;
        sum.communication += sample.communication;
        sum.wait += sample.wait;
        sum.unpack += sample.unpack;
    }
    let profile = average_profile(sum, iterations);
    let allocations = ALLOCATIONS.load(Relaxed) as f64 / iterations as f64;
    comm.barrier();

    let configured_threads = kryst::parallel::threads::current_rayon_threads();
    let local_values = [
        profile.total.as_nanos() as f64,
        profile.local_compute.as_nanos() as f64,
        profile.packing.as_nanos() as f64,
        profile.communication.as_nanos() as f64,
        profile.wait.as_nanos() as f64,
        profile.unpack.as_nanos() as f64,
        local.nnz() as f64,
        allocations,
        thread_partition_imbalance(&local, configured_threads),
    ];
    let mut gathered = Vec::new();
    comm.gather(&local_values, &mut gathered, 0);
    if comm.rank() != 0 {
        return;
    }

    let ranks: Vec<&[f64]> = gathered.chunks_exact(local_values.len()).collect();
    let maximum = |index: usize| ranks.iter().map(|rank| rank[index]).fold(0.0, f64::max);
    let mean =
        |index: usize| ranks.iter().map(|rank| rank[index]).sum::<f64>() / ranks.len() as f64;
    let wall_nanos = maximum(0);
    let nnz = global.nnz().max(1);
    let diagnostics = op.plan_diagnostics();
    println!(
        "{}",
        json!({
            "schema_version": 1,
            "scope": "distributed",
            "case": case,
            "scalar": if cfg!(feature = "complex") { "complex64" } else { "f64" },
            "features": {
                "rayon": cfg!(feature = "rayon"),
                "mpi": cfg!(feature = "mpi"),
                "simd": cfg!(feature = "simd"),
                "backend_faer": cfg!(feature = "backend-faer"),
                "backend_sprs": cfg!(feature = "backend-sprs"),
            },
            "implementation": "dist_csr",
            "ranks": comm.size(),
            "threads_per_rank": configured_threads,
            "rows": global.nrows(),
            "cols": global.ncols(),
            "nnz": global.nnz(),
            "nanoseconds_per_spmv": wall_nanos,
            "nanoseconds_per_nonzero": wall_nanos / nnz as f64,
            "effective_bandwidth_gb_s": estimated_bytes(global) as f64 / wall_nanos,
            "allocations_per_spmv_max_rank": maximum(7),
            "measurement_iterations": iterations,
            "local_compute_nanoseconds_max_rank": maximum(1),
            "local_compute_nanoseconds_mean_rank": mean(1),
            "packing_nanoseconds_max_rank": maximum(2),
            "communication_nanoseconds_max_rank": maximum(3),
            "wait_nanoseconds_max_rank": maximum(4),
            "unpack_nanoseconds_max_rank": maximum(5),
            "strong_scaling_efficiency": if comm.size() == 1 { Some(1.0) } else { None },
            "rank_load_imbalance": imbalance(ranks.iter().map(|rank| rank[6])),
            "rank_time_imbalance": imbalance(ranks.iter().map(|rank| rank[0])),
            "thread_partition_load_imbalance_max_rank": maximum(8),
            "halo_recv_volume": diagnostics.halo_recv_volume,
            "halo_send_volume": diagnostics.halo_send_volume,
            "overlap_mode": format!("{:?}", diagnostics.overlap_mode),
            "local_kernel_strategy": format!("{:?}", diagnostics.kernel_strategy),
            "local_spmv_kernel": diagnostics.local_spmv_kernel.map(|kernel| format!("{kernel:?}")),
        })
    );
}

fn main() {
    let options = parse_options();
    let comm = bench_comm();
    for (name, matrix) in cases(options.size) {
        if options
            .cases
            .as_ref()
            .is_some_and(|selected| !selected.contains(name))
        {
            continue;
        }
        if comm.size() == 1 && !options.distributed_only {
            run_shared_memory(name, &matrix, options.iterations);
        }
        run_distributed(name, &matrix, options.iterations, &comm);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_baseline_family_matches_reference_and_overwrites_output() {
        for (name, matrix) in cases(216) {
            let x: Vec<S> = (0..matrix.ncols())
                .map(|index| scalar(0.75 + index as f64 * 0.001, index as f64 * 0.0002))
                .collect();
            let expected = reference_spmv(&matrix, &x);
            let mut y = vec![scalar(99.0, -17.0); matrix.nrows()];
            matrix.try_spmv(&x, &mut y).unwrap();
            assert_close(name, &expected, &y);

            #[cfg(feature = "rayon")]
            {
                y.fill(scalar(-31.0, 4.0));
                matrix.try_spmv_parallel(&x, &mut y).unwrap();
                assert_close(name, &expected, &y);
            }
        }
    }

    #[test]
    fn profiled_single_rank_distcsr_matches_serial() {
        let matrix = stencil_5(256);
        let x: Vec<S> = (0..matrix.ncols())
            .map(|index| scalar(1.0 + index as f64 * 0.001, index as f64 * 0.0001))
            .collect();
        let expected = reference_spmv(&matrix, &x);
        let comm = UniverseComm::NoComm(NoComm);
        let op = DistCsrOp::from_local_rows(matrix.nrows(), 0, &matrix, &[0, matrix.nrows()], comm)
            .unwrap();
        let mut y = vec![scalar(12.0, 3.0); matrix.nrows()];
        let profile = op.profile_matvec(&x, &mut y).unwrap();
        assert!(profile.total >= profile.local_compute);
        assert_close("single-rank DistCsrOp", &expected, &y);
    }
}
