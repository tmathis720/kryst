use kryst::Comm;
use kryst::config::options::KspOptions;
use kryst::context::ksp_context::{KspContext, SolverType};
use kryst::matrix::utils::poisson_3d;
use kryst::matrix::{CsrMatrix, DistCsrOp};
use kryst::parallel::{NoComm, UniverseComm};
use kryst::preconditioner::PcSide;
use rand::{Rng, SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

#[cfg(feature = "mpi")]
use kryst::parallel::MpiComm;

#[derive(Debug, Deserialize)]
struct FixturesManifest {
    schema_version: u32,
    cases: Vec<BenchCase>,
}

#[derive(Debug, Deserialize)]
struct BenchCase {
    id: String,
    matrix: MatrixSpec,
    process_count: usize,
    partition_seed: u64,
    solver: SolverSpec,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum MatrixSpec {
    Poisson2d {
        n: usize,
    },
    Poisson3d {
        nx: usize,
        ny: usize,
        nz: usize,
    },
    PowerLaw {
        n: usize,
        avg_degree: usize,
        seed: u64,
    },
    BlockSystem {
        n_blocks: usize,
        block_size: usize,
        overlap: usize,
    },
}

#[derive(Debug, Deserialize)]
struct SolverSpec {
    ksp_type: String,
    pc_global: String,
    pc_local: String,
    pc_dist_local_apply: String,
    rtol: f64,
    maxits: usize,
    restart: usize,
}

#[derive(Debug, Deserialize)]
struct ExpectationsManifest {
    schema_version: u32,
    expectations: Vec<CaseExpectation>,
}

#[derive(Debug, Deserialize)]
struct CaseExpectation {
    id: String,
    iterations: BandUsize,
    final_residual: BandF64,
    route: RouteExpectation,
}

#[derive(Debug, Deserialize)]
struct BandUsize {
    min: usize,
    max: usize,
}

#[derive(Debug, Deserialize)]
struct BandF64 {
    min: f64,
    max: f64,
}

#[derive(Debug, Deserialize)]
struct RouteExpectation {
    selected: String,
    fallback_max: usize,
}

#[derive(Debug, Serialize)]
struct Artifact {
    schema_version: u32,
    cases: Vec<CaseArtifact>,
}

#[derive(Debug, Serialize)]
struct CaseArtifact {
    id: String,
    process_count: usize,
    status: String,
    details: BTreeMap<String, Value>,
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

#[derive(Clone, Copy, Debug)]
enum TimingDetail {
    Off,
    Basic,
    High,
}

impl TimingDetail {
    fn as_str(self) -> &'static str {
        match self {
            TimingDetail::Off => "off",
            TimingDetail::Basic => "basic",
            TimingDetail::High => "high",
        }
    }
}

fn parse_args() -> (PathBuf, PathBuf, PathBuf, TimingDetail) {
    let mut fixtures = PathBuf::from("benchmarks/distcsr/fixtures.json");
    let mut expectations = PathBuf::from("benchmarks/distcsr/expectations.json");
    let mut output = PathBuf::from("benchmarks/distcsr/artifacts/latest.json");
    let mut timing_detail = TimingDetail::Basic;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--fixtures" => fixtures = PathBuf::from(args.next().expect("missing fixtures path")),
            "--expectations" => {
                expectations = PathBuf::from(args.next().expect("missing expectations path"))
            }
            "--output" => output = PathBuf::from(args.next().expect("missing output path")),
            "--timing-detail" => {
                let raw = args.next().expect("missing timing detail value");
                timing_detail = match raw.as_str() {
                    "off" => TimingDetail::Off,
                    "basic" => TimingDetail::Basic,
                    "high" => TimingDetail::High,
                    _ => panic!("unsupported timing detail: {raw}"),
                };
            }
            other => panic!("unsupported argument: {other}"),
        }
    }
    (fixtures, expectations, output, timing_detail)
}

#[cfg(feature = "metrics")]
fn solve_metrics_nanos(stats: &kryst::utils::convergence::SolveStats<f64>) -> (u64, u64, u64) {
    (
        stats.metrics.matvec_nanos,
        stats.metrics.pc_apply_nanos,
        stats.metrics.reduction_wait_nanos,
    )
}

#[cfg(not(feature = "metrics"))]
fn solve_metrics_nanos(_stats: &kryst::utils::convergence::SolveStats<f64>) -> (u64, u64, u64) {
    (0, 0, 0)
}

fn build_part_prefix(n_global: usize, size: usize, seed: u64) -> Vec<usize> {
    let mut rng = StdRng::seed_from_u64(seed);
    let base = n_global / size;
    let rem = n_global % size;
    let mut chunks: Vec<usize> = (0..size).map(|r| base + usize::from(r < rem)).collect();
    for _ in 0..(size * 3).max(1) {
        let i = rng.gen_range(0..size);
        let j = rng.gen_range(0..size);
        if i != j && chunks[i] > 1 {
            chunks[i] -= 1;
            chunks[j] += 1;
        }
    }
    let mut prefix = vec![0usize];
    for c in chunks {
        prefix.push(prefix.last().copied().unwrap_or(0) + c);
    }
    if let Some(last) = prefix.last_mut() {
        *last = n_global;
    }
    prefix
}

fn slice_rows(a: &CsrMatrix<f64>, row_start: usize, n_local: usize) -> CsrMatrix<f64> {
    let row_end = row_start + n_local;
    let mut row_ptr = vec![0usize];
    let mut col_idx = Vec::new();
    let mut values = Vec::new();
    for row in row_start..row_end {
        let (cols, vals) = a.row(row);
        col_idx.extend_from_slice(cols);
        values.extend_from_slice(vals);
        row_ptr.push(col_idx.len());
    }
    CsrMatrix::from_csr(n_local, a.ncols(), row_ptr, col_idx, values)
}

fn poisson2d_csr(n: usize) -> CsrMatrix<f64> {
    let nn = n * n;
    let mut row_ptr = vec![0usize];
    let mut col_idx = Vec::with_capacity(5 * nn);
    let mut vals = Vec::with_capacity(5 * nn);
    for i in 0..n {
        for j in 0..n {
            let row = i * n + j;
            let mut entries: [(usize, f64); 5] = [(usize::MAX, 0.0); 5];
            let mut len = 0usize;
            let mut diag = 0.0;
            if j > 0 {
                entries[len] = (row - 1, -1.0);
                len += 1;
                diag += 1.0;
            }
            if j + 1 < n {
                entries[len] = (row + 1, -1.0);
                len += 1;
                diag += 1.0;
            }
            if i > 0 {
                entries[len] = (row - n, -1.0);
                len += 1;
                diag += 1.0;
            }
            if i + 1 < n {
                entries[len] = (row + n, -1.0);
                len += 1;
                diag += 1.0;
            }
            entries[len] = (row, diag);
            len += 1;
            entries[..len].sort_unstable_by_key(|(c, _)| *c);
            for k in 0..len {
                col_idx.push(entries[k].0);
                vals.push(entries[k].1);
            }
            row_ptr.push(col_idx.len());
        }
    }
    CsrMatrix::from_csr(nn, nn, row_ptr, col_idx, vals)
}

fn powerlaw_like(n: usize, avg_degree: usize, seed: u64) -> CsrMatrix<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut row_ptr = vec![0usize];
    let mut col_idx = Vec::new();
    let mut vals = Vec::new();
    for i in 0..n {
        let base = rng.gen_range((avg_degree / 2).max(1)..=(avg_degree * 3 / 2).max(2));
        let burst = if rng.r#gen::<f64>() < 0.05 {
            rng.gen_range(avg_degree.max(1)..=(4 * avg_degree.max(1)))
        } else {
            0
        };
        let deg = (base + burst).min(n.saturating_sub(1)).max(1);
        let mut set: BTreeSet<usize> = BTreeSet::new();
        set.insert(i);
        while set.len() < deg {
            set.insert(rng.gen_range(0..n));
        }
        for &j in &set {
            col_idx.push(j);
            let mut v = 0.5 + rng.r#gen::<f64>();
            if j != i && rng.r#gen::<f64>() < 0.2 {
                v = -v;
            }
            vals.push(v);
        }
        row_ptr.push(col_idx.len());
    }
    CsrMatrix::from_csr(n, n, row_ptr, col_idx, vals)
}

fn block_system(n_blocks: usize, block_size: usize, overlap: usize) -> CsrMatrix<f64> {
    let n = n_blocks * block_size;
    let mut row_ptr = vec![0usize];
    let mut col_idx = Vec::new();
    let mut vals = Vec::new();
    for b in 0..n_blocks {
        let start = b * block_size;
        let end = start + block_size;
        for i in start..end {
            if i > start {
                col_idx.push(i - 1);
                vals.push(-1.0);
            }
            col_idx.push(i);
            vals.push(2.0);
            if i + 1 < end {
                col_idx.push(i + 1);
                vals.push(-1.0);
            }
            if overlap > 0 && b + 1 < n_blocks {
                let next_start = (b + 1) * block_size;
                for k in 0..overlap.min(block_size) {
                    col_idx.push(next_start + k);
                    vals.push(-0.15);
                }
            }
            row_ptr.push(col_idx.len());
        }
    }
    CsrMatrix::from_csr(n, n, row_ptr, col_idx, vals)
}

fn matrix_from_spec(spec: &MatrixSpec) -> CsrMatrix<f64> {
    match spec {
        MatrixSpec::Poisson2d { n } => poisson2d_csr(*n),
        MatrixSpec::Poisson3d { nx, ny, nz } => poisson_3d(*nx, *ny, *nz),
        MatrixSpec::PowerLaw {
            n,
            avg_degree,
            seed,
        } => powerlaw_like(*n, *avg_degree, *seed),
        MatrixSpec::BlockSystem {
            n_blocks,
            block_size,
            overlap,
        } => block_system(*n_blocks, *block_size, *overlap),
    }
}

fn fallback_total(view: &BTreeMap<String, Value>) -> usize {
    view.get("pc_dist_fallback_counters")
        .and_then(Value::as_object)
        .map(|o| o.values().filter_map(Value::as_u64).sum::<u64>() as usize)
        .unwrap_or(0)
}

fn main() {
    let (fixtures_path, expectations_path, output_path, timing_detail) = parse_args();
    let fixtures: FixturesManifest =
        serde_json::from_str(&fs::read_to_string(fixtures_path).expect("read fixtures"))
            .expect("parse fixtures");
    let expectations: ExpectationsManifest =
        serde_json::from_str(&fs::read_to_string(expectations_path).expect("read expectations"))
            .expect("parse expectations");
    assert_eq!(fixtures.schema_version, 1);
    assert_eq!(expectations.schema_version, 1);
    let expected_by_id: BTreeMap<String, CaseExpectation> = expectations
        .expectations
        .into_iter()
        .map(|e| (e.id.clone(), e))
        .collect();
    let comm = bench_comm();
    let mut artifact = Artifact {
        schema_version: 1,
        cases: Vec::new(),
    };

    for case in fixtures.cases {
        let mut details = BTreeMap::new();
        details.insert(
            "required_process_count".into(),
            Value::from(case.process_count as u64),
        );
        details.insert(
            "actual_process_count".into(),
            Value::from(comm.size() as u64),
        );

        if case.process_count != comm.size() {
            details.insert(
                "skip_reason".into(),
                Value::from("process_count_mismatch_for_replay"),
            );
            artifact.cases.push(CaseArtifact {
                id: case.id,
                process_count: comm.size(),
                status: "skipped".into(),
                details,
            });
            continue;
        }

        let expectation = expected_by_id
            .get(&case.id)
            .expect("missing case expectation");
        let a_global = matrix_from_spec(&case.matrix);
        let n_global = a_global.nrows();
        let part_prefix = build_part_prefix(n_global, comm.size(), case.partition_seed);
        let row_start = part_prefix[comm.rank()];
        let n_local = part_prefix[comm.rank() + 1] - row_start;
        let a_local = slice_rows(&a_global, row_start, n_local);
        let dist =
            DistCsrOp::from_local_rows(n_global, row_start, &a_local, &part_prefix, comm.clone())
                .expect("dist csr build");

        let opts = KspOptions::from_args(&[
            "-ksp_type",
            &case.solver.ksp_type,
            "-pc_global",
            &case.solver.pc_global,
            "-pc_local",
            &case.solver.pc_local,
            "-pc_dist_local_apply",
            &case.solver.pc_dist_local_apply,
            "-ksp_rtol",
            &case.solver.rtol.to_string(),
            "-ksp_maxits",
            &case.solver.maxits.to_string(),
        ])
        .expect("ksp options parse");

        let mut ksp = KspContext::new();
        ksp.set_type(SolverType::Gmres).expect("set gmres");
        ksp.set_pc_type_from_str("block_jacobi")
            .expect("set block_jacobi");
        ksp.set_pc_side(PcSide::Left);
        ksp.set_restart(case.solver.restart);
        ksp.set_from_options(&opts).expect("set options");
        ksp.set_operators(Arc::new(dist), None);

        let b = vec![1.0; n_local];
        let mut x = vec![0.0; n_local];
        let solve_start = Instant::now();
        let stats = ksp.solve(&b, &mut x).expect("solve");
        let solve_wall_nanos = solve_start.elapsed().as_nanos() as u64;
        let view = ksp.view();

        let route_policy = view
            .solver_config
            .get("pc_dist_route_policy")
            .and_then(Value::as_str)
            .unwrap_or("unknown")
            .to_string();
        let selected_route = view
            .solver_config
            .get("pc_dist_selected_route")
            .and_then(Value::as_str)
            .unwrap_or("unknown")
            .to_string();
        let fallback_chain = view
            .solver_config
            .get("pc_dist_fallback_chain")
            .cloned()
            .unwrap_or_else(|| Value::Array(Vec::new()));
        let fallback_reason = view
            .solver_config
            .get("pc_dist_fallback_reason")
            .cloned()
            .unwrap_or(Value::Null);
        let fallback_counters = view
            .solver_config
            .get("pc_dist_fallback_counters")
            .cloned()
            .unwrap_or_else(|| Value::Object(serde_json::Map::new()));
        let fallback_count = fallback_total(&view.solver_config);
        let residual = stats.final_residual;
        let pass = (expectation.iterations.min..=expectation.iterations.max)
            .contains(&stats.iterations)
            && residual >= expectation.final_residual.min
            && residual <= expectation.final_residual.max
            && selected_route == expectation.route.selected
            && fallback_count <= expectation.route.fallback_max;

        details.insert("iterations".into(), Value::from(stats.iterations as u64));
        details.insert("final_residual".into(), Value::from(residual));
        details.insert("reason".into(), Value::from(format!("{:?}", stats.reason)));
        details.insert(
            "num_global_reductions".into(),
            Value::from(stats.counters.num_global_reductions as u64),
        );

        if !matches!(timing_detail, TimingDetail::Off) {
            let (matvec_nanos, pc_apply_nanos, reduction_nanos) = solve_metrics_nanos(&stats);
            let halo_nanos = if matches!(timing_detail, TimingDetail::High) {
                // Reserved for high-detail split timings; left zero unless explicit
                // matvec-internal profiling is enabled.
                0u64
            } else {
                0u64
            };
            let known = matvec_nanos
                .saturating_add(halo_nanos)
                .saturating_add(pc_apply_nanos)
                .saturating_add(reduction_nanos);
            let other_nanos = solve_wall_nanos.saturating_sub(known);
            let iters = stats.iterations.max(1) as f64;

            let mut totals = serde_json::Map::new();
            totals.insert(
                "solve_wall".into(),
                Value::from(solve_wall_nanos as f64 * 1e-9),
            );
            totals.insert("matvec".into(), Value::from(matvec_nanos as f64 * 1e-9));
            totals.insert("halo".into(), Value::from(halo_nanos as f64 * 1e-9));
            totals.insert("pc_apply".into(), Value::from(pc_apply_nanos as f64 * 1e-9));
            totals.insert(
                "global_reduction".into(),
                Value::from(reduction_nanos as f64 * 1e-9),
            );
            totals.insert("other".into(), Value::from(other_nanos as f64 * 1e-9));

            let mut per_iter = serde_json::Map::new();
            per_iter.insert(
                "matvec".into(),
                Value::from(matvec_nanos as f64 * 1e-9 / iters),
            );
            per_iter.insert("halo".into(), Value::from(halo_nanos as f64 * 1e-9 / iters));
            per_iter.insert(
                "pc_apply".into(),
                Value::from(pc_apply_nanos as f64 * 1e-9 / iters),
            );
            per_iter.insert(
                "global_reduction".into(),
                Value::from(reduction_nanos as f64 * 1e-9 / iters),
            );
            per_iter.insert(
                "other".into(),
                Value::from(other_nanos as f64 * 1e-9 / iters),
            );

            details.insert("timing_detail".into(), Value::from(timing_detail.as_str()));
            details.insert("timing_totals_sec".into(), Value::Object(totals));
            details.insert("timing_per_iter_avg_sec".into(), Value::Object(per_iter));
            #[cfg(not(feature = "metrics"))]
            details.insert(
                "timing_note".into(),
                Value::from(
                    "build without `metrics`: matvec/pc_apply/global_reduction are zero-filled to keep CI overhead bounded",
                ),
            );
        }

        details.insert("pc_dist_route_policy".into(), Value::from(route_policy));
        details.insert("pc_dist_selected_route".into(), Value::from(selected_route));
        details.insert("pc_dist_fallback_chain".into(), fallback_chain);
        details.insert("pc_dist_fallback_reason".into(), fallback_reason);
        details.insert("pc_dist_fallback_counters".into(), fallback_counters);
        details.insert("fallback_total".into(), Value::from(fallback_count as u64));

        artifact.cases.push(CaseArtifact {
            id: case.id,
            process_count: comm.size(),
            status: if pass { "pass" } else { "fail" }.into(),
            details,
        });
    }

    artifact.cases.sort_by(|a, b| a.id.cmp(&b.id));
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).expect("create output dir");
    }
    fs::write(
        output_path,
        serde_json::to_string_pretty(&artifact).expect("serialize artifact"),
    )
    .expect("write artifact");
}
