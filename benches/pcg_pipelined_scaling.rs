use kryst::config::options::CgVariant;
use kryst::context::ksp_context::Workspace;
use kryst::matrix::utils::poisson_3d;
use kryst::parallel::{NoComm, UniverseComm};
use kryst::preconditioner::PcSide;
use kryst::solver::pcg::{PCG_PIPELINED_DEFAULT_REPLACE_EVERY, PcgSolver, PcgVariant};
use kryst::solver::{CgSolver, LinearSolver};
use std::time::Instant;

fn run_pcg_variant(
    a: &dyn kryst::matrix::op::LinOp<S = f64>,
    b: &[f64],
    variant: PcgVariant,
) -> (
    std::time::Duration,
    kryst::utils::convergence::SolveStats<f64>,
) {
    let mut solver = PcgSolver::new(1e-8, 2000);
    solver.set_variant(variant);
    let mut wk = Workspace::default();
    solver.setup_workspace(&mut wk);
    let mut x = vec![0.0; b.len()];
    let comm = UniverseComm::NoComm(NoComm);
    let start = Instant::now();
    let stats = solver
        .solve(a, None, b, &mut x, PcSide::Left, &comm, None, Some(&mut wk))
        .expect("PCG converged");
    (start.elapsed(), stats)
}

fn run_cg_variant(
    a: &dyn kryst::matrix::op::LinOp<S = f64>,
    b: &[f64],
    variant: CgVariant,
) -> (
    std::time::Duration,
    kryst::utils::convergence::SolveStats<f64>,
) {
    let mut solver = CgSolver::new(1e-8, 2000);
    solver.set_variant(variant);
    if matches!(variant, CgVariant::Pipelined) {
        solver.set_pipelined_residual_refresh_every(Some(PCG_PIPELINED_DEFAULT_REPLACE_EVERY));
    }
    let mut wk = Workspace::default();
    solver.setup_workspace(&mut wk);
    let mut x = vec![0.0; b.len()];
    let comm = UniverseComm::NoComm(NoComm);
    let start = Instant::now();
    let stats = LinearSolver::solve(
        &mut solver,
        a,
        None,
        b,
        &mut x,
        PcSide::Left,
        &comm,
        None,
        Some(&mut wk),
    )
    .expect("CG converged");
    (start.elapsed(), stats)
}

fn weak_scaling_case(ranks: usize) -> (usize, usize, usize) {
    const BASE: usize = 24;
    (BASE, BASE, BASE * ranks)
}

fn strong_scaling_dims() -> (usize, usize, usize) {
    (64, 64, 64)
}

fn solve_case(label: &str, ranks: usize, dims: (usize, usize, usize)) {
    let (nx, ny, nz) = dims;
    let a = poisson_3d(nx, ny, nz);
    let n = a.nrows();
    let b = vec![1.0; n];

    println!("{label:>12} | ranks = {ranks:>3} | dims = {nx}×{ny}×{nz} | dofs = {n}");

    let (cg_time, cg_stats) = run_cg_variant(&a, &b, CgVariant::Classic);
    let (classic_time, classic_stats) = run_pcg_variant(&a, &b, PcgVariant::Classic);
    let (pipelined_time, pipelined_stats) = run_pcg_variant(
        &a,
        &b,
        PcgVariant::Pipelined {
            replace_every: PCG_PIPELINED_DEFAULT_REPLACE_EVERY,
        },
    );

    let iter_gap = (classic_stats.iterations as isize - pipelined_stats.iterations as isize).abs();
    assert!(
        iter_gap <= 1,
        "iteration counts diverged: classic={} pipelined={}",
        classic_stats.iterations,
        pipelined_stats.iterations
    );

    assert!(
        pipelined_stats.counters.overlap_global_reductions > 0,
        "pipelined benchmark expected overlap-aware reductions"
    );
    let speedup = classic_time.as_secs_f64() / pipelined_time.as_secs_f64();
    println!(
        "    cg       : {:>6} iters | {:>8.3} ms",
        cg_stats.iterations,
        cg_time.as_secs_f64() * 1.0e3
    );
    println!(
        "    classic  : {:>6} iters | {:>8.3} ms",
        classic_stats.iterations,
        classic_time.as_secs_f64() * 1.0e3
    );
    println!(
        "    pipelined: {:>6} iters | {:>8.3} ms | speedup ×{speedup:.3}",
        pipelined_stats.iterations,
        pipelined_time.as_secs_f64() * 1.0e3
    );
    println!(
        "    delta: reductions={} runtime_ms={:.3}",
        classic_stats.counters.num_global_reductions as isize
            - pipelined_stats.counters.num_global_reductions as isize,
        (classic_time.as_secs_f64() - pipelined_time.as_secs_f64()) * 1.0e3
    );
    println!();
}

fn main() {
    #[cfg(feature = "logging")]
    let _ = env_logger::builder().is_test(true).try_init();

    let strong_dims = strong_scaling_dims();
    println!(
        "== Strong scaling (fixed {}×{}×{} grid) ==",
        strong_dims.0, strong_dims.1, strong_dims.2
    );
    for &ranks in &[1usize, 4, 16, 64] {
        solve_case("strong", ranks, strong_dims);
    }

    println!("== Weak scaling (24³ unknowns per virtual rank) ==");
    for &ranks in &[1usize, 4, 16, 64] {
        solve_case("weak", ranks, weak_scaling_case(ranks));
    }
}
