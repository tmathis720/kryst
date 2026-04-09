use kryst::context::ksp_context::Workspace;
use kryst::matrix::utils::{poisson, poisson_3d};
use kryst::parallel::{NoComm, UniverseComm};
use kryst::preconditioner::PcSide;
use kryst::solver::LinearSolver;
use kryst::solver::bicgstab::{BiCgStabSolver, BiCgStabVariant};
use kryst::solver::fgmres::{FgmresSolver, FgmresVariant};
use kryst::solver::gmres::{GmresSolver, GmresVariant};
use kryst::solver::idrs::IdrsBuilder;
use kryst::solver::pcg::{PCG_PIPELINED_DEFAULT_REPLACE_EVERY, PcgSolver, PcgVariant};
use kryst::solver::qmr::QmrSolver;
use std::time::Instant;

fn report(name: &str, iters: usize, reductions: usize, elapsed_ms: f64, model: Option<&str>) {
    println!(
        "    {name:20} | iters={iters:5} | reductions={reductions:6} | {:8.3} ms | model={}",
        elapsed_ms,
        model.unwrap_or("n/a")
    );
}

fn strong_weak_cases() -> Vec<(&'static str, usize)> {
    vec![
        ("strong", 1),
        ("strong", 4),
        ("strong", 16),
        ("weak", 1),
        ("weak", 4),
        ("weak", 16),
    ]
}

fn main() {
    let comm = UniverseComm::NoComm(NoComm);
    println!("== solver variant scaling microbench ==");
    for (mode, rank_factor) in strong_weak_cases() {
        let (n2, a3) = if mode == "strong" {
            (poisson::poisson_5pt_2d(220), poisson_3d(40, 40, 20))
        } else {
            let g = 120 * rank_factor;
            (
                poisson::poisson_5pt_2d(g),
                poisson_3d(24, 24, 12 * rank_factor),
            )
        };

        println!("-- {mode} scale, virtual_ranks={rank_factor} --");

        let b2 = vec![1.0; n2.nrows()];
        let mut x2 = vec![0.0; n2.nrows()];

        let mut pcg_classic = PcgSolver::new(1e-8, 2_000);
        pcg_classic.set_variant(PcgVariant::Classic);
        let mut ws = Workspace::default();
        let t0 = Instant::now();
        let s = pcg_classic
            .solve(
                &n2,
                None,
                &b2,
                &mut x2,
                PcSide::Left,
                &comm,
                None,
                Some(&mut ws),
            )
            .expect("pcg classic");
        report(
            "pcg-classic",
            s.iterations,
            s.counters.num_global_reductions,
            t0.elapsed().as_secs_f64() * 1e3,
            s.reduction_model.as_ref().map(|m| m.variant),
        );

        x2.fill(0.0);
        let mut pcg_pipe = PcgSolver::new(1e-8, 2_000);
        pcg_pipe.set_variant(PcgVariant::Pipelined {
            replace_every: PCG_PIPELINED_DEFAULT_REPLACE_EVERY,
        });
        let mut ws = Workspace::default();
        let t0 = Instant::now();
        let s = pcg_pipe
            .solve(
                &n2,
                None,
                &b2,
                &mut x2,
                PcSide::Left,
                &comm,
                None,
                Some(&mut ws),
            )
            .expect("pcg pipelined");
        report(
            "pcg-pipelined",
            s.iterations,
            s.counters.num_global_reductions,
            t0.elapsed().as_secs_f64() * 1e3,
            s.reduction_model.as_ref().map(|m| m.variant),
        );

        assert!(
            s.counters.overlap_global_reductions > 0,
            "benchmark gate failed: pcg pipelined should record overlap reductions"
        );

        let b3 = vec![1.0; a3.nrows()];
        let mut x3 = vec![0.0; a3.nrows()];
        let mut gmres = GmresSolver::new(30, 1e-8, 600);
        gmres.set_variant(GmresVariant::Classical);
        let mut ws = Workspace::default();
        let t0 = Instant::now();
        let s = gmres
            .solve_f64(
                &a3,
                None,
                &b3,
                &mut x3,
                PcSide::Left,
                &comm,
                None,
                Some(&mut ws),
            )
            .expect("gmres classic");
        report(
            "gmres-classical",
            s.iterations,
            s.counters.num_global_reductions,
            t0.elapsed().as_secs_f64() * 1e3,
            s.reduction_model.as_ref().map(|m| m.variant),
        );

        x3.fill(0.0);
        gmres.set_variant(GmresVariant::Pipelined);
        let mut ws = Workspace::default();
        let t0 = Instant::now();
        let s = gmres
            .solve_f64(
                &a3,
                None,
                &b3,
                &mut x3,
                PcSide::Left,
                &comm,
                None,
                Some(&mut ws),
            )
            .expect("gmres pipelined");
        report(
            "gmres-pipelined",
            s.iterations,
            s.counters.num_global_reductions,
            t0.elapsed().as_secs_f64() * 1e3,
            s.reduction_model.as_ref().map(|m| m.variant),
        );

        x3.fill(0.0);
        let mut fgm = FgmresSolver::new(1e-8, 600, 30);
        fgm.set_variant(FgmresVariant::Pipelined);
        let mut ws = Workspace::default();
        let t0 = Instant::now();
        let s = fgm
            .solve_f64(
                &a3,
                None,
                &b3,
                &mut x3,
                PcSide::Left,
                &comm,
                None,
                Some(&mut ws),
            )
            .expect("fgmres pipelined");
        report(
            "fgmres-pipelined",
            s.iterations,
            s.counters.num_global_reductions,
            t0.elapsed().as_secs_f64() * 1e3,
            s.reduction_model.as_ref().map(|m| m.variant),
        );

        // Nonsymmetric latency-sensitive family: BiCGStab variants.
        let mut xb = vec![0.0; a3.nrows()];
        let mut bicg_classic = BiCgStabSolver::new(1e-8, 600);
        bicg_classic.set_variant(BiCgStabVariant::Classic);
        let mut ws = Workspace::default();
        let t0 = Instant::now();
        let bicg_classic_start = Instant::now();
        let s_bicg_classic = bicg_classic
            .solve_f64(
                &a3,
                None,
                &b3,
                &mut xb,
                PcSide::Left,
                &comm,
                None,
                Some(&mut ws),
            )
            .expect("bicgstab classic");
        report(
            "bicgstab-classic",
            s_bicg_classic.iterations,
            s_bicg_classic.counters.num_global_reductions,
            t0.elapsed().as_secs_f64() * 1e3,
            s_bicg_classic.reduction_model.as_ref().map(|m| m.variant),
        );
        let bicg_classic_ms = bicg_classic_start.elapsed().as_secs_f64() * 1e3;

        xb.fill(0.0);
        let mut bicg_fewerchecks = BiCgStabSolver::new(1e-8, 600);
        bicg_fewerchecks.set_variant(BiCgStabVariant::FewerChecks);
        let mut ws = Workspace::default();
        let t0 = Instant::now();
        let bicg_fewerchecks_start = Instant::now();
        let s_bicg_fewerchecks = bicg_fewerchecks
            .solve_f64(
                &a3,
                None,
                &b3,
                &mut xb,
                PcSide::Left,
                &comm,
                None,
                Some(&mut ws),
            )
            .expect("bicgstab fewerchecks");
        report(
            "bicgstab-fewer-checks",
            s_bicg_fewerchecks.iterations,
            s_bicg_fewerchecks.counters.num_global_reductions,
            t0.elapsed().as_secs_f64() * 1e3,
            s_bicg_fewerchecks
                .reduction_model
                .as_ref()
                .map(|m| m.variant),
        );
        let bicg_fewerchecks_ms = bicg_fewerchecks_start.elapsed().as_secs_f64() * 1e3;

        let mut xq = vec![0.0; a3.nrows()];
        let mut qmr = QmrSolver::new(1e-8, 600);
        let mut ws = Workspace::default();
        let t0 = Instant::now();
        let s_qmr = qmr
            .solve_f64(
                &a3,
                None,
                &b3,
                &mut xq,
                PcSide::Left,
                &comm,
                None,
                Some(&mut ws),
            )
            .expect("qmr");
        report(
            "qmr",
            s_qmr.iterations,
            s_qmr.counters.num_global_reductions,
            t0.elapsed().as_secs_f64() * 1e3,
            s_qmr.reduction_model.as_ref().map(|m| m.variant),
        );

        let mut xi = vec![0.0; a3.nrows()];
        let mut idrs = IdrsBuilder::new().s(4).tol(1e-8).maxit(600).build();
        let t0 = Instant::now();
        let s_idrs = idrs
            .solve_f64(
                &a3,
                None,
                &b3,
                &mut xi,
                PcSide::Left,
                &comm,
                None,
                Some(&ws),
            )
            .expect("idrs");
        report(
            "idrs",
            s_idrs.iterations,
            s_idrs.counters.num_global_reductions,
            t0.elapsed().as_secs_f64() * 1e3,
            s_idrs.reduction_model.as_ref().map(|m| m.variant),
        );

        println!(
            "      deltas: bicg-fewerchecks vs classic => reductions={} runtime_ms={:.3}",
            s_bicg_classic.counters.num_global_reductions as isize
                - s_bicg_fewerchecks.counters.num_global_reductions as isize,
            bicg_classic_ms - bicg_fewerchecks_ms
        );

        assert!(
            s_bicg_fewerchecks.counters.overlap_global_reductions > 0,
            "benchmark gate failed: bicgstab fewerchecks should record overlap reductions"
        );
        assert!(
            s_bicg_fewerchecks.counters.num_global_reductions
                <= s_bicg_classic.counters.num_global_reductions,
            "benchmark gate failed: fewerchecks did not reduce synchronization"
        );
    }
}
