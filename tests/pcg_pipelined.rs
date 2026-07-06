#![cfg(all(feature = "backend-faer", not(feature = "complex")))]
use faer::Mat;
use fixtures::csr_poisson_1d;
use kryst::algebra::prelude::*;
use kryst::context::ksp_context::Workspace;
use kryst::error::KError;
use kryst::matrix::op::LinOp;
#[cfg(feature = "rayon")]
use kryst::parallel::RayonComm;
use kryst::parallel::{Comm, NoComm, UniverseComm};
use kryst::preconditioner::PcSide;
use kryst::preconditioner::Preconditioner;
use kryst::preconditioner::jacobi::Jacobi;
use kryst::solver::pcg::{PcgSolver, PcgVariant};
use kryst::solver::{LinearSolver, MonitorAction};
use kryst::utils::reduction::{install_test_counter, take_test_counter};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

fn build_dense_poisson(n: usize) -> Mat<R> {
    let mut a = Mat::<R>::zeros(n, n);
    for i in 0..n {
        a[(i, i)] = R::from(2.0);
        if i > 0 {
            a[(i, i - 1)] = R::from(-1.0);
        }
        if i + 1 < n {
            a[(i, i + 1)] = R::from(-1.0);
        }
    }
    a
}

struct NegatingPc {
    n: usize,
}

impl NegatingPc {
    fn new(n: usize) -> Self {
        Self { n }
    }
}

impl Preconditioner for NegatingPc {
    fn dims(&self) -> (usize, usize) {
        (self.n, self.n)
    }

    fn setup(&mut self, _a: &dyn LinOp<S = S>) -> Result<(), KError> {
        Ok(())
    }

    fn apply(&self, _side: PcSide, x: &[S], y: &mut [S]) -> Result<(), KError> {
        assert_eq!(x.len(), self.n);
        assert_eq!(y.len(), self.n);
        for (yi, &xi) in y.iter_mut().zip(x.iter()) {
            *yi = -xi;
        }
        Ok(())
    }
}

struct WrongDimPc;

impl Preconditioner for WrongDimPc {
    fn dims(&self) -> (usize, usize) {
        (1, 1)
    }

    fn setup(&mut self, _a: &dyn LinOp<S = S>) -> Result<(), KError> {
        Ok(())
    }

    fn apply(&self, _side: PcSide, _x: &[S], _y: &mut [S]) -> Result<(), KError> {
        panic!("dimension validation should reject before PC application");
    }
}

fn pcg_variants() -> [PcgVariant; 2] {
    [
        PcgVariant::Classic,
        PcgVariant::Pipelined { replace_every: 0 },
    ]
}

#[test]
fn pipelined_matches_classic_solution() {
    let n = 32;
    let a = build_dense_poisson(n);
    let b: Vec<R> = vec![R::from(1.0); n];

    let comm = UniverseComm::NoComm(NoComm);

    let mut classic = PcgSolver::new(1e-10, 200);
    let mut pipeline = PcgSolver::new(1e-10, 200);
    pipeline.set_variant(PcgVariant::Pipelined { replace_every: 0 });

    let mut x_classic: Vec<R> = vec![R::default(); n];
    let mut x_pipe: Vec<R> = vec![R::default(); n];

    let mut wk_classic = Workspace::default();
    classic.setup_workspace(&mut wk_classic);
    let stats_classic = classic
        .solve_with_comm(
            &a,
            None,
            &b,
            &mut x_classic,
            PcSide::Left,
            &comm,
            None,
            Some(&mut wk_classic),
        )
        .expect("classic PCG converged");

    let mut wk_pipe = Workspace::default();
    pipeline.setup_workspace(&mut wk_pipe);
    let stats_pipe = pipeline
        .solve_with_comm(
            &a,
            None,
            &b,
            &mut x_pipe,
            PcSide::Left,
            &comm,
            None,
            Some(&mut wk_pipe),
        )
        .expect("pipelined PCG converged");

    assert!(
        stats_pipe.iterations <= stats_classic.iterations + 2,
        "classic iterations: {}, pipelined iterations: {}",
        stats_classic.iterations,
        stats_pipe.iterations
    );
    let mut diff = R::default();
    for (xc, xp) in x_classic.iter().zip(&x_pipe) {
        diff = diff.max((xc - xp).abs());
    }
    assert!(diff < R::from(1e-8));
    let rel_res = stats_pipe.final_residual / stats_classic.final_residual.max(R::from(1e-30));
    assert!(rel_res < R::from(2.0));
}

#[test]
fn pcg_variants_reject_non_left_preconditioning_side() {
    let a = build_dense_poisson(2);
    let b: Vec<R> = vec![R::from(1.0); 2];
    let comm = UniverseComm::NoComm(NoComm);

    for variant in pcg_variants() {
        for side in [PcSide::Right, PcSide::Symmetric] {
            let mut solver = PcgSolver::new(1e-8, 4).with_variant(variant);
            let mut wk = Workspace::default();
            solver.setup_workspace(&mut wk);
            let mut x: Vec<R> = vec![R::default(); 2];

            let err = solver
                .solve_with_comm(&a, None, &b, &mut x, side, &comm, None, Some(&mut wk))
                .expect_err("PCG must reject non-left preconditioning");

            match err {
                KError::InvalidInput(msg) => {
                    let msg = msg.to_lowercase();
                    assert!(msg.contains("left"), "{msg}");
                    assert!(msg.contains("pcg"), "{msg}");
                }
                other => panic!("unexpected error for {variant:?}/{side:?}: {other:?}"),
            }
        }
    }
}

#[test]
fn pcg_variants_reject_explicit_preconditioner_dimension_mismatch() {
    let a = build_dense_poisson(2);
    let b: Vec<R> = vec![R::from(1.0); 2];
    let comm = UniverseComm::NoComm(NoComm);

    for variant in pcg_variants() {
        let mut solver = PcgSolver::new(1e-8, 4).with_variant(variant);
        let mut pc = WrongDimPc;
        let mut wk = Workspace::default();
        solver.setup_workspace(&mut wk);
        let mut x: Vec<R> = vec![R::default(); 2];

        let err = solver
            .solve_with_comm(
                &a,
                Some(&mut pc),
                &b,
                &mut x,
                PcSide::Left,
                &comm,
                None,
                Some(&mut wk),
            )
            .expect_err("PCG must reject mismatched preconditioner dimensions");

        match err {
            KError::InvalidInput(msg) => {
                let msg = msg.to_lowercase();
                assert!(msg.contains("dimension mismatch"), "{msg}");
                assert!(msg.contains("preconditioner"), "{msg}");
            }
            other => panic!("unexpected error for {variant:?}: {other:?}"),
        }
    }
}

#[test]
fn pcg_variants_detect_indefinite_preconditioner() {
    let a = build_dense_poisson(2);
    let b: Vec<R> = vec![R::from(1.0); 2];
    let comm = UniverseComm::NoComm(NoComm);

    for variant in pcg_variants() {
        let mut solver = PcgSolver::new(1e-8, 4).with_variant(variant);
        let mut pc = NegatingPc::new(2);
        let mut wk = Workspace::default();
        solver.setup_workspace(&mut wk);
        let mut x: Vec<R> = vec![R::default(); 2];

        let err = solver
            .solve_with_comm(
                &a,
                Some(&mut pc),
                &b,
                &mut x,
                PcSide::Left,
                &comm,
                None,
                Some(&mut wk),
            )
            .expect_err("PCG must reject indefinite preconditioner");

        assert!(
            matches!(err, KError::IndefinitePreconditioner),
            "unexpected error for {variant:?}: {err:?}"
        );
    }
}

#[test]
fn pcg_variants_detect_indefinite_matrix() {
    let mut a = Mat::<R>::zeros(2, 2);
    a[(0, 0)] = R::default();
    a[(1, 1)] = R::from(1.0);
    let b: Vec<R> = vec![R::from(1.0), R::default()];
    let comm = UniverseComm::NoComm(NoComm);

    for variant in pcg_variants() {
        let mut solver = PcgSolver::new(1e-8, 4).with_variant(variant);
        let mut wk = Workspace::default();
        solver.setup_workspace(&mut wk);
        let mut x: Vec<R> = vec![R::default(); 2];

        let err = solver
            .solve_with_comm(
                &a,
                None,
                &b,
                &mut x,
                PcSide::Left,
                &comm,
                None,
                Some(&mut wk),
            )
            .expect_err("PCG must reject indefinite matrix");

        assert!(
            matches!(err, KError::IndefiniteMatrix),
            "unexpected error for {variant:?}: {err:?}"
        );
    }
}

#[cfg(feature = "rayon")]
#[test]
fn pcg_direct_rayon_comm_routes_through_canonical_cg() {
    let mut a = Mat::<R>::zeros(2, 2);
    a[(0, 0)] = R::NAN;
    a[(1, 1)] = R::from(1.0);
    let b: Vec<R> = vec![R::from(1.0), R::default()];
    let comm = RayonComm::new();

    for variant in pcg_variants() {
        let mut solver = PcgSolver::new(1e-8, 4).with_variant(variant);
        let mut wk = Workspace::default();
        solver.setup_workspace(&mut wk);
        let mut x: Vec<R> = vec![R::default(); 2];

        let err = solver
            .solve_with_comm(
                &a,
                None,
                &b,
                &mut x,
                PcSide::Left,
                &comm,
                None,
                Some(&mut wk),
            )
            .expect_err("direct RayonComm PCG should use canonical CG error handling");

        match err {
            KError::NonFiniteReduction { context, .. } => {
                assert!(context.contains("cg"), "{context}");
            }
            other => panic!("unexpected error for {variant:?}: {other:?}"),
        }
    }
}

#[derive(Clone)]
struct SplitOnlyComm {
    split_calls: Arc<AtomicUsize>,
}

impl Comm for SplitOnlyComm {
    type Vec = Vec<R>;
    type Request<'a> = ();

    fn rank(&self) -> usize {
        0
    }

    fn size(&self) -> usize {
        1
    }

    fn barrier(&self) {}

    #[cfg(feature = "mpi")]
    fn scatter<T: Clone + mpi::datatype::Equivalence>(
        &self,
        global: &[T],
        out: &mut [T],
        _root: usize,
    ) {
        out.clone_from_slice(&global[..out.len()]);
    }

    #[cfg(not(feature = "mpi"))]
    fn scatter<T: Clone>(&self, global: &[T], out: &mut [T], _root: usize) {
        out.clone_from_slice(&global[..out.len()]);
    }

    #[cfg(feature = "mpi")]
    fn gather<T: Clone + mpi::datatype::Equivalence>(
        &self,
        local: &[T],
        out: &mut Vec<T>,
        _root: usize,
    ) {
        out.clear();
        out.extend_from_slice(local);
    }

    #[cfg(not(feature = "mpi"))]
    fn gather<T: Clone>(&self, local: &[T], out: &mut Vec<T>, _root: usize) {
        out.clear();
        out.extend_from_slice(local);
    }

    fn all_reduce_f64(&self, local: f64) -> f64 {
        local
    }

    fn split(&self, _color: i32, _key: i32) -> UniverseComm {
        self.split_calls.fetch_add(1, Ordering::Relaxed);
        UniverseComm::NoComm(NoComm)
    }

    fn irecv_from<'a>(&'a self, _buf: &'a mut [f64], _src: i32) -> Self::Request<'a> {}

    fn isend_to<'a>(&'a self, _buf: &'a [f64], _dest: i32) -> Self::Request<'a> {}

    fn irecv_from_u64<'a>(&'a self, _buf: &'a mut [u64], _src: i32) -> Self::Request<'a> {}

    fn isend_to_u64<'a>(&'a self, _buf: &'a [u64], _dest: i32) -> Self::Request<'a> {}

    fn wait_all<'a>(&self, _reqs: &mut [Self::Request<'a>]) {}
}

#[test]
fn pcg_generic_comm_split_routes_through_canonical_cg() {
    let mut a = Mat::<R>::zeros(2, 2);
    a[(0, 0)] = R::NAN;
    a[(1, 1)] = R::from(1.0);
    let b: Vec<R> = vec![R::from(1.0), R::default()];
    let split_calls = Arc::new(AtomicUsize::new(0));
    let comm = SplitOnlyComm {
        split_calls: split_calls.clone(),
    };

    for variant in pcg_variants() {
        let mut solver = PcgSolver::new(1e-8, 4).with_variant(variant);
        let mut wk = Workspace::default();
        solver.setup_workspace(&mut wk);
        let mut x: Vec<R> = vec![R::default(); 2];

        let err = solver
            .solve_with_comm(
                &a,
                None,
                &b,
                &mut x,
                PcSide::Left,
                &comm,
                None,
                Some(&mut wk),
            )
            .expect_err("generic Comm PCG should use canonical CG after split");

        match err {
            KError::NonFiniteReduction { context, .. } => {
                assert!(context.contains("cg"), "{context}");
            }
            other => panic!("unexpected error for {variant:?}: {other:?}"),
        }
    }

    assert_eq!(split_calls.load(Ordering::Relaxed), pcg_variants().len());
}

#[test]
fn pipelined_reports_reduction_counts() -> Result<(), KError> {
    let n = 32;
    let a = csr_poisson_1d(n);
    let b: Vec<R> = vec![R::from(1.0); n];

    let comm = UniverseComm::NoComm(NoComm);
    install_test_counter(true);
    let mut solver =
        PcgSolver::new(1e-12, 100).with_variant(PcgVariant::Pipelined { replace_every: 0 });
    debug_assert!(matches!(
        solver.variant(),
        PcgVariant::Pipelined { replace_every: 0 }
    ));

    let mut wk = Workspace::default();
    solver.setup_workspace(&mut wk);
    let mut x: Vec<R> = vec![R::default(); n];

    let op: &dyn LinOp<S = f64> = &a;
    let mut pc = Jacobi::new();
    pc.setup(op)?;

    let stats = solver
        .solve_with_comm(
            op,
            Some(&mut pc),
            &b,
            &mut x,
            PcSide::Left,
            &comm,
            None,
            Some(&mut wk),
        )
        .expect("pipelined PCG converged");

    let counters = take_test_counter();
    install_test_counter(false);

    let expected = stats.iterations + 1; // fused startup tuple plus one fused tuple per iteration

    assert!(
        counters.allreduces >= expected,
        "allreduces {} smaller than expected {}",
        counters.allreduces,
        expected
    );
    assert!(
        counters.allreduces <= expected + 6,
        "allreduces {} larger than expected upper bound {}",
        counters.allreduces,
        expected + 6
    );
    assert!(
        stats.counters.num_global_reductions >= counters.allreduces,
        "reported {} reductions, less than observed {} allreduces",
        stats.counters.num_global_reductions,
        counters.allreduces
    );
    assert!(
        stats.counters.num_global_reductions > 0,
        "serialized solver reported zero reductions"
    );
    Ok(())
}

#[test]
fn pcg_async_configuration_is_forwarded_to_canonical_cg() {
    let mut solver = PcgSolver::new(1e-12, 100);
    assert!(solver.async_enabled());
    assert_eq!(solver.async_min_n(), 10_000);

    solver.set_async_enabled(false);
    solver.set_async_min_n(123);
    assert!(!solver.async_enabled());
    assert_eq!(solver.async_min_n(), 123);
}

#[test]
fn pcg_wrapper_restores_workspace_reduction_engine() {
    let n = 16;
    let a = csr_poisson_1d(n);
    let b: Vec<R> = vec![R::from(1.0); n];
    let mut x: Vec<R> = vec![R::default(); n];
    let comm = UniverseComm::NoComm(NoComm);
    let op: &dyn LinOp<S = f64> = &a;

    let mut solver =
        PcgSolver::new(1e-12, 100).with_variant(PcgVariant::Pipelined { replace_every: 0 });
    let mut wk = Workspace::default();
    solver.setup_workspace(&mut wk);
    assert!(wk.reduction_engine().is_none());

    let stats = solver
        .solve_with_comm(
            op,
            None,
            &b,
            &mut x,
            PcSide::Left,
            &comm,
            None,
            Some(&mut wk),
        )
        .expect("PCG solve");
    assert!(stats.reason.is_converged());

    assert!(
        wk.reduction_engine().is_none(),
        "PCG wrapper should restore an initially absent reduction engine"
    );
}

#[test]
fn pcg_true_residual_monitor_is_forwarded_to_canonical_cg_and_restored() {
    let n = 16;
    let a = csr_poisson_1d(n);
    let b: Vec<R> = vec![R::from(1.0); n];
    let comm = UniverseComm::NoComm(NoComm);
    let op: &dyn LinOp<S = f64> = &a;

    let calls = Arc::new(AtomicUsize::new(0));
    let calls_cb = calls.clone();
    let mut solver = PcgSolver::new(1e-12, 100)
        .with_variant(PcgVariant::Pipelined { replace_every: 0 })
        .with_true_residual_monitor(Box::new(move |_iter, residual, _reductions| {
            assert!(residual.is_finite());
            calls_cb.fetch_add(1, Ordering::SeqCst);
            MonitorAction::Continue
        }));

    for _ in 0..2 {
        let mut wk = Workspace::default();
        solver.setup_workspace(&mut wk);
        let mut x: Vec<R> = vec![R::default(); n];
        let stats = solver
            .solve_with_comm(
                op,
                None,
                &b,
                &mut x,
                PcSide::Left,
                &comm,
                None,
                Some(&mut wk),
            )
            .expect("PCG solve");
        assert!(stats.reason.is_converged());
    }

    assert!(
        calls.load(Ordering::SeqCst) >= 4,
        "true residual monitor should be invoked across repeated wrapper solves"
    );
}

#[test]
fn pipelined_reductions_scale_with_iteration_count() -> Result<(), KError> {
    let n = 64;
    let a = csr_poisson_1d(n);
    let b: Vec<R> = vec![R::from(1.0); n];
    let comm = UniverseComm::NoComm(NoComm);
    let op: &dyn LinOp<S = f64> = &a;

    let run = |tol: f64, maxits: usize| -> Result<_, KError> {
        let mut solver =
            PcgSolver::new(tol, maxits).with_variant(PcgVariant::Pipelined { replace_every: 0 });
        let mut wk = Workspace::default();
        solver.setup_workspace(&mut wk);
        let mut x: Vec<R> = vec![R::default(); n];
        let mut pc = Jacobi::new();
        pc.setup(op)?;
        solver.solve_with_comm(
            op,
            Some(&mut pc),
            &b,
            &mut x,
            PcSide::Left,
            &comm,
            None,
            Some(&mut wk),
        )
    };

    let short = run(1e-2, 4)?;
    let long = run(1e-12, 150)?;
    assert!(long.iterations > short.iterations);
    assert!(long.counters.num_global_reductions > short.counters.num_global_reductions);

    // Pipelined PCG executes one fused iterative reduction per iteration plus startup overhead.
    let delta_iter = long.iterations - short.iterations;
    let delta_red = long
        .counters
        .num_global_reductions
        .saturating_sub(short.counters.num_global_reductions);
    assert!(
        delta_red >= delta_iter,
        "expected reductions to grow at least linearly with iterations (Δred={}, Δiter={})",
        delta_red,
        delta_iter
    );
    Ok(())
}

mod fixtures;
