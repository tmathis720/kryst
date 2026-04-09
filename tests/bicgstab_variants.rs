#![cfg(all(feature = "backend-faer", not(feature = "complex")))]

use faer::Mat;
use kryst::config::options::KspOptions;
use kryst::context::ksp_context::KspContext;
use kryst::error::KError;
use kryst::matrix::op::{LinOp, LinOpF64};
use kryst::parallel::{NoComm, UniverseComm};
use kryst::preconditioner::{PcSide, Preconditioner};
use kryst::solver::{
    BiCgStabBreakdownPolicy, BiCgStabSolver, BiCgStabVariant, MonitorAction, MonitorCallback,
};
use kryst::utils::convergence::{AcceptanceStatus, ConvergedReason};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

fn nonsym_tridiag(n: usize) -> Mat<f64> {
    let mut a = Mat::<f64>::zeros(n, n);
    for i in 0..n {
        a[(i, i)] = 4.0;
        if i > 0 {
            a[(i, i - 1)] = -1.0;
        }
        if i + 1 < n {
            a[(i, i + 1)] = 2.0;
        }
    }
    a
}

struct NanOnceOp {
    a: Mat<f64>,
    nan_call: usize,
    calls: AtomicUsize,
}

struct ScriptedMatvecOp {
    n: usize,
    scripted_outputs: Vec<Vec<f64>>,
    calls: AtomicUsize,
}

impl ScriptedMatvecOp {
    fn new(n: usize, scripted_outputs: Vec<Vec<f64>>) -> Self {
        Self {
            n,
            scripted_outputs,
            calls: AtomicUsize::new(0),
        }
    }

    fn matvec_impl(&self, x: &[f64], y: &mut [f64]) {
        let call = self.calls.fetch_add(1, Ordering::SeqCst);
        if let Some(scripted) = self.scripted_outputs.get(call) {
            y.copy_from_slice(scripted);
            return;
        }
        y.copy_from_slice(x);
    }
}

impl LinOp for ScriptedMatvecOp {
    type S = f64;

    fn dims(&self) -> (usize, usize) {
        (self.n, self.n)
    }

    fn matvec(&self, x: &[f64], y: &mut [f64]) {
        self.matvec_impl(x, y);
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl LinOpF64 for ScriptedMatvecOp {
    fn dims(&self) -> (usize, usize) {
        <Self as LinOp>::dims(self)
    }

    fn matvec(&self, x: &[f64], y: &mut [f64]) {
        self.matvec_impl(x, y);
    }
}

struct CountingIdentityPc {
    side_hits: Arc<AtomicUsize>,
}

impl CountingIdentityPc {
    fn new(side_hits: Arc<AtomicUsize>) -> Self {
        Self { side_hits }
    }
}

impl Preconditioner for CountingIdentityPc {
    fn setup(&mut self, _a: &dyn LinOp<S = f64>) -> Result<(), KError> {
        Ok(())
    }

    fn apply(&self, _side: PcSide, x: &[f64], y: &mut [f64]) -> Result<(), KError> {
        self.side_hits.fetch_add(1, Ordering::Relaxed);
        y.copy_from_slice(x);
        Ok(())
    }
}

impl NanOnceOp {
    fn new(a: Mat<f64>, nan_call: usize) -> Self {
        Self {
            a,
            nan_call,
            calls: AtomicUsize::new(0),
        }
    }

    fn matvec_impl(&self, x: &[f64], y: &mut [f64]) {
        let call = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
        if call == self.nan_call {
            y.fill(f64::NAN);
            return;
        }
        for i in 0..self.a.nrows() {
            let mut acc = 0.0;
            for j in 0..self.a.ncols() {
                acc += self.a[(i, j)] * x[j];
            }
            y[i] = acc;
        }
    }
}

impl LinOp for NanOnceOp {
    type S = f64;

    fn dims(&self) -> (usize, usize) {
        (self.a.nrows(), self.a.ncols())
    }

    fn matvec(&self, x: &[f64], y: &mut [f64]) {
        self.matvec_impl(x, y);
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl LinOpF64 for NanOnceOp {
    fn dims(&self) -> (usize, usize) {
        <Self as LinOp>::dims(self)
    }

    fn matvec(&self, x: &[f64], y: &mut [f64]) {
        self.matvec_impl(x, y);
    }
}

#[test]
fn bicgstab_fewerchecks_keeps_convergence_and_reduces_syncs() {
    let n = 36;
    let a = nonsym_tridiag(n);
    let b = vec![1.0; n];
    let comm = UniverseComm::NoComm(NoComm);

    let mut x_classic = vec![0.0; n];
    let mut classic = BiCgStabSolver::new(1e-9, 300);
    classic.set_variant(BiCgStabVariant::Classic);
    let mut ws_classic = kryst::context::ksp_context::Workspace::default();
    let stats_classic = classic
        .solve_f64(
            &a,
            None,
            &b,
            &mut x_classic,
            kryst::preconditioner::PcSide::Left,
            &comm,
            None,
            Some(&mut ws_classic),
        )
        .unwrap();

    let mut x_low = vec![0.0; n];
    let mut low = BiCgStabSolver::new(1e-9, 300);
    low.set_variant(BiCgStabVariant::FewerChecks);
    let mut ws_low = kryst::context::ksp_context::Workspace::default();
    let stats_low = low
        .solve_f64(
            &a,
            None,
            &b,
            &mut x_low,
            kryst::preconditioner::PcSide::Left,
            &comm,
            None,
            Some(&mut ws_low),
        )
        .unwrap();

    assert!(stats_low.final_residual <= 1e-7);
    assert!(
        stats_low.counters.num_global_reductions <= stats_classic.counters.num_global_reductions
    );
}

#[test]
fn bicgstab_variant_selectable_from_ksp_options() {
    let opts = KspOptions::from_args(&[
        "-ksp_type",
        "bicgstab",
        "-ksp_bicgstab_variant",
        "fewerchecks",
    ])
    .unwrap();
    let mut ksp = KspContext::new();
    ksp.set_from_options(&opts).unwrap();

    let n = 20;
    let a = nonsym_tridiag(n);
    let b = vec![1.0; n];
    let mut x = vec![0.0; n];
    ksp.set_operators(Arc::new(a), None);
    let stats = ksp.solve(&b, &mut x).unwrap();

    assert_eq!(
        stats.reduction_model.as_ref().map(|m| m.variant),
        Some("bicgstab-fewer-checks")
    );
}

#[test]
fn bicgstab_rejects_symmetric_pc_side() {
    let n = 24;
    let a = nonsym_tridiag(n);
    let b = vec![1.0; n];
    let mut x = vec![0.0; n];
    let comm = UniverseComm::NoComm(NoComm);
    let mut solver = BiCgStabSolver::new(1e-9, 100);
    let mut ws = kryst::context::ksp_context::Workspace::default();

    let err = solver
        .solve_f64(
            &a,
            None,
            &b,
            &mut x,
            PcSide::Symmetric,
            &comm,
            None,
            Some(&mut ws),
        )
        .expect_err("BiCGStab must reject PcSide::Symmetric");

    match err {
        KError::InvalidInput(msg) => {
            assert!(msg.contains("PcSide::Symmetric"));
            assert!(msg.contains("unsupported"));
            assert!(msg.contains("PcSide::Left"));
            assert!(msg.contains("PcSide::Right"));
        }
        other => panic!("expected InvalidInput for symmetric side, got: {other:?}"),
    }
}

#[test]
fn bicgstab_rho_breakdown_policy_strict_exits() {
    let mut a = Mat::<f64>::zeros(3, 3);
    a[(0, 0)] = 4.0;
    a[(0, 1)] = 1.0;
    a[(1, 0)] = -1.0;
    a[(1, 1)] = 3.0;
    a[(1, 2)] = 1.0;
    a[(2, 1)] = -2.0;
    a[(2, 2)] = 5.0;
    let op = NanOnceOp::new(a, 2);
    let b = vec![1.0, 2.0, -1.0];
    let mut x = vec![0.0; 3];
    let mut solver = BiCgStabSolver::new(1e-10, 40);
    solver.set_breakdown_policy(BiCgStabBreakdownPolicy::Strict);
    let comm = UniverseComm::NoComm(NoComm);
    let mut ws = kryst::context::ksp_context::Workspace::default();
    let stats = solver
        .solve_f64(
            &op,
            None,
            &b,
            &mut x,
            PcSide::Left,
            &comm,
            None,
            Some(&mut ws),
        )
        .expect("strict solve should return stats");

    assert!(matches!(
        stats.reason,
        ConvergedReason::DivergedBreakdownBiCG | ConvergedReason::ConvergedHappyBreakdown
    ));
    assert_eq!(stats.counters.residual_replacements, 0);
}

#[test]
fn bicgstab_rho_breakdown_policy_refresh_shadow_recovers() {
    let a = nonsym_tridiag(3);
    let b = vec![1.0e-20, -2.0e-20, 1.0e-20];
    let comm = UniverseComm::NoComm(NoComm);

    let mut x_strict = vec![0.0; 3];
    let mut strict = BiCgStabSolver::new(0.0, 40);
    strict.atol = 0.0;
    strict.set_breakdown_policy(BiCgStabBreakdownPolicy::Strict);
    let mut ws_strict = kryst::context::ksp_context::Workspace::default();
    let strict_stats = strict
        .solve_f64(
            &a,
            None,
            &b,
            &mut x_strict,
            PcSide::Left,
            &comm,
            None,
            Some(&mut ws_strict),
        )
        .expect("strict solve should return stats");

    let mut x = vec![0.0; 3];
    let mut solver = BiCgStabSolver::new(0.0, 40);
    solver.atol = 0.0;
    solver.set_breakdown_policy(BiCgStabBreakdownPolicy::RefreshShadow { max_refreshes: 1 });
    let mut ws = kryst::context::ksp_context::Workspace::default();
    let stats = solver
        .solve_f64(
            &a,
            None,
            &b,
            &mut x,
            PcSide::Left,
            &comm,
            None,
            Some(&mut ws),
        )
        .expect("refresh solve should return stats");

    assert!(stats.counters.residual_replacements >= 1);
    assert!(stats.iterations > strict_stats.iterations);
}

#[test]
fn bicgstab_handles_t_zero_and_s_zero_as_converged_for_left_and_right() {
    let comm = UniverseComm::NoComm(NoComm);
    let b = vec![1.0, -2.0];
    for side in [PcSide::Left, PcSide::Right] {
        let op = ScriptedMatvecOp::new(
            2,
            vec![
                vec![0.0, 0.0],  // initial A*x0
                vec![1.0, -2.0], // v = A*p
                vec![0.0, 0.0],  // t = A*s
            ],
        );
        let mut solver = BiCgStabSolver::new(1e-12, 10);
        solver.set_variant(BiCgStabVariant::FewerChecks);
        let mut ws = kryst::context::ksp_context::Workspace::default();
        let mut x = vec![0.0, 0.0];
        let stats = solver
            .solve_f64(&op, None, &b, &mut x, side, &comm, None, Some(&mut ws))
            .expect("solve should return stats");

        assert!(
            stats.reason.is_converged(),
            "side={side:?}, stats={stats:?}"
        );
        assert!(
            stats.final_residual <= 1e-10,
            "side={side:?}, stats={stats:?}"
        );
    }
}

#[test]
fn bicgstab_handles_t_zero_and_s_nonzero_as_breakdown_for_left_and_right() {
    let comm = UniverseComm::NoComm(NoComm);
    let b = vec![1.0, 1.0];
    for side in [PcSide::Left, PcSide::Right] {
        let op = ScriptedMatvecOp::new(
            2,
            vec![
                vec![0.0, 0.0], // initial A*x0
                vec![1.0, 0.0], // v = A*p -> s != 0 after alpha step
                vec![0.0, 0.0], // t = A*s
            ],
        );
        let mut solver = BiCgStabSolver::new(1e-12, 10);
        solver.set_variant(BiCgStabVariant::FewerChecks);
        solver.atol = 0.0;
        solver.rtol = 0.0;
        let mut ws = kryst::context::ksp_context::Workspace::default();
        let mut x = vec![0.0, 0.0];
        let stats = solver
            .solve_f64(&op, None, &b, &mut x, side, &comm, None, Some(&mut ws))
            .expect("solve should return stats");

        assert_eq!(
            stats.reason,
            ConvergedReason::DivergedBreakdownBiCG,
            "side={side:?}, stats={stats:?}"
        );
    }
}

#[test]
fn bicgstab_preconditioned_left_and_right_support_nonzero_initial_guess() {
    let n = 24;
    let a = nonsym_tridiag(n);
    let b = vec![1.0; n];
    let comm = UniverseComm::NoComm(NoComm);

    for side in [PcSide::Left, PcSide::Right] {
        let hits = Arc::new(AtomicUsize::new(0));
        let mut pc = CountingIdentityPc::new(Arc::clone(&hits));
        pc.setup(&a).expect("pc setup");
        let mut solver = BiCgStabSolver::new(1e-10, 300);
        let mut ws = kryst::context::ksp_context::Workspace::default();
        let mut x = vec![0.35; n];
        let x0 = x.clone();

        let stats = solver
            .solve_f64(&a, Some(&pc), &b, &mut x, side, &comm, None, Some(&mut ws))
            .expect("preconditioned solve should return stats");

        assert!(matches!(
            stats.reason,
            ConvergedReason::ConvergedRtol | ConvergedReason::ConvergedAtol
        ));
        assert!(
            hits.load(Ordering::Relaxed) > 0,
            "preconditioner was not used on side {side:?}"
        );
        assert_ne!(x, x0, "nonzero initial guess path should update x");
    }
}

#[test]
fn bicgstab_alpha_den_breakdown_salvage_branch_is_reachable() {
    let comm = UniverseComm::NoComm(NoComm);
    let b = vec![1.0, 0.0];
    let op = ScriptedMatvecOp::new(
        2,
        vec![
            vec![0.0, 0.0], // A*x0
            vec![0.0, 1.0], // v orthogonal to r_hat => alpha_den = 0
        ],
    );
    let mut solver = BiCgStabSolver::new(0.0, 6);
    solver.atol = 0.0;
    solver.rtol = 0.0;
    let mut ws = kryst::context::ksp_context::Workspace::default();
    let mut x = vec![0.0, 0.0];

    let stats = solver
        .solve_f64(
            &op,
            None,
            &b,
            &mut x,
            PcSide::Left,
            &comm,
            None,
            Some(&mut ws),
        )
        .expect("solve should return stats");

    assert_eq!(stats.reason, ConvergedReason::DivergedBreakdownBiCG);
    assert_eq!(stats.acceptance_status, AcceptanceStatus::Failed);
    assert!(stats.final_residual > 0.0);
}

#[test]
fn bicgstab_omega_den_breakdown_salvage_branch_is_reachable() {
    let comm = UniverseComm::NoComm(NoComm);
    let b = vec![1.0, 0.0];
    let op = ScriptedMatvecOp::new(
        2,
        vec![
            vec![0.0, 0.0],      // A*x0
            vec![1.0, 1.0],      // v => alpha_den != 0, s != 0
            vec![f64::NAN, 0.0], // t => omega_den non-finite
        ],
    );
    let mut solver = BiCgStabSolver::new(0.0, 6);
    solver.atol = 0.0;
    solver.rtol = 0.0;
    solver.set_variant(BiCgStabVariant::FewerChecks);
    let mut ws = kryst::context::ksp_context::Workspace::default();
    let mut x = vec![0.0, 0.0];

    let stats = solver
        .solve_f64(
            &op,
            None,
            &b,
            &mut x,
            PcSide::Left,
            &comm,
            None,
            Some(&mut ws),
        )
        .expect("solve should return stats");

    assert_eq!(stats.reason, ConvergedReason::DivergedBreakdownBiCG);
    assert_eq!(stats.acceptance_status, AcceptanceStatus::Failed);
}

#[test]
fn bicgstab_reliable_residual_replacements_interact_with_monitors() {
    let n = 32;
    let a = nonsym_tridiag(n);
    let b = vec![1.0; n];
    let comm = UniverseComm::NoComm(NoComm);
    let mut x = vec![0.2; n];
    let mut solver = BiCgStabSolver::new(1e-9, 300);
    solver.set_variant(BiCgStabVariant::Reliable {
        residual_replace_every: 1,
    });
    let mut ws = kryst::context::ksp_context::Workspace::default();

    let observed_residuals = Arc::new(Mutex::new(Vec::new()));
    let observed_residuals_cl = Arc::clone(&observed_residuals);
    let monitor: Box<MonitorCallback<f64>> = Box::new(move |_iter, residual, _| {
        observed_residuals_cl
            .lock()
            .expect("hist lock")
            .push(residual);
        MonitorAction::Continue
    });
    let monitors = vec![monitor];

    let stats = solver
        .solve_f64(
            &a,
            None,
            &b,
            &mut x,
            PcSide::Left,
            &comm,
            Some(&monitors),
            Some(&mut ws),
        )
        .expect("reliable solve should return stats");

    let observed = observed_residuals.lock().expect("hist lock").len();
    assert!(observed > 0, "monitor should observe residual history");
    assert!(
        stats.counters.residual_replacements >= stats.iterations,
        "replacement counter should include reliable residual replacements"
    );
}
