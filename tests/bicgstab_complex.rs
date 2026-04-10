#![cfg(feature = "complex")]

mod support;

use approx::assert_abs_diff_eq;
use kryst::algebra::bridge::BridgeScratch;
use kryst::algebra::prelude::*;
use kryst::context::ksp_context::Workspace;
use kryst::error::KError;
use kryst::ops::klinop::KLinOp;
use kryst::ops::kpc::KPreconditioner;
use kryst::parallel::{NoComm, UniverseComm};
use kryst::preconditioner::PcSide;
use kryst::solver::BiCgStabSolver;
use kryst::utils::convergence::{AcceptanceStatus, ConvergedReason};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use support::complex_dense::diagonally_dominant_system;

const SEED: u64 = 0xB1C6_57AB_u64;
const ROW_SCALE: f64 = 0.2;
const DIAG_SHIFT: f64 = 1.0;

struct CountingIdentityPc {
    n: usize,
    hits: Arc<AtomicUsize>,
}

impl CountingIdentityPc {
    fn new(n: usize, hits: Arc<AtomicUsize>) -> Self {
        Self { n, hits }
    }
}

impl KPreconditioner for CountingIdentityPc {
    type Scalar = S;

    fn dims(&self) -> (usize, usize) {
        (self.n, self.n)
    }

    fn apply_s(
        &self,
        _side: PcSide,
        x: &[S],
        y: &mut [S],
        _scratch: &mut BridgeScratch,
    ) -> Result<(), KError> {
        self.hits.fetch_add(1, Ordering::Relaxed);
        y.copy_from_slice(x);
        Ok(())
    }
}

struct ScriptedMatvecOp {
    n: usize,
    scripted_outputs: Vec<Vec<S>>,
    calls: AtomicUsize,
}

impl ScriptedMatvecOp {
    fn new(n: usize, scripted_outputs: Vec<Vec<S>>) -> Self {
        Self {
            n,
            scripted_outputs,
            calls: AtomicUsize::new(0),
        }
    }
}

impl KLinOp for ScriptedMatvecOp {
    type Scalar = S;

    fn dims(&self) -> (usize, usize) {
        (self.n, self.n)
    }

    fn matvec_s(&self, x: &[S], y: &mut [S], _scratch: &mut BridgeScratch) {
        let call = self.calls.fetch_add(1, Ordering::SeqCst);
        if let Some(scripted) = self.scripted_outputs.get(call) {
            y.copy_from_slice(scripted);
        } else {
            y.copy_from_slice(x);
        }
    }
}

#[test]
fn bicgstab_solves_random_complex_system() {
    let n = 7;
    let (op, x_true, b) = diagonally_dominant_system(n, SEED, ROW_SCALE, DIAG_SHIFT);
    let comm = UniverseComm::NoComm(NoComm);
    let mut solver = BiCgStabSolver::new(1e-10, 300);
    let mut work = Workspace::new(n);
    let mut x = vec![S::zero(); n];

    let stats = solver
        .solve(
            &op,
            None,
            &b,
            &mut x,
            PcSide::Left,
            &comm,
            None,
            Some(&mut work),
        )
        .expect("BiCGStab solve");

    assert!(matches!(
        stats.reason,
        ConvergedReason::ConvergedRtol | ConvergedReason::ConvergedAtol
    ));

    let err = op.residual_norm(&x, &b);
    assert!(err < 1e-9, "BiCGStab residual too large: {err:e}");

    for (approx, exact) in x.iter().zip(x_true.iter()) {
        assert_abs_diff_eq!(approx.real(), exact.real(), epsilon = 1e-7);
        assert_abs_diff_eq!(approx.imag(), exact.imag(), epsilon = 1e-7);
    }
}

#[test]
fn bicgstab_with_sor_preconditioner_complex() {
    use kryst::matrix::sparse::CsrMatrix;
    use kryst::preconditioner::sor::{MatSorType, SorPc};

    let n = 7;
    let (op, _x_true, b) = diagonally_dominant_system(n, SEED, ROW_SCALE, DIAG_SHIFT);
    let comm = UniverseComm::NoComm(NoComm);
    let mut solver = BiCgStabSolver::new(1e-9, 300);
    let mut work = Workspace::new(n);
    let mut x = vec![S::zero(); n];

    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::new();
    let mut vals = Vec::new();
    row_ptr.push(0);
    for i in 0..n {
        if i > 0 {
            col_idx.push(i - 1);
            vals.push(S::from_real(-0.1));
        }
        col_idx.push(i);
        vals.push(S::from_real(2.0));
        if i + 1 < n {
            col_idx.push(i + 1);
            vals.push(S::from_real(-0.1));
        }
        row_ptr.push(col_idx.len());
    }
    let a = CsrMatrix::from_csr(n, n, row_ptr, col_idx, vals);
    let mut sor = SorPc::new(1.0, 1, MatSorType::APPLY_LOWER, 0.0);
    kryst::preconditioner::Preconditioner::setup(&mut sor, &a).unwrap();

    let stats = solver
        .solve(
            &op,
            Some(&sor as &dyn KPreconditioner<Scalar = S>),
            &b,
            &mut x,
            PcSide::Left,
            &comm,
            None,
            Some(&mut work),
        )
        .expect("BiCGStab+SOR solve");
    assert!(matches!(
        stats.reason,
        ConvergedReason::ConvergedRtol | ConvergedReason::ConvergedAtol
    ));
}

#[test]
fn bicgstab_complex_left_right_preconditioned_nonzero_guess_reports_stats() {
    let n = 6;
    let (op, x_true, b) = diagonally_dominant_system(n, SEED ^ 0x55AA, ROW_SCALE, DIAG_SHIFT);
    let comm = UniverseComm::NoComm(NoComm);

    for side in [PcSide::Left, PcSide::Right] {
        let hits = Arc::new(AtomicUsize::new(0));
        let pc = CountingIdentityPc::new(n, Arc::clone(&hits));
        let mut solver = BiCgStabSolver::new(1e-10, 400);
        let mut work = Workspace::new(n);
        let mut x = vec![S::from_real(0.15); n];
        let x0 = x.clone();

        let stats = solver
            .solve(
                &op,
                Some(&pc),
                &b,
                &mut x,
                side,
                &comm,
                None,
                Some(&mut work),
            )
            .expect("BiCGStab complex solve");

        assert!(
            matches!(
                stats.reason,
                ConvergedReason::ConvergedRtol | ConvergedReason::ConvergedAtol
            ),
            "side={side:?}, stats={stats:?}"
        );
        assert_ne!(x, x0, "side={side:?} should update nonzero initial guess");
        assert!(
            hits.load(Ordering::Relaxed) > 0,
            "side={side:?} should apply PC"
        );
        assert!(
            stats.counters.num_global_reductions > 0,
            "side={side:?}, {stats:?}"
        );
        assert_eq!(
            stats.counters.residual_replacements, 0,
            "side={side:?}, {stats:?}"
        );

        let err = op.residual_norm(&x, &b);
        assert!(err < 1e-8, "side={side:?}, residual too large: {err:e}");
        for (approx, exact) in x.iter().zip(x_true.iter()) {
            assert_abs_diff_eq!(approx.real(), exact.real(), epsilon = 2e-6);
            assert_abs_diff_eq!(approx.imag(), exact.imag(), epsilon = 2e-6);
        }
    }
}

#[test]
fn bicgstab_complex_breakdown_threshold_uses_magnitude_for_complex_scalars() {
    let comm = UniverseComm::NoComm(NoComm);
    let b = vec![S::from_real(1.0), S::zero()];
    let op = ScriptedMatvecOp::new(
        2,
        vec![
            vec![S::zero(), S::zero()],
            vec![S::from_parts(0.0, 1.0e-40), S::zero()],
        ],
    );

    let mut solver = BiCgStabSolver::new(0.0, 8);
    solver.atol = 0.0;
    let mut work = Workspace::new(2);
    let mut x = vec![S::zero(), S::zero()];

    let stats = solver
        .solve(
            &op,
            None,
            &b,
            &mut x,
            PcSide::Left,
            &comm,
            None,
            Some(&mut work),
        )
        .expect("solve should return stats");

    assert_eq!(stats.reason, ConvergedReason::DivergedBreakdownBiCG);
    assert_eq!(stats.acceptance_status, AcceptanceStatus::Failed);
    assert!(stats.final_residual > 0.0);
    assert!(stats.counters.num_global_reductions > 0);
    assert_eq!(stats.counters.residual_replacements, 0);
}

#[test]
fn bicgstab_complex_breakdown_salvage_reconciles_reason_and_metadata() {
    let comm = UniverseComm::NoComm(NoComm);
    let b = vec![S::from_real(1.0), S::zero()];
    let op = ScriptedMatvecOp::new(
        2,
        vec![
            vec![S::zero(), S::zero()],
            vec![S::from_parts(0.0, 1.0e-40), S::zero()],
        ],
    );

    let mut solver = BiCgStabSolver::new(0.5, 8);
    solver.atol = 0.0;
    let mut work = Workspace::new(2);
    let mut x = vec![S::from_real(1.0), S::zero()];

    let stats = solver
        .solve(
            &op,
            None,
            &b,
            &mut x,
            PcSide::Left,
            &comm,
            None,
            Some(&mut work),
        )
        .expect("solve should return stats");

    assert_eq!(stats.reason, ConvergedReason::ConvergedHappyBreakdown);
    assert_eq!(stats.acceptance_status, AcceptanceStatus::OkWithWarning);
    assert_eq!(
        stats.breakdown_reason,
        Some(ConvergedReason::DivergedBreakdownBiCG)
    );
    assert!(
        stats
            .residual_override_note
            .as_deref()
            .is_some_and(|note| note.contains("salvaged"))
    );
    assert!(stats.final_residual <= 1.0, "stats={stats:?}");
    assert!(stats.counters.num_global_reductions > 0);
}
