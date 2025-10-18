#![allow(clippy::float_cmp)]

#[cfg(feature = "complex")]
mod support;

use kryst::algebra::bridge::BridgeScratch;
use kryst::algebra::prelude::*;
use kryst::context::ksp_context::Workspace;
use kryst::ops::klinop::KLinOp;
use kryst::parallel::{NoComm, UniverseComm};
use kryst::preconditioner::PcSide;
use kryst::solver::CgSolver;
use kryst::solver::cg::debug::{self, IterEvent};
use kryst::utils::convergence::ConvergedReason;
use std::sync::{Arc, Mutex, OnceLock};

#[cfg(feature = "complex")]
use support::complex_dense::hermitian_pos_def_system;

fn cg_debug_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

struct HookGuard;

impl HookGuard {
    fn install<F>(hook: F) -> Self
    where
        F: Fn(IterEvent) + Send + Sync + 'static,
    {
        debug::set_iter_hook(Some(Box::new(hook)));
        HookGuard
    }
}

impl Drop for HookGuard {
    fn drop(&mut self) {
        debug::clear_iter_hook();
    }
}

#[cfg(feature = "complex")]
#[test]
fn cg_complex_hpd_invariants() {
    let _guard = cg_debug_lock().lock().unwrap();
    debug::reset_counters();

    let events = Arc::new(Mutex::new(Vec::<IterEvent>::new()));
    let event_sink = events.clone();
    let _hook = HookGuard::install(move |evt| {
        event_sink.lock().unwrap().push(evt);
    });

    let n = 7;
    let (op, _x_true, b) = hermitian_pos_def_system(n, 0xC0DE_u64, 3.0);
    let comm = UniverseComm::NoComm(NoComm);
    let mut solver = CgSolver::new(1e-12, 200);
    let mut work = Workspace::new(n);
    let mut x = vec![S::zero(); n];

    let residuals = Arc::new(Mutex::new(Vec::<R>::new()));
    let monitor_res = residuals.clone();
    let monitors: Vec<Box<dyn Fn(usize, R) + Send + Sync>> = vec![Box::new(move |_iter, res| {
        monitor_res.lock().unwrap().push(res);
    })];

    let stats = solver
        .solve(
            &op,
            None,
            &b,
            &mut x,
            PcSide::Left,
            &comm,
            Some(&monitors),
            Some(&mut work),
        )
        .expect("CG solve");

    assert!(matches!(
        stats.reason,
        ConvergedReason::ConvergedRtol | ConvergedReason::ConvergedAtol
    ));

    let residuals = residuals.lock().unwrap();
    assert!(residuals.len() >= 2);
    for window in residuals.windows(2) {
        let prev = window[0];
        let next = window[1];
        assert!(
            next <= prev * (1.0 + 5e-13),
            "residual increased: {prev} -> {next}"
        );
    }

    let events = events.lock().unwrap();
    assert!(!events.is_empty());
    for evt in events.iter() {
        assert!(evt.p_ap > R::zero(), "nonpositive curvature: {}", evt.p_ap);
        assert!(evt.p_ap.is_finite());
        assert!(evt.alpha.is_finite());
        if let Some(beta) = evt.beta {
            assert!(beta.is_finite());
        }
    }

    assert_eq!(debug::large_imag_count(), 0);
}

#[cfg(not(feature = "complex"))]
#[test]
fn cg_real_spd_invariants() {
    let _guard = cg_debug_lock().lock().unwrap();
    debug::reset_counters();

    let events = Arc::new(Mutex::new(Vec::<IterEvent>::new()));
    let event_sink = events.clone();
    let _hook = HookGuard::install(move |evt| {
        event_sink.lock().unwrap().push(evt);
    });

    let diag = vec![
        S::from_real(3.0),
        S::from_real(5.0),
        S::from_real(7.5),
        S::from_real(2.25),
    ];
    let x_true = vec![
        S::from_real(1.0),
        S::from_real(-2.0),
        S::from_real(0.5),
        S::from_real(3.0),
    ];
    let b: Vec<S> = diag
        .iter()
        .zip(x_true.iter())
        .map(|(&d, &x)| d * x)
        .collect();
    let mut x = vec![S::zero(); diag.len()];
    let op = DiagonalOp { diag: diag.clone() };
    let comm = UniverseComm::NoComm(NoComm);
    let mut solver = CgSolver::new(1e-10, 200);
    let mut work = Workspace::new(diag.len());

    let residuals = Arc::new(Mutex::new(Vec::<R>::new()));
    let residual_sink = residuals.clone();
    let monitors: Vec<Box<dyn Fn(usize, R) + Send + Sync>> = vec![Box::new(move |_iter, res| {
        residual_sink.lock().unwrap().push(res);
    })];

    let stats = solver
        .solve(
            &op,
            None,
            &b,
            &mut x,
            PcSide::Left,
            &comm,
            Some(&monitors),
            Some(&mut work),
        )
        .expect("CG solve");

    assert!(matches!(
        stats.reason,
        ConvergedReason::ConvergedRtol | ConvergedReason::ConvergedAtol
    ));

    let residuals = residuals.lock().unwrap();
    assert!(residuals.len() >= 2);
    for window in residuals.windows(2) {
        let prev = window[0];
        let next = window[1];
        assert!(
            next <= prev * (1.0 + 5e-13),
            "residual increased: {prev} -> {next}"
        );
    }

    let events = events.lock().unwrap();
    assert!(!events.is_empty());
    for evt in events.iter() {
        assert!(evt.p_ap > R::zero(), "nonpositive curvature: {}", evt.p_ap);
        assert!(evt.p_ap.is_finite());
        assert!(evt.alpha.is_finite());
        if let Some(beta) = evt.beta {
            assert!(beta.is_finite());
        }
    }

    assert_eq!(debug::large_imag_count(), 0);
}

#[derive(Clone)]
struct DiagonalOp {
    diag: Vec<S>,
}

impl KLinOp for DiagonalOp {
    type Scalar = S;

    fn dims(&self) -> (usize, usize) {
        (self.diag.len(), self.diag.len())
    }

    fn matvec_s(&self, x: &[S], y: &mut [S], _scratch: &mut BridgeScratch) {
        for (yi, (di, xi)) in y.iter_mut().zip(self.diag.iter().zip(x.iter())) {
            *yi = *di * *xi;
        }
    }
}

#[test]
fn cg_scalar_coefficients_are_real() {
    let _guard = cg_debug_lock().lock().unwrap();
    debug::reset_counters();

    let events = Arc::new(Mutex::new(Vec::<IterEvent>::new()));
    let event_sink = events.clone();
    let _hook = HookGuard::install(move |evt| {
        event_sink.lock().unwrap().push(evt);
    });

    let diag = vec![S::from_real(4.0), S::from_real(6.0), S::from_real(9.0)];
    let op = DiagonalOp { diag };
    let b = vec![S::from_real(1.0); 3];
    let mut x = vec![S::zero(); 3];
    let comm = UniverseComm::NoComm(NoComm);
    let mut solver = CgSolver::new(1e-12, 32);
    let mut work = Workspace::new(3);

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
        .expect("CG solve");

    assert!(matches!(
        stats.reason,
        ConvergedReason::ConvergedRtol | ConvergedReason::ConvergedAtol
    ));

    let events = events.lock().unwrap();
    for evt in events.iter() {
        assert!(evt.alpha.is_finite());
        if let Some(beta) = evt.beta {
            assert!(beta.is_finite());
        }
    }

    assert_eq!(debug::large_imag_count(), 0);
}

#[test]
fn cg_rejects_right_preconditioning_side() {
    let diag = vec![S::from_real(2.0)];
    let op = DiagonalOp { diag };
    let b = vec![S::from_real(1.0)];
    let mut x = vec![S::zero(); 1];
    let comm = UniverseComm::NoComm(NoComm);
    let mut solver = CgSolver::new(1e-6, 4);
    let mut work = Workspace::new(1);

    let err = solver
        .solve(
            &op,
            None,
            &b,
            &mut x,
            PcSide::Right,
            &comm,
            None,
            Some(&mut work),
        )
        .unwrap_err();

    assert!(matches!(err, kryst::error::KError::InvalidInput(_)));
}

#[test]
fn cg_detects_indefinite_matrix() {
    let diag = vec![S::zero(), S::from_real(1.0)];
    let op = DiagonalOp { diag };
    let b = vec![S::from_real(1.0), S::zero()];
    let mut x = vec![S::zero(); 2];
    let comm = UniverseComm::NoComm(NoComm);
    let mut solver = CgSolver::new(1e-6, 8);
    let mut work = Workspace::new(2);

    let err = solver
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
        .unwrap_err();

    assert!(matches!(err, kryst::error::KError::IndefiniteMatrix));
}
