#![cfg(feature = "backend-faer")]
#![allow(clippy::too_many_arguments)]

use std::sync::{Arc, Mutex};

use faer::Mat;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use kryst::algebra::bridge::BridgeScratch;
use kryst::algebra::prelude::*;
use kryst::context::ksp_context::Workspace;
use kryst::error::KError;
use kryst::ops::klinop::KLinOp;
use kryst::ops::kpc::KPreconditioner;
use kryst::parallel::{NoComm, UniverseComm, global_nrm2};
use kryst::preconditioner::PcSide;
use kryst::solver::cg::{CgNormType, CgSolver};
use kryst::utils::convergence::ConvergedReason;

// ---------- Dense S-native operator (test-only) ----------
struct DenseOpS {
    a: Mat<S>,
}

impl DenseOpS {
    fn new(a: Mat<S>) -> Self {
        Self { a }
    }

    fn n(&self) -> usize {
        self.a.nrows()
    }
}

impl KLinOp for DenseOpS {
    type Scalar = S;

    fn dims(&self) -> (usize, usize) {
        (self.a.nrows(), self.a.ncols())
    }

    fn matvec_s(&self, x: &[S], y: &mut [S], _scratch: &mut BridgeScratch) {
        debug_assert_eq!(x.len(), self.a.ncols());
        debug_assert_eq!(y.len(), self.a.nrows());

        for i in 0..self.a.nrows() {
            let mut acc = S::zero();
            for j in 0..self.a.ncols() {
                let aij = self.a[(i, j)];
                acc = acc + aij * x[j];
            }
            y[i] = acc;
        }
    }
}

// ---------- Optional: S-native Jacobi for tests ----------
struct JacobiS {
    diag_inv: Vec<S>,
}

impl JacobiS {
    fn from_dense(a: &Mat<S>) -> Result<Self, KError> {
        let n = a.nrows();
        let mut d = Vec::with_capacity(n);
        for i in 0..n {
            let aii = a[(i, i)];
            let den = aii.abs();
            if !den.is_finite() || den <= 1e-30 {
                return Err(KError::InvalidInput("JacobiS: near-zero diag".into()));
            }
            d.push(aii.inv());
        }
        Ok(Self { diag_inv: d })
    }
}

impl KPreconditioner for JacobiS {
    type Scalar = S;

    fn dims(&self) -> (usize, usize) {
        let n = self.diag_inv.len();
        (n, n)
    }

    fn apply_s(
        &self,
        side: PcSide,
        x: &[S],
        y: &mut [S],
        _scratch: &mut BridgeScratch,
    ) -> Result<(), KError> {
        if side != PcSide::Left {
            return Err(KError::InvalidInput("JacobiS supports only Left".into()));
        }
        debug_assert_eq!(x.len(), y.len());
        for (yi, (&xi, &dii)) in y.iter_mut().zip(x.iter().zip(self.diag_inv.iter())) {
            *yi = dii * xi;
        }
        Ok(())
    }
}

// ---------- RNG & builders ----------
fn rng() -> StdRng {
    StdRng::seed_from_u64(0xC0FFEE_u64)
}

#[inline]
fn rand_s<RNG: Rng>(rng: &mut RNG) -> S {
    #[cfg(feature = "complex")]
    {
        let re = rng.gen_range(-0.5..0.5);
        let im = rng.gen_range(-0.5..0.5);
        S::from_parts(re, im)
    }
    #[cfg(not(feature = "complex"))]
    {
        S::from_real(rng.gen_range(-0.5..0.5))
    }
}

fn random_mat_s(n: usize, rng: &mut StdRng) -> Mat<S> {
    let mut m = Mat::<S>::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            m[(i, j)] = rand_s(rng);
        }
    }
    m
}

fn random_vec_s(n: usize, rng: &mut StdRng) -> Vec<S> {
    (0..n).map(|_| rand_s(rng)).collect()
}

/// Build A = Bᴴ B + α I (HPD).
fn build_hpd(n: usize, alpha: R, rng: &mut StdRng) -> Mat<S> {
    let b = random_mat_s(n, rng);
    let mut c = Mat::<S>::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let mut acc = S::zero();
            for k in 0..n {
                let bik = b[(k, i)];
                let bkj = b[(k, j)];
                acc = acc + bik.conj() * bkj;
            }
            c[(i, j)] = acc;
        }
    }
    for i in 0..n {
        let aii = &mut c[(i, i)];
        *aii = *aii + S::from_real(alpha);
    }
    c
}

/// Build Hermitian indefinite: start HPD then flip one eigenvalue by subtracting τ * e_ie_iᴴ.
fn build_herm_indef(n: usize, alpha: R, tau: R, rng: &mut StdRng) -> Mat<S> {
    let mut a = build_hpd(n, alpha, rng);
    let i = n / 2;
    let aii = &mut a[(i, i)];
    *aii = *aii - S::from_real(tau);
    a
}

/// Apply A to x: y = A x
fn apply(a: &DenseOpS, x: &[S]) -> Vec<S> {
    let mut y = vec![S::zero(); a.n()];
    let mut scratch = BridgeScratch::new();
    a.matvec_s(x, &mut y, &mut scratch);
    y
}

// ---------------------------------------------------------
// 1) HPD system converges
#[test]
fn cg_hpd_converges() {
    let mut rng = rng();
    let n = 64usize;
    let alpha: R = 1e-2;
    let a_mat = build_hpd(n, alpha, &mut rng);
    let a = DenseOpS::new(a_mat);

    let x_star = random_vec_s(n, &mut rng);
    let b = apply(&a, &x_star);

    let comm = UniverseComm::NoComm(NoComm);
    let mut work = Workspace::new(n);
    let mut solver = CgSolver::new(1e-8, 5 * n);

    let mut x = vec![S::zero(); n];

    let stats = solver
        .solve_with_comm(
            &a,
            None,
            &b,
            &mut x,
            PcSide::Left,
            &comm,
            None,
            Some(&mut work),
        )
        .expect("CG solve");

    assert!(
        matches!(
            stats.reason,
            ConvergedReason::ConvergedRtol | ConvergedReason::ConvergedAtol
        ),
        "unexpected convergence reason: {:?} (iters={})",
        stats.reason,
        stats.iterations
    );
    assert!(
        stats.iterations <= 2 * n,
        "expected ≤ 2n iters, got {}",
        stats.iterations
    );

    let ax = apply(&a, &x);
    let mut true_r = vec![S::zero(); n];
    for i in 0..n {
        true_r[i] = b[i] - ax[i];
    }

    let err_x = {
        let mut diff = x.clone();
        for i in 0..n {
            diff[i] = diff[i] - x_star[i];
        }
        global_nrm2(&comm, &diff) / (global_nrm2(&comm, &x_star) + 1e-300)
    };
    let rel_r = global_nrm2(&comm, &true_r) / (global_nrm2(&comm, &b) + 1e-300);

    assert!(err_x <= 1e-8, "||x-x*||/||x*||={} not ≤ 1e-8", err_x);
    assert!(rel_r <= 1e-8, "||r||/||b||={} not ≤ 1e-8", rel_r);
}

// ---------------------------------------------------------
// 2) Left Jacobi monotone
#[test]
fn pcg_left_jacobi_monotone_precond() {
    let mut rng = rng();
    let n = 64usize;
    let alpha: R = 1e-2;
    let a_mat = build_hpd(n, alpha, &mut rng);
    let a = DenseOpS::new(a_mat.clone());

    let x_star = random_vec_s(n, &mut rng);
    let b = apply(&a, &x_star);

    let pc = JacobiS::from_dense(&a_mat).ok();
    let pc_ref: Option<&dyn KPreconditioner<Scalar = S>> =
        pc.as_ref().map(|p| p as &dyn KPreconditioner<Scalar = S>);

    let comm = UniverseComm::NoComm(NoComm);
    let mut work = Workspace::new(n);
    let mut solver = CgSolver::new(1e-8, 5 * n);
    solver.set_norm(CgNormType::Preconditioned);

    let hist = Arc::new(Mutex::new(Vec::<R>::new()));
    let hist_clone = hist.clone();
    let monitor = Box::new(move |_k: usize, r: R| {
        hist_clone.lock().unwrap().push(r);
    });
    let monitors: Vec<Box<dyn Fn(usize, R) + Send + Sync>> = vec![monitor];

    let mut x = vec![S::zero(); n];
    let stats = solver
        .solve_with_comm(
            &a,
            pc_ref,
            &b,
            &mut x,
            PcSide::Left,
            &comm,
            Some(&monitors),
            Some(&mut work),
        )
        .expect("PCG");

    assert!(
        matches!(
            stats.reason,
            ConvergedReason::ConvergedRtol | ConvergedReason::ConvergedAtol
        ),
        "unexpected convergence reason: {:?}",
        stats.reason
    );

    let h = hist.lock().unwrap();
    if let (Some(&first), Some(&last)) = (h.first(), h.last()) {
        assert!(
            last <= first * 1e-4,
            "preconditioned residual did not decrease enough: {} -> {}",
            first,
            last
        );
    }
}

// ---------------------------------------------------------
// 3) Side semantics
#[test]
fn cg_rejects_right_side() {
    let mut rng = rng();
    let n = 16usize;
    let a_mat = build_hpd(n, 1e-2, &mut rng);
    let a = DenseOpS::new(a_mat);
    let b = random_vec_s(n, &mut rng);

    let comm = UniverseComm::NoComm(NoComm);
    let mut work = Workspace::new(n);
    let mut solver = CgSolver::new(1e-8, n);

    let mut x = vec![S::zero(); n];
    let err = solver
        .solve_with_comm(
            &a,
            None,
            &b,
            &mut x,
            PcSide::Right,
            &comm,
            None,
            Some(&mut work),
        )
        .unwrap_err();

    assert!(matches!(err, KError::InvalidInput(_)));
}

// ---------------------------------------------------------
// 4) Indefinite guard
#[test]
fn cg_detects_indefinite_matrix() {
    let mut rng = rng();
    let n = 32usize;
    let a_mat = build_herm_indef(n, 1e-2, 2.0, &mut rng);
    let a = DenseOpS::new(a_mat);

    let b = random_vec_s(n, &mut rng);
    let comm = UniverseComm::NoComm(NoComm);
    let mut work = Workspace::new(n);
    let mut solver = CgSolver::new(1e-8, n);

    let mut x = vec![S::zero(); n];
    let err = solver
        .solve_with_comm(
            &a,
            None,
            &b,
            &mut x,
            PcSide::Left,
            &comm,
            None,
            Some(&mut work),
        )
        .unwrap_err();

    assert!(matches!(err, KError::IndefiniteMatrix));
}

// ---------------------------------------------------------
// 5) Norm type semantics
fn run_and_capture_norms(norm_type: CgNormType) -> Vec<R> {
    let mut rng = rng();
    let n = 8usize;
    let a_mat = build_hpd(n, 1e-2, &mut rng);
    let a = DenseOpS::new(a_mat);

    let b = random_vec_s(n, &mut rng);
    let comm = UniverseComm::NoComm(NoComm);
    let mut work = Workspace::new(n);
    let mut solver = CgSolver::new(1e-12, n);
    solver.set_norm(norm_type);

    let hist = Arc::new(Mutex::new(Vec::<R>::new()));
    let hist_clone = hist.clone();
    let monitor = Box::new(move |k: usize, r: R| {
        if k <= 2 {
            hist_clone.lock().unwrap().push(r);
        }
    });
    let monitors: Vec<Box<dyn Fn(usize, R) + Send + Sync>> = vec![monitor];

    let mut x = vec![S::zero(); n];
    let _ = solver.solve_with_comm(
        &a,
        None,
        &b,
        &mut x,
        PcSide::Left,
        &comm,
        Some(&monitors),
        Some(&mut work),
    );
    hist.lock().unwrap().clone()
}

#[test]
fn cg_norm_type_semantics_first_iters() {
    let norms_pre = run_and_capture_norms(CgNormType::Preconditioned);
    let norms_un = run_and_capture_norms(CgNormType::Unpreconditioned);
    let norms_nat = run_and_capture_norms(CgNormType::Natural);
    let norms_none = run_and_capture_norms(CgNormType::None);

    assert!(norms_none.iter().all(|&r| r == 0.0));

    assert!((norms_pre[0] - norms_un[0]).abs() <= 1e-12);
    assert!((norms_pre[0] - norms_nat[0]).abs() <= 1e-12);
}

// ---------------------------------------------------------
// 6) Trust region behavior
#[test]
fn cg_trust_region_caps_norm() {
    let mut rng = rng();
    let n = 64usize;
    let a_mat = build_hpd(n, 1e-2, &mut rng);
    let a = DenseOpS::new(a_mat);
    let b = random_vec_s(n, &mut rng);

    let comm = UniverseComm::NoComm(NoComm);
    let mut work = Workspace::new(n);
    let mut solver = CgSolver::new(1e-12, n);
    solver.set_trust_region(1e-2);

    let mut x = vec![S::zero(); n];
    let stats = solver
        .solve_with_comm(
            &a,
            None,
            &b,
            &mut x,
            PcSide::Left,
            &comm,
            None,
            Some(&mut work),
        )
        .expect("CG TR");

    assert!(matches!(
        stats.reason,
        ConvergedReason::ConvergedTrustRegion
    ));
    let xnorm = global_nrm2(&comm, &x);
    assert!((xnorm - 1e-2).abs() <= 1e-8, "||x||={} not ≈ 1e-2", xnorm);
}
