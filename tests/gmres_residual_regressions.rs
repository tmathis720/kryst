use std::sync::{Arc, Mutex};

use kryst::algebra::prelude::*;
use kryst::context::ksp_context::{KspContext, SolverType};
use kryst::context::pc_context::PcType;
use kryst::matrix::LinOp;
use kryst::matrix::op::CsrOp;
use kryst::matrix::sparse::CsrMatrix as RealCsrMatrix;
use kryst::solver::MonitorAction;

struct DenseOp {
    n: usize,
    a: Vec<S>,
}

impl DenseOp {
    fn new(n: usize, a: Vec<S>) -> Self {
        Self { n, a }
    }

    fn matvec_local(&self, x: &[S], y: &mut [S]) {
        for i in 0..self.n {
            let mut sum = S::zero();
            for j in 0..self.n {
                sum += self.a[i * self.n + j] * x[j];
            }
            y[i] = sum;
        }
    }
}

impl LinOp for DenseOp {
    type S = S;

    fn dims(&self) -> (usize, usize) {
        (self.n, self.n)
    }

    fn matvec(&self, x: &[S], y: &mut [S]) {
        self.matvec_local(x, y);
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[test]
fn gmres_dense_nonsymmetric_full_cycle_matches_lu_reference_accuracy() {
    let n = 8usize;
    let mut a = vec![S::zero(); n * n];
    for i in 0..n {
        a[i * n + i] = S::from_real(3.5 + 0.2 * i as f64);
        if i + 1 < n {
            a[i * n + (i + 1)] = S::from_real(-0.4 - 0.1 * i as f64);
            a[(i + 1) * n + i] = S::from_real(0.25 + 0.05 * i as f64);
        }
        if i + 2 < n {
            a[i * n + (i + 2)] = S::from_real(0.12);
        }
    }
    a[0 * n + 3] = S::from_real(-0.31);
    a[5 * n + 1] = S::from_real(0.27);

    let op = Arc::new(DenseOp::new(n, a));
    let x_true: Vec<S> = (0..n)
        .map(|i| S::from_real(((i as f64) * 0.37).sin()))
        .collect();
    let mut b = vec![S::zero(); n];
    op.matvec(&x_true, &mut b);

    let mut x = vec![S::zero(); n];
    let mut ksp = KspContext::new();
    ksp.set_type(SolverType::Gmres).expect("set gmres");
    ksp.set_pc_type(PcType::None, None).expect("set pc none");
    ksp.set_tolerances(1e-12, 1e-14, 1e6, n);
    ksp.set_restart(n);
    ksp.set_operators(op.clone(), None);

    let stats = ksp.solve(&b, &mut x).expect("gmres solve");
    assert!(
        stats.final_residual < 1e-10,
        "final residual={}",
        stats.final_residual
    );

    let mut ax = vec![S::zero(); n];
    op.matvec(&x, &mut ax);
    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..n {
        let ri = (b[i] - ax[i]).real();
        num += ri * ri;
        let bi = b[i].real();
        den += bi * bi;
    }
    let rel = num.sqrt() / den.sqrt().max(1e-32);
    assert!(rel < 1e-10, "relative residual={rel}");
}

#[test]
fn gmres_sparse_no_pc_reports_true_residual_in_stats() {
    let n = 6usize;
    let row_ptr = vec![0, 2, 5, 8, 11, 14, 16];
    let col_idx = vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5];
    let vals = vec![
        5.0, -0.3, 0.2, 5.4, -0.5, 0.1, 5.2, -0.4, 0.3, 5.1, -0.6, 0.2, 5.3, -0.4, 0.1, 5.0,
    ];

    let csr = Arc::new(RealCsrMatrix::from_csr(n, n, row_ptr, col_idx, vals));
    let op = Arc::new(CsrOp::new(csr));

    let x_true: Vec<S> = (0..n)
        .map(|i| S::from_real(1.0 + (i as f64) * 0.2))
        .collect();
    let mut b = vec![S::zero(); n];
    // Build RHS explicitly in the real backend and convert into `S`.
    {
        let dense_like = op.clone();
        let x_real: Vec<f64> = x_true.iter().map(|v| v.real()).collect();
        let mut b_real = vec![0.0; n];
        dense_like.matvec(&x_real, &mut b_real);
        for i in 0..n {
            b[i] = S::from_real(b_real[i]);
        }
    }

    let mut x = vec![S::zero(); n];
    let mut ksp = KspContext::new();
    ksp.set_type(SolverType::Gmres).expect("set gmres");
    ksp.set_pc_type(PcType::None, None).expect("set pc none");
    ksp.set_tolerances(1e-14, 1e-16, 1e6, n);
    ksp.set_restart(n);
    ksp.set_operators(op.clone(), None);

    let stats = ksp.solve(&b, &mut x).expect("gmres solve");

    let mut ax = vec![S::zero(); n];
    let x_real: Vec<f64> = x.iter().map(|v| v.real()).collect();
    let mut ax_real = vec![0.0; n];
    op.matvec(&x_real, &mut ax_real);
    for i in 0..n {
        ax[i] = S::from_real(ax_real[i]);
    }
    let mut num = 0.0;
    for i in 0..n {
        let ri = (b[i] - ax[i]).real();
        num += ri * ri;
    }
    let true_residual = num.sqrt();
    let rel_gap = (true_residual - stats.final_residual).abs() / true_residual.max(1e-32);
    assert!(
        rel_gap < 1e-10,
        "stats must track true residual, rel gap={rel_gap:e}"
    );
}

#[test]
fn gmres_restart_boundary_update_reduces_residual() {
    let n = 5usize;
    let a = vec![
        4.0, -1.0, 0.0, 0.5, 0.0, 0.3, 3.8, -0.7, 0.0, 0.2, 0.0, 0.8, 3.5, -1.1, 0.0, 0.1, 0.0,
        0.6, 4.1, -0.9, 0.0, 0.2, 0.0, 0.4, 3.9,
    ]
    .into_iter()
    .map(S::from_real)
    .collect();
    let op = Arc::new(DenseOp::new(n, a));

    let x_true: Vec<S> = vec![1.0, -0.5, 0.75, -1.2, 0.4]
        .into_iter()
        .map(S::from_real)
        .collect();
    let mut b = vec![S::zero(); n];
    op.matvec(&x_true, &mut b);

    let mut x = vec![S::zero(); n];
    let mut ksp = KspContext::new();
    ksp.set_type(SolverType::Gmres).expect("set gmres");
    ksp.set_pc_type(PcType::None, None).expect("set pc none");
    ksp.set_tolerances(1e-10, 1e-14, 1e6, 4);
    ksp.set_restart(2);
    ksp.set_operators(op, None);

    let history = Arc::new(Mutex::new(Vec::<f64>::new()));
    let history_clone = history.clone();
    ksp.add_monitor(Box::new(move |_it, residual, _reductions| {
        history_clone.lock().expect("lock history").push(residual);
        MonitorAction::Continue
    }));

    let _stats = ksp.solve(&b, &mut x).expect("gmres solve");
    let history = history.lock().expect("lock history");
    assert!(history.len() >= 2, "need at least two monitor entries");
    let initial = history[0];
    let final_monitor = *history.last().expect("last history");
    assert!(
        final_monitor < initial,
        "restart-boundary update failed to reduce residual: initial={initial:e}, final={final_monitor:e}"
    );
}
