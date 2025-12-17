#![cfg(all(feature = "backend-faer", not(feature = "complex")))]
use kryst::algebra::blas::nrm2;
use kryst::algebra::bridge::BridgeScratch;
use kryst::algebra::prelude::*;
use kryst::assert_vec_close;
use kryst::context::ksp_context::Workspace;
use kryst::error::KError;
use kryst::matrix::op::CsrOp;
use kryst::matrix::sparse::CsrMatrix as RealCsrMatrix;
use kryst::ops::klinop::KLinOp;
use kryst::ops::kpc::KPreconditioner;
use kryst::ops::wrap::{as_s_op, as_s_pc};
use kryst::parallel::{NoComm, UniverseComm};
use kryst::preconditioner::Jacobi;
use kryst::preconditioner::amg::{AMGBuilder, RelaxType};
use kryst::preconditioner::ilu_csr::{IluCsr, IluCsrConfig};
use kryst::preconditioner::{PcSide, Preconditioner};
use kryst::solver::{BiCgStabSolver, CgSolver, GmresSolver, LinearSolver};

use faer::Mat;
use std::sync::Arc;
struct NativeDiagOp {
    diag: Vec<S>,
}

impl NativeDiagOp {
    fn new(diag: Vec<S>) -> Self {
        Self { diag }
    }
}

impl KLinOp for NativeDiagOp {
    type Scalar = S;

    fn dims(&self) -> (usize, usize) {
        let n = self.diag.len();
        (n, n)
    }

    fn matvec_s(&self, x: &[S], y: &mut [S], _scratch: &mut BridgeScratch) {
        debug_assert_eq!(x.len(), self.diag.len());
        debug_assert_eq!(y.len(), self.diag.len());
        for ((yi, &di), &xi) in y.iter_mut().zip(self.diag.iter()).zip(x.iter()) {
            *yi = di * xi;
        }
    }
}

struct NativeJacobiPc {
    inv_diag: Vec<S>,
}

impl NativeJacobiPc {
    fn new(diag: &[S]) -> Self {
        let inv_diag = diag.iter().map(|&d| d.inv()).collect();
        Self { inv_diag }
    }
}

impl KPreconditioner for NativeJacobiPc {
    type Scalar = S;

    fn dims(&self) -> (usize, usize) {
        let n = self.inv_diag.len();
        (n, n)
    }

    fn apply_s(
        &self,
        _side: PcSide,
        x: &[S],
        y: &mut [S],
        _scratch: &mut BridgeScratch,
    ) -> Result<(), KError> {
        if x.len() != self.inv_diag.len() || y.len() != self.inv_diag.len() {
            return Err(KError::InvalidInput(format!(
                "NativeJacobiPc::apply_s dimension mismatch: n={}, x.len()={}, y.len()={}",
                self.inv_diag.len(),
                x.len(),
                y.len()
            )));
        }
        for ((yi, &wi), &xi) in y.iter_mut().zip(self.inv_diag.iter()).zip(x.iter()) {
            *yi = wi * xi;
        }
        Ok(())
    }
}

struct JacobiF64 {
    // INTENTIONAL f64: exercises the legacy preconditioner API.
    inv_diag: Vec<f64>,
}

impl JacobiF64 {
    fn new(inv_diag: Vec<f64>) -> Self {
        Self { inv_diag }
    }
}

impl Preconditioner for JacobiF64 {
    fn dims(&self) -> (usize, usize) {
        let n = self.inv_diag.len();
        (n, n)
    }

    fn setup(&mut self, _a: &dyn kryst::matrix::op::LinOp<S = f64>) -> Result<(), KError> {
        Ok(())
    }

    fn apply(&self, _side: PcSide, x: &[f64], y: &mut [f64]) -> Result<(), KError> {
        debug_assert_eq!(x.len(), self.inv_diag.len());
        debug_assert_eq!(y.len(), self.inv_diag.len());
        for ((yi, &wi), &xi) in y.iter_mut().zip(self.inv_diag.iter()).zip(x.iter()) {
            *yi = wi * xi;
        }
        Ok(())
    }
}

fn cg_reference_system() -> (Vec<R>, Vec<R>) {
    let four = S::from_real(4.0).real();
    let five = S::from_real(5.0).real();
    let six = S::from_real(6.0).real();
    let diag = vec![four, five, six];
    let x_true = vec![
        S::from_real(1.0).real(),
        S::from_real(-1.0).real(),
        S::from_real(2.0).real(),
    ];
    (diag, x_true)
}

#[test]
fn cg_runs_with_native_and_wrapped_backends() {
    let (diag, x_true) = cg_reference_system();
    let n = diag.len();

    let diag_s: Vec<S> = diag.iter().copied().map(S::from_real).collect();
    let b_native: Vec<S> = diag_s
        .iter()
        .zip(x_true.iter())
        .map(|(&d, &x)| d * S::from_real(x))
        .collect();

    let mut x_native = vec![S::zero(); n];
    let mut solver_native = CgSolver::new(1e-12, 32);
    let comm = UniverseComm::NoComm(NoComm);
    let mut workspace_native = Workspace::new(n);
    solver_native.setup_workspace(&mut workspace_native);
    let op_native = NativeDiagOp::new(diag_s.clone());

    let stats_native = solver_native
        .solve(
            &op_native,
            None,
            &b_native,
            &mut x_native,
            PcSide::Left,
            &comm,
            None,
            Some(&mut workspace_native),
        )
        .expect("native CG solve");
    assert!(matches!(
        stats_native.reason,
        kryst::utils::convergence::ConvergedReason::ConvergedRtol
            | kryst::utils::convergence::ConvergedReason::ConvergedAtol
    ));
    let mut scratch = BridgeScratch::default();
    let mut ax = vec![S::zero(); n];
    op_native.matvec_s(&x_native, &mut ax, &mut scratch);
    for (ai, bi) in ax.iter_mut().zip(b_native.iter()) {
        *ai -= *bi;
    }
    assert!(nrm2(&ax) < 1e-10);

    let zero = S::zero().real();
    let mat = Mat::<f64>::from_fn(n, n, |i, j| if i == j { diag[i] } else { zero });
    let b_real: Vec<R> = diag
        .iter()
        .zip(x_true.iter())
        .map(|(&d, &x)| d * x)
        .collect();
    let b_wrapped: Vec<S> = b_real.iter().copied().map(S::from_real).collect();
    let mut x_wrapped = vec![S::zero(); n];
    let mut solver_wrapped = CgSolver::new(1e-12, 32);
    let mut workspace_wrapped = Workspace::new(n);
    solver_wrapped.setup_workspace(&mut workspace_wrapped);
    let op_wrapped = as_s_op(&mat);
    let stats_wrapped = solver_wrapped
        .solve(
            &op_wrapped,
            None,
            &b_wrapped,
            &mut x_wrapped,
            PcSide::Left,
            &comm,
            None,
            Some(&mut workspace_wrapped),
        )
        .expect("wrapped CG solve");
    assert!(matches!(
        stats_wrapped.reason,
        kryst::utils::convergence::ConvergedReason::ConvergedRtol
            | kryst::utils::convergence::ConvergedReason::ConvergedAtol
    ));
    let mut scratch_wrapped = BridgeScratch::default();
    let mut ax_wrapped = vec![S::zero(); n];
    op_wrapped.matvec_s(&x_wrapped, &mut ax_wrapped, &mut scratch_wrapped);
    for (ai, bi) in ax_wrapped.iter_mut().zip(b_wrapped.iter()) {
        *ai -= *bi;
    }
    assert!(nrm2(&ax_wrapped) < 1e-10);

    let one = S::from_real(1.0).real();
    let inv_diag: Vec<f64> = diag.iter().map(|&d| one / d).collect();
    let pc_f64 = JacobiF64::new(inv_diag);
    let pc_wrapped = as_s_pc(&pc_f64);
    let mut pc_out = vec![S::zero(); n];
    let mut pc_scratch = BridgeScratch::default();
    pc_wrapped
        .apply_s(PcSide::Left, &b_wrapped, &mut pc_out, &mut pc_scratch)
        .expect("pc apply via bridge");
    for (i, (yi, bi)) in pc_out.iter().zip(b_wrapped.iter()).enumerate() {
        assert!(yi.real().is_finite(), "pc output {i} should be finite");
        assert!(bi.real().is_finite());
    }
}

#[test]
fn gmres_runs_with_native_s_backend() {
    let diag_s: Vec<S> = vec![
        S::from_real(4.0),
        S::from_real(5.0),
        S::from_real(6.0),
        S::from_real(7.0),
    ];
    let x_true: Vec<S> = vec![
        S::from_real(1.0),
        S::from_real(-1.0),
        S::from_real(2.0),
        S::from_real(0.5),
    ];
    let b: Vec<S> = diag_s
        .iter()
        .zip(x_true.iter())
        .map(|(&d, &x)| d * x)
        .collect();
    let mut x = vec![S::zero(); diag_s.len()];

    let mut gmres = GmresSolver::new(8, 1e-12, 64);
    let comm = UniverseComm::NoComm(NoComm);
    let mut workspace = Workspace::new(diag_s.len());
    gmres.setup_workspace(&mut workspace);

    let op = NativeDiagOp::new(diag_s.clone());
    let pc = NativeJacobiPc::new(&diag_s);

    let _stats = gmres
        .solve(
            &op,
            Some(&pc),
            &b,
            &mut x,
            PcSide::Left,
            &comm,
            None,
            Some(&mut workspace),
        )
        .expect("native GMRES solve");

    let mut scratch = BridgeScratch::default();
    let mut ax = vec![S::zero(); diag_s.len()];
    op.matvec_s(&x, &mut ax, &mut scratch);
    for (ai, bi) in ax.iter_mut().zip(b.iter()) {
        *ai -= *bi;
    }
    assert!(nrm2(&ax) < 1e-10);
}

#[test]
fn gmres_runs_with_wrapped_f64_backends() {
    let diag = vec![
        S::from_real(3.0).real(),
        S::from_real(4.0).real(),
        S::from_real(5.0).real(),
    ];
    let x_true = vec![
        S::from_real(1.0).real(),
        S::from_real(-2.0).real(),
        S::from_real(0.5).real(),
    ];
    let n = diag.len();

    let row_ptr: Vec<usize> = (0..=n).collect();
    let col_idx: Vec<usize> = (0..n).collect();
    let values = diag.clone();
    let csr = Arc::new(RealCsrMatrix::from_csr(n, n, row_ptr, col_idx, values));
    let op_f64 = CsrOp::new(csr);

    let mut jacobi = Jacobi::new();
    jacobi.setup(&op_f64).expect("jacobi setup");

    let op = as_s_op(&op_f64);
    let pc = as_s_pc(&jacobi);

    let b: Vec<S> = diag
        .iter()
        .zip(x_true.iter())
        .map(|(&d, &x)| S::from_real(d * x))
        .collect();
    let mut x = vec![S::zero(); n];

    let mut gmres = GmresSolver::new(6, 1e-12, 64);
    let comm = UniverseComm::NoComm(NoComm);
    let mut workspace = Workspace::new(n);
    gmres.setup_workspace(&mut workspace);

    let _stats = gmres
        .solve(
            &op,
            Some(&pc),
            &b,
            &mut x,
            PcSide::Left,
            &comm,
            None,
            Some(&mut workspace),
        )
        .expect("wrapped GMRES solve");

    let mut scratch = BridgeScratch::default();
    let mut ax = vec![S::zero(); n];
    op.matvec_s(&x, &mut ax, &mut scratch);
    for (ai, bi) in ax.iter_mut().zip(b.iter()) {
        *ai -= *bi;
    }
    assert!(nrm2(&ax) < 1e-10);
}

#[test]
fn jacobi_preconditioner_exposes_scalar_generic_bridge() {
    let diag = vec![
        S::from_real(2.0).real(),
        S::from_real(3.0).real(),
        S::from_real(5.0).real(),
    ];
    let n = diag.len();
    let zero = S::zero().real();
    let mat = Mat::<f64>::from_fn(n, n, |i, j| if i == j { diag[i] } else { zero });

    let mut jacobi = Jacobi::new();
    jacobi.setup(&mat).expect("jacobi setup");
    assert_eq!(<Jacobi as KPreconditioner>::dims(&jacobi), (n, n));

    let rhs_s: Vec<S> = (0..n).map(|i| S::from_real((i + 1) as f64)).collect();
    let mut out_s = vec![S::zero(); n];
    let mut scratch = BridgeScratch::default();
    <Jacobi as KPreconditioner>::apply_s(&jacobi, PcSide::Left, &rhs_s, &mut out_s, &mut scratch)
        .expect("jacobi apply_s");

    let rhs_real: Vec<R> = rhs_s.iter().map(|v| v.real()).collect();
    let mut out_real: Vec<R> = vec![R::default(); n];
    jacobi
        .apply(PcSide::Left, &rhs_real, &mut out_real)
        .expect("jacobi apply reference");

    let expected: Vec<S> = out_real.iter().copied().map(S::from_real).collect();
    assert_vec_close!("jacobi bridge compare", &out_s, &expected);
}

#[test]
fn ilucsr_exposes_kpreconditioner_interface() {
    let n = 3;
    let row_ptr = vec![0, 1, 2, 3];
    let col_idx = vec![0, 1, 2];
    let values = vec![
        S::from_real(4.0).real(),
        S::from_real(5.0).real(),
        S::from_real(6.0).real(),
    ];
    let csr = Arc::new(RealCsrMatrix::from_csr(n, n, row_ptr, col_idx, values));
    let op = CsrOp::new(csr.clone());

    let mut ilu = IluCsr::new_with_config(IluCsrConfig::default());
    ilu.setup(&op).expect("IluCsr setup");

    let dims = KPreconditioner::dims(&ilu);
    assert_eq!(dims, (n, n));

    let rhs: Vec<S> = (0..n).map(|i| S::from_real((i + 2) as f64)).collect();
    let mut out = vec![S::zero(); n];
    let mut scratch = BridgeScratch::default();
    KPreconditioner::apply_s(&ilu, PcSide::Left, &rhs, &mut out, &mut scratch)
        .expect("IluCsr apply_s");

    let rhs_r: Vec<R> = rhs.iter().map(|v| v.real()).collect();
    let mut out_r: Vec<R> = vec![R::default(); n];
    ilu.apply(PcSide::Left, &rhs_r, &mut out_r)
        .expect("IluCsr apply reference");

    let expected: Vec<S> = out_r.iter().copied().map(S::from_real).collect();
    assert_vec_close!("ilu bridge compare", &out, &expected);
}

struct NativeMatrixOp {
    n: usize,
    data: Vec<S>,
}

impl NativeMatrixOp {
    fn new(n: usize, data: Vec<S>) -> Self {
        debug_assert_eq!(data.len(), n * n);
        Self { n, data }
    }
}

impl KLinOp for NativeMatrixOp {
    type Scalar = S;

    fn dims(&self) -> (usize, usize) {
        (self.n, self.n)
    }

    fn matvec_s(&self, x: &[S], y: &mut [S], _scratch: &mut BridgeScratch) {
        debug_assert_eq!(x.len(), self.n);
        debug_assert_eq!(y.len(), self.n);
        for i in 0..self.n {
            let mut sum = S::zero();
            for j in 0..self.n {
                sum = self.data[i * self.n + j].mul_add(x[j], sum);
            }
            y[i] = sum;
        }
    }
}

fn bicg_reference_system() -> (Mat<f64>, Vec<R>) {
    let four = S::from_real(4.0).real();
    let three = S::from_real(3.0).real();
    let two = S::from_real(2.0).real();
    let one = S::from_real(1.0).real();
    let minus_two = S::from_real(-2.0).real();
    let a = Mat::<f64>::from_fn(2, 2, |i, j| match (i, j) {
        (0, 0) => four,
        (0, 1) => one,
        (1, 0) => two,
        (1, 1) => three,
        _ => unreachable!(),
    });
    let x_true = vec![one, minus_two];
    (a, x_true)
}

#[test]
fn bicgstab_runs_with_native_and_wrapped_backends() {
    let (a_f64, x_true) = bicg_reference_system();
    let n = x_true.len();
    let four = S::from_real(4.0);
    let three = S::from_real(3.0);
    let two = S::from_real(2.0);
    let one = S::from_real(1.0);
    let data_s: Vec<S> = (0..n * n)
        .map(|idx| {
            let i = idx / n;
            let j = idx % n;
            match (i, j) {
                (0, 0) => four,
                (0, 1) => one,
                (1, 0) => two,
                (1, 1) => three,
                _ => unreachable!(),
            }
        })
        .collect();
    let x_true_s: Vec<S> = x_true.iter().copied().map(S::from_real).collect();
    let b_native: Vec<S> = {
        let mut tmp = vec![S::zero(); n];
        let op = NativeMatrixOp::new(n, data_s.clone());
        let mut scratch = BridgeScratch::default();
        op.matvec_s(&x_true_s, &mut tmp, &mut scratch);
        tmp
    };
    let comm = UniverseComm::NoComm(NoComm);
    let mut solver_native = BiCgStabSolver::new(1e-12, 64);
    let mut workspace_native = Workspace::new(n);
    solver_native.setup_workspace(&mut workspace_native);
    let op_native = NativeMatrixOp::new(n, data_s.clone());
    let mut x_native = vec![S::zero(); n];
    let stats_native = solver_native
        .solve(
            &op_native,
            None,
            &b_native,
            &mut x_native,
            PcSide::Left,
            &comm,
            None,
            Some(&mut workspace_native),
        )
        .expect("native BiCGStab solve");
    assert!(matches!(
        stats_native.reason,
        kryst::utils::convergence::ConvergedReason::ConvergedRtol
            | kryst::utils::convergence::ConvergedReason::ConvergedAtol
    ));
    let mut scratch_native = BridgeScratch::default();
    let mut ax_native = vec![S::zero(); n];
    op_native.matvec_s(&x_native, &mut ax_native, &mut scratch_native);
    for (ai, bi) in ax_native.iter_mut().zip(b_native.iter()) {
        *ai -= *bi;
    }
    assert!(nrm2(&ax_native) < 1e-10);

    let op_wrapped = as_s_op(&a_f64);
    let b_wrapped: Vec<S> = {
        let mut tmp = vec![S::zero(); n];
        let mut scratch = BridgeScratch::default();
        op_wrapped.matvec_s(&x_true_s, &mut tmp, &mut scratch);
        tmp
    };
    let mut solver_wrapped = BiCgStabSolver::new(1e-12, 64);
    let mut workspace_wrapped = Workspace::new(n);
    solver_wrapped.setup_workspace(&mut workspace_wrapped);
    let mut x_wrapped = vec![S::zero(); n];
    let stats_wrapped = solver_wrapped
        .solve(
            &op_wrapped,
            None,
            &b_wrapped,
            &mut x_wrapped,
            PcSide::Left,
            &comm,
            None,
            Some(&mut workspace_wrapped),
        )
        .expect("wrapped BiCGStab solve");
    assert!(matches!(
        stats_wrapped.reason,
        kryst::utils::convergence::ConvergedReason::ConvergedRtol
            | kryst::utils::convergence::ConvergedReason::ConvergedAtol
    ));
    let mut scratch_wrapped = BridgeScratch::default();
    let mut ax_wrapped = vec![S::zero(); n];
    op_wrapped.matvec_s(&x_wrapped, &mut ax_wrapped, &mut scratch_wrapped);
    for (ai, bi) in ax_wrapped.iter_mut().zip(b_wrapped.iter()) {
        *ai -= *bi;
    }
    assert!(nrm2(&ax_wrapped) < 1e-10);
}

#[test]
fn amg_exposes_kpreconditioner_interface() {
    let n = 6;
    let two = S::from_real(2.0).real();
    let minus_one = S::from_real(-1.0).real();
    let zero = S::zero().real();
    let laplacian = Mat::<f64>::from_fn(n, n, |i, j| {
        if i == j {
            two
        } else if (i as isize - j as isize).abs() == 1 {
            minus_one
        } else {
            zero
        }
    });

    let mut amg = AMGBuilder::new()
        .relaxation_type(RelaxType::Jacobi)
        .grid_relax_type_all(RelaxType::Jacobi)
        .build(&Mat::<f64>::zeros(0, 0))
        .expect("AMG build");
    amg.setup(&laplacian).expect("AMG setup");

    let dims = KPreconditioner::dims(&amg);
    assert_eq!(dims, (n, n));

    let n_real = S::from_real(n as f64).real();
    let rhs: Vec<S> = (0..n)
        .map(|i| {
            let numerator = S::from_real(i as f64 + 1.0).real();
            S::from_real(numerator / n_real)
        })
        .collect();
    let mut out = vec![S::zero(); n];
    let mut scratch = BridgeScratch::default();
    KPreconditioner::apply_s(&amg, PcSide::Left, &rhs, &mut out, &mut scratch)
        .expect("AMG apply_s");

    let rhs_r: Vec<R> = rhs.iter().map(|z| z.real()).collect();
    let mut out_r: Vec<R> = vec![R::default(); n];
    amg.apply(PcSide::Left, &rhs_r, &mut out_r)
        .expect("AMG apply");

    let expected: Vec<S> = out_r.iter().copied().map(S::from_real).collect();
    assert_vec_close!("amg bridge compare", &out, &expected);
}
