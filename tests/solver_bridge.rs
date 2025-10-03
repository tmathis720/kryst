use kryst::algebra::blas::nrm2;
use kryst::algebra::bridge::BridgeScratch;
use kryst::algebra::prelude::*;
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
use kryst::solver::{BiCgStabSolver, CgSolver, LinearSolver};

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

struct JacobiF64 {
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

fn cg_reference_system() -> (Vec<f64>, Vec<f64>) {
    let diag = vec![4.0, 5.0, 6.0];
    let x_true = vec![1.0, -1.0, 2.0];
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

    let mat = Mat::<f64>::from_fn(n, n, |i, j| if i == j { diag[i] } else { 0.0 });
    let b_f64: Vec<f64> = diag
        .iter()
        .zip(x_true.iter())
        .map(|(&d, &x)| d * x)
        .collect();
    let b_wrapped: Vec<S> = b_f64.iter().copied().map(S::from_real).collect();
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

    let pc_f64 = JacobiF64::new(diag.iter().map(|&d| 1.0 / d).collect());
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
fn jacobi_preconditioner_exposes_scalar_generic_bridge() {
    let diag = vec![2.0, 3.0, 5.0];
    let n = diag.len();
    let mat = Mat::<f64>::from_fn(n, n, |i, j| if i == j { diag[i] } else { 0.0 });

    let mut jacobi = Jacobi::new();
    jacobi.setup(&mat).expect("jacobi setup");
    assert_eq!(<Jacobi as KPreconditioner>::dims(&jacobi), (n, n));

    let rhs_s: Vec<S> = (0..n).map(|i| S::from_real((i + 1) as f64)).collect();
    let mut out_s = vec![S::zero(); n];
    let mut scratch = BridgeScratch::default();
    <Jacobi as KPreconditioner>::apply_s(&jacobi, PcSide::Left, &rhs_s, &mut out_s, &mut scratch)
        .expect("jacobi apply_s");

    let rhs_real: Vec<f64> = rhs_s.iter().map(|v| v.real()).collect();
    let mut out_real = vec![0.0; n];
    jacobi
        .apply(PcSide::Left, &rhs_real, &mut out_real)
        .expect("jacobi apply reference");

    for (ys, yr) in out_s.iter().zip(out_real.iter()) {
        assert!((ys.real() - yr).abs() < 1e-12);
    }
}

#[test]
fn ilucsr_exposes_kpreconditioner_interface() {
    let n = 3;
    let row_ptr = vec![0, 1, 2, 3];
    let col_idx = vec![0, 1, 2];
    let values = vec![4.0, 5.0, 6.0];
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

    let rhs_r: Vec<f64> = rhs.iter().map(|v| v.real()).collect();
    let mut out_r = vec![0.0; n];
    ilu.apply(PcSide::Left, &rhs_r, &mut out_r)
        .expect("IluCsr apply reference");

    for (ys, yr) in out.iter().zip(out_r.iter()) {
        assert!((ys.real() - yr).abs() < 1e-12);
    }
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

fn bicg_reference_system() -> (Mat<f64>, Vec<f64>) {
    let a = Mat::<f64>::from_fn(2, 2, |i, j| match (i, j) {
        (0, 0) => 4.0,
        (0, 1) => 1.0,
        (1, 0) => 2.0,
        (1, 1) => 3.0,
        _ => unreachable!(),
    });
    let x_true = vec![1.0, -2.0];
    (a, x_true)
}

#[test]
fn bicgstab_runs_with_native_and_wrapped_backends() {
    let (a_f64, x_true) = bicg_reference_system();
    let n = x_true.len();
    let data_s: Vec<S> = (0..n * n)
        .map(|idx| {
            let i = idx / n;
            let j = idx % n;
            let val = match (i, j) {
                (0, 0) => 4.0,
                (0, 1) => 1.0,
                (1, 0) => 2.0,
                (1, 1) => 3.0,
                _ => unreachable!(),
            };
            S::from_real(val)
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
    let laplacian = Mat::<f64>::from_fn(n, n, |i, j| {
        if i == j {
            2.0
        } else if (i as isize - j as isize).abs() == 1 {
            -1.0
        } else {
            0.0
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

    let rhs: Vec<S> = (0..n)
        .map(|i| S::from_real((i as f64 + 1.0) / (n as f64)))
        .collect();
    let mut out = vec![S::zero(); n];
    let mut scratch = BridgeScratch::default();
    KPreconditioner::apply_s(&amg, PcSide::Left, &rhs, &mut out, &mut scratch)
        .expect("AMG apply_s");

    let rhs_r: Vec<f64> = rhs.iter().map(|z| z.real()).collect();
    let mut out_r = vec![0.0; n];
    amg.apply(PcSide::Left, &rhs_r, &mut out_r)
        .expect("AMG apply");

    for (ys, &yr) in out.iter().zip(out_r.iter()) {
        assert!((ys.real() - yr).abs() <= 1e-12);
    }
}
