use faer::Mat;
use kryst::algebra::prelude::*;
use kryst::context::ksp_context::{KspContext, SolverType};
use kryst::context::pc_context::PcType;
use kryst::error::KError;
use kryst::matrix::format::AsFormat;
use kryst::matrix::op::{DenseOp, LinOp};
use kryst::matrix::sparse::CsrMatrix;
use kryst::matrix::{CsrOp, convert::csr_from_linop};
use kryst::preconditioner::{PcReusePolicy, PcSide, Preconditioner};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

struct CountPc {
    numeric: Arc<AtomicUsize>,
    symbolic: Arc<AtomicUsize>,
}
impl CountPc {
    fn new() -> (Self, Arc<AtomicUsize>, Arc<AtomicUsize>) {
        let n = Arc::new(AtomicUsize::new(0));
        let s = Arc::new(AtomicUsize::new(0));
        (
            Self {
                numeric: n.clone(),
                symbolic: s.clone(),
            },
            n,
            s,
        )
    }
}
impl Preconditioner for CountPc {
    fn setup(&mut self, _a: &dyn LinOp<S = f64>) -> Result<(), KError> {
        Ok(())
    }
    fn apply(&self, _side: PcSide, x: &[f64], y: &mut [f64]) -> Result<(), KError> {
        y.copy_from_slice(x);
        Ok(())
    }
    fn supports_numeric_update(&self) -> bool {
        true
    }
    fn update_numeric(&mut self, _a: &dyn LinOp<S = f64>) -> Result<(), KError> {
        self.numeric.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
    fn update_symbolic(&mut self, _a: &dyn LinOp<S = f64>) -> Result<(), KError> {
        self.symbolic.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

#[derive(Default)]
struct PatternCheckPc {
    row_ptr: Vec<usize>,
    col_idx: Vec<usize>,
}

impl PatternCheckPc {
    /// Extract a CSR-like sparsity pattern from a LinOp in a stable way.
    fn pattern_from_linop(a: &dyn LinOp<S = f64>) -> Result<(Vec<usize>, Vec<usize>), KError> {
        // Direct CSR => copy the pattern.
        if let Some(csr) = a.as_any().downcast_ref::<CsrMatrix<f64>>() {
            return Ok((csr.row_ptr().to_vec(), csr.col_idx().to_vec()));
        }

        // Direct Mat => build pattern deterministically.
        #[cfg(feature = "backend-faer")]
        if let Some(d) = a.as_any().downcast_ref::<Mat<f64>>() {
            let m = d.nrows();
            let n = d.ncols();
            let mut row_ptr = Vec::with_capacity(m + 1);
            let mut col_idx = Vec::new();
            row_ptr.push(0);
            for i in 0..m {
                for j in 0..n {
                    if d[(i, j)] != 0.0 {
                        col_idx.push(j);
                    }
                }
                row_ptr.push(col_idx.len());
            }
            return Ok((row_ptr, col_idx));
        }

        // Fallback to the generic converter.
        let csr = csr_from_linop(a, R::default())?;
        Ok((csr.row_ptr().to_vec(), csr.col_idx().to_vec()))
    }
}

impl Preconditioner for PatternCheckPc {
    fn setup(&mut self, a: &dyn LinOp<S = f64>) -> Result<(), KError> {
        let (row_ptr, col_idx) = Self::pattern_from_linop(a)?;
        self.row_ptr = row_ptr;
        self.col_idx = col_idx;
        Ok(())
    }

    fn apply(&self, _side: PcSide, x: &[f64], y: &mut [f64]) -> Result<(), KError> {
        y.copy_from_slice(x);
        Ok(())
    }

    fn supports_numeric_update(&self) -> bool {
        true
    }

    fn update_numeric(&mut self, a: &dyn LinOp<S = f64>) -> Result<(), KError> {
        let (row_ptr, col_idx) = Self::pattern_from_linop(a)?;
        if self.row_ptr != row_ptr || self.col_idx != col_idx {
            return Err(KError::Unsupported("pattern changed; need update_symbolic"));
        }
        Ok(())
    }

    fn update_symbolic(&mut self, a: &dyn LinOp<S = f64>) -> Result<(), KError> {
        self.setup(a)
    }
}

#[test]
fn pc_rebuilds_on_structure_change() {
    let a1 = Arc::new(CsrMatrix::from_csr(
        2,
        2,
        vec![0, 1, 2],
        vec![0, 1],
        vec![S::from_real(1.0).real(), S::from_real(2.0).real()],
    ));
    let op1 = Arc::new(CsrOp::new(a1));
    let a2 = Arc::new(CsrMatrix::from_csr(
        2,
        2,
        vec![0, 1, 2],
        vec![0, 1],
        vec![S::from_real(3.0).real(), S::from_real(4.0).real()],
    ));
    let op2 = Arc::new(CsrOp::new(a2));

    let mut ksp = KspContext::new();
    ksp.set_type(SolverType::Gmres).unwrap();
    ksp.set_pc_type(PcType::Jacobi, None).unwrap();
    ksp.set_operators(op1.clone(), None);

    ksp.setup().unwrap();
    let sid1 = ksp.last_pc_sid();

    ksp.set_operators(op2.clone(), None);
    op2.mark_structure_changed();
    ksp.setup().unwrap();
    assert_ne!(sid1, ksp.last_pc_sid());
}

#[test]
fn jacobi_numeric_update_without_rebuild() {
    let a = Arc::new(CsrMatrix::from_csr(
        2,
        2,
        vec![0, 1, 2],
        vec![0, 1],
        vec![S::from_real(1.0).real(), S::from_real(2.0).real()],
    ));
    let op = Arc::new(CsrOp::new(a));

    let mut ksp = KspContext::new();
    ksp.set_type(SolverType::Gmres).unwrap();
    ksp.set_pc_type(PcType::Jacobi, None).unwrap();
    ksp.set_pc_reuse_policy(PcReusePolicy::ReuseNumeric);
    ksp.set_operators(op.clone(), None);

    ksp.setup().unwrap();
    let sid0 = ksp.last_pc_sid();
    let vid0 = ksp.last_pc_vid();

    op.mark_values_changed();
    ksp.setup().unwrap();
    assert_eq!(sid0, ksp.last_pc_sid());
    assert_ne!(vid0, ksp.last_pc_vid());
}

#[test]
fn denseop_cache_invalidation() {
    let mat = Arc::new(Mat::from_fn(2, 2, |i, j| {
        if i == j {
            S::one().real()
        } else {
            R::default()
        }
    }));
    let op = DenseOp::new(mat);
    let csr1 = op.to_csr_cached(R::default());
    let p1 = Arc::as_ptr(&csr1);
    op.mark_values_changed();
    let csr2 = op.to_csr_cached(R::default());
    let p2 = Arc::as_ptr(&csr2);
    assert_ne!(p1, p2);
}

#[cfg(not(feature = "mat-values-fingerprint"))]
#[test]
fn unknown_vid_triggers_numeric_refresh() {
    let (pc, numeric, _) = CountPc::new();
    let a = Arc::new(Mat::from_fn(2, 2, |i, j| {
        if i == j {
            S::one().real()
        } else {
            R::default()
        }
    }));
    let mut ksp = KspContext::new();
    ksp.set_type(SolverType::Gmres).unwrap();
    ksp.set_pc_box_for_tests(Box::new(pc));
    ksp.set_operators(a.clone(), None);
    ksp.setup().unwrap();
    ksp.setup().unwrap();
    assert_eq!(numeric.load(Ordering::SeqCst), 1);
}

#[test]
fn pattern_mismatch_in_numeric_update() {
    let mut pc = PatternCheckPc::default();
    let a1 = Mat::from_fn(2, 2, |i, j| {
        if i == j {
            S::one().real()
        } else {
            R::default()
        }
    });
    pc.setup(&a1).unwrap();
    let a2 = Mat::from_fn(2, 2, |i, j| {
        if i == j {
            S::one().real()
        } else if i == 0 && j == 1 {
            S::from_real(0.5).real()
        } else {
            R::default()
        }
    });
    let err = pc.update_numeric(&a2).unwrap_err();
    match err {
        KError::Unsupported(msg) => assert!(msg.contains("pattern changed")),
        other => panic!("unexpected error: {:?}", other),
    }
}

#[test]
fn values_id_known_triggers_single_numeric_update() {
    let (pc, numeric, _) = CountPc::new();
    let mat = Arc::new(Mat::from_fn(2, 2, |i, j| {
        if i == j {
            S::one().real()
        } else {
            R::default()
        }
    }));
    let op = Arc::new(DenseOp::new(mat.clone()));
    let mut ksp = KspContext::new();
    ksp.set_type(SolverType::Gmres).unwrap();
    ksp.set_pc_box_for_tests(Box::new(pc));
    ksp.set_operators(op.clone(), None);
    ksp.setup().unwrap();
    // trigger numeric refresh
    op.mark_values_changed();
    ksp.setup().unwrap();
    ksp.setup().unwrap();
    assert_eq!(numeric.load(Ordering::SeqCst), 1);
}

#[test]
fn policy_never_forces_symbolic_update() {
    let (pc, numeric, symbolic) = CountPc::new();
    let mat = Arc::new(Mat::from_fn(2, 2, |i, j| {
        if i == j {
            S::one().real()
        } else {
            R::default()
        }
    }));
    let op = Arc::new(DenseOp::new(mat.clone()));
    let mut ksp = KspContext::new();
    ksp.set_type(SolverType::Gmres).unwrap();
    ksp.set_pc_box_for_tests(Box::new(pc));
    ksp.set_pc_reuse_policy(PcReusePolicy::Never);
    ksp.set_operators(op.clone(), None);
    ksp.setup().unwrap();
    op.mark_values_changed();
    ksp.setup().unwrap();
    assert_eq!(numeric.load(Ordering::SeqCst), 0);
    assert_eq!(symbolic.load(Ordering::SeqCst), 1);
}
