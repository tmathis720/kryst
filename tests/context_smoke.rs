use std::any::Any;
use std::sync::Arc;

use kryst::context::ksp_context::{KspContext, SolverType};
#[cfg(all(
    feature = "backend-faer",
    feature = "dense-direct",
    not(feature = "complex")
))]
use kryst::context::pc_context::PcType;
use kryst::matrix::op::{LinOp, StructureId, ValuesId};
use kryst::parallel::{NoComm, UniverseComm};
use kryst::utils::convergence::ConvergedReason;

#[derive(Clone)]
struct TinyDiagOp {
    diag: [f64; 2],
    comm: UniverseComm,
    sid: StructureId,
    vid: ValuesId,
}

impl TinyDiagOp {
    fn new(diag: [f64; 2], comm: UniverseComm) -> Self {
        Self {
            diag,
            comm,
            sid: StructureId(1),
            vid: ValuesId(1),
        }
    }
}

impl LinOp for TinyDiagOp {
    type S = f64;

    fn dims(&self) -> (usize, usize) {
        (self.diag.len(), self.diag.len())
    }

    fn matvec(&self, x: &[Self::S], y: &mut [Self::S]) {
        for (yi, (&d, &xi)) in y.iter_mut().zip(self.diag.iter().zip(x.iter())) {
            *yi = d * xi;
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn structure_id(&self) -> StructureId {
        self.sid
    }

    fn values_id(&self) -> ValuesId {
        self.vid
    }

    fn comm(&self) -> UniverseComm {
        self.comm.clone()
    }
}

#[test]
fn context_smoke_cg() {
    let op = Arc::new(TinyDiagOp::new([2.0, 4.0], UniverseComm::NoComm(NoComm)));
    let b = vec![2.0, 8.0];
    let mut x = vec![0.0; 2];

    let mut ksp = KspContext::new();
    ksp.rtol = 1e-12;
    ksp.atol = 1e-12;
    ksp.set_type(SolverType::Cg).unwrap();
    ksp.set_operators(op, None);
    ksp.setup().unwrap();

    let stats = ksp.solve(&b, &mut x).unwrap();
    assert!(matches!(
        stats.reason,
        ConvergedReason::ConvergedRtol | ConvergedReason::ConvergedAtol
    ));
    assert!((x[0] - 1.0).abs() < 1e-10);
    assert!((x[1] - 2.0).abs() < 1e-10);
    assert!(stats.final_residual < 1e-10);
}

#[cfg(all(
    feature = "backend-faer",
    feature = "dense-direct",
    not(feature = "complex")
))]
#[test]
fn context_smoke_preonly_lu() {
    use faer::Mat;
    use kryst::matrix::op::DenseOp;

    let mat = Arc::new(Mat::from_fn(2, 2, |i, j| {
        if i == j { 3.0 + i as f64 } else { 0.0 }
    }));
    let op = Arc::new(DenseOp::new(mat));
    let b = vec![3.0, 8.0];
    let mut x = vec![0.0; 2];

    let mut ksp = KspContext::new();
    ksp.set_preonly_with_pc(PcType::Lu, None).unwrap();
    ksp.set_operators(op, None);
    ksp.setup().unwrap();

    let stats = ksp.solve(&b, &mut x).unwrap();
    assert!(matches!(stats.reason, ConvergedReason::ConvergedAtol));
    assert!((x[0] - 1.0).abs() < 1e-12);
    assert!((x[1] - 2.0).abs() < 1e-12);
    assert!(stats.final_residual < 1e-12);
}
