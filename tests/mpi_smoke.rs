#![cfg(not(feature = "complex"))]
#![cfg(feature = "mpi")]

use std::any::Any;
use std::sync::Arc;

use kryst::context::ksp_context::{KspContext, SolverType};
use kryst::matrix::op::{LinOp, StructureId, ValuesId};
use kryst::parallel::{Comm, MpiComm, UniverseComm};
use kryst::utils::convergence::ConvergedReason;

#[derive(Clone)]
struct ReplicatedDiagOp {
    diag: [f64; 2],
    comm: UniverseComm,
    sid: StructureId,
    vid: ValuesId,
}

impl ReplicatedDiagOp {
    fn new(diag: [f64; 2], comm: UniverseComm) -> Self {
        Self {
            diag,
            comm,
            sid: StructureId(2),
            vid: ValuesId(2),
        }
    }
}

impl LinOp for ReplicatedDiagOp {
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
fn mpi_dot_allreduce_matches_expected() {
    let comm = UniverseComm::Mpi(Arc::new(MpiComm::new()));
    comm.set_reproducible(true);

    let rank = comm.rank();
    let values: Vec<f64> = (0..4).map(|i| i as f64 + rank as f64).collect();
    let dot = comm.dot(&values, &values);

    let expected = (0..comm.size())
        .map(|r| {
            let offset = r as f64;
            (0..4)
                .map(|i| {
                    let v = i as f64 + offset;
                    v * v
                })
                .sum::<f64>()
        })
        .sum::<f64>();

    assert!((dot - expected).abs() < 1e-10);
}

#[test]
fn mpi_replicated_cg_succeeds() {
    let comm = UniverseComm::Mpi(Arc::new(MpiComm::new()));
    comm.set_reproducible(true);

    let op = Arc::new(ReplicatedDiagOp::new([2.0, 5.0], comm.clone()));
    let b = vec![2.0, 15.0];
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
    assert!((x[0] - 1.0).abs() < 1e-12);
    assert!((x[1] - 3.0).abs() < 1e-12);
    let global_norm = comm.dot(&x, &x);
    let expected_norm = comm.size() as f64 * (1.0 + 9.0);
    assert!((global_norm - expected_norm).abs() < 1e-10);
}
