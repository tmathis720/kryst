use std::sync::Arc;

use faer::Mat;
use kryst::algebra::prelude::*;
use kryst::context::ksp_context::{KspContext, SolverType};
use kryst::matrix::op::{LinOp, wrap_with_comm};
use kryst::parallel::{NoComm, UniverseComm};

#[test]
#[should_panic]
fn mismatch_comm_panics() {
    let a = Mat::<f64>::from_fn(
        4,
        4,
        |i, j| {
            if i == j { R::from(1.0) } else { R::default() }
        },
    );
    let p = a.clone();
    let a_op: Arc<dyn LinOp<S = f64>> = wrap_with_comm(Arc::new(a), UniverseComm::NoComm(NoComm));

    #[cfg(feature = "mpi")]
    let p_comm = UniverseComm::Mpi(std::sync::Arc::new(
        kryst::parallel::MpiComm::try_new().unwrap(),
    ));
    #[cfg(all(not(feature = "mpi"), feature = "rayon"))]
    let p_comm = UniverseComm::Rayon(kryst::parallel::RayonComm::new());
    #[cfg(not(any(feature = "mpi", feature = "rayon")))]
    let p_comm = UniverseComm::Serial;

    let p_op: Arc<dyn LinOp<S = f64>> = wrap_with_comm(Arc::new(p), p_comm);
    let mut ksp = KspContext::new();
    ksp.set_operators(a_op, Some(p_op));
}

#[test]
fn residual_uses_comm_serial() {
    let a = Mat::<f64>::from_fn(
        3,
        3,
        |i, j| {
            if i == j { R::from(2.0) } else { R::default() }
        },
    );
    let b = vec![R::from(1.0); 3];
    let mut x = vec![R::default(); 3];

    let a_op = wrap_with_comm(Arc::new(a), UniverseComm::NoComm(NoComm));
    let mut ksp = KspContext::new();
    ksp.set_type(SolverType::Gmres).unwrap();
    ksp.set_operators(a_op, None);
    let stats = ksp.solve(&b, &mut x).unwrap();
    assert!(stats.final_residual.is_finite());
}
