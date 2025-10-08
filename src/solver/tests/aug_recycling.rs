use crate::algebra::prelude::*;
use crate::context::ksp_context::Workspace;
use crate::error::KError;
use crate::parallel::{NoComm, UniverseComm};
use crate::preconditioner::PcSide;
use crate::solver::gmres::{AugmentationPolicy, GmresSolver};

use super::util;

#[test]
fn gmres_configures_recycling_workspace() -> Result<(), KError> {
    let mut solver = GmresSolver::new(12, 1e-8, 200);
    solver.augmentation = AugmentationPolicy::GmresDR { k: 6 };
    let a = util::spd_poisson2d(8);
    let b = util::rhs_random(a.nrows(), 11);
    let mut x: Vec<R> = vec![R::default(); a.nrows()];
    let mut ws = Workspace::default();
    let comm = UniverseComm::NoComm(NoComm);
    solver.solve_f64(
        &a,
        None,
        &b,
        &mut x,
        PcSide::Left,
        &comm,
        None,
        Some(&mut ws),
    )?;

    assert_eq!(ws.gmres_recycle.capacity(), 6);
    assert!(matches!(
        ws.gmres_recycle.policy(),
        AugmentationPolicy::GmresDR { k } if k == 6
    ));
    Ok(())
}
