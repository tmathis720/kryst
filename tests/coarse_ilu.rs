#![cfg(feature = "backend-faer")]
use faer::Mat;
use fixtures::csr_poisson_1d;
use kryst::algebra::prelude::*;
use kryst::preconditioner::amg::{AMGBuilder, CoarseSolve, RelaxPhase, RelaxType};
use kryst::preconditioner::{PcSide, Preconditioner};

#[test]
fn ilu_coarse_matches_dense() {
    let a = csr_poisson_1d(64);
    let rhs = vec![R::from(1.0); a.nrows()];

    let mut amg_dense = AMGBuilder::new()
        .relaxation_type(RelaxType::Jacobi)
        .grid_relax_type_all(RelaxType::Jacobi)
        .num_grid_sweeps(RelaxPhase::Coarsest, 0)
        .coarse_solve(CoarseSolve::DirectDense)
        .require_spd(false)
        .build(&Mat::<R>::zeros(0, 0))
        .unwrap();
    amg_dense.setup(&a).unwrap();
    let mut x_dense = vec![R::default(); a.nrows()];
    amg_dense.apply(PcSide::Left, &rhs, &mut x_dense).unwrap();

    let mut amg_ilu = AMGBuilder::new()
        .relaxation_type(RelaxType::Jacobi)
        .grid_relax_type_all(RelaxType::Jacobi)
        .num_grid_sweeps(RelaxPhase::Coarsest, 0)
        .coarse_solve(CoarseSolve::ILU)
        .require_spd(false)
        .build(&Mat::<R>::zeros(0, 0))
        .unwrap();
    amg_ilu.setup(&a).unwrap();
    let mut x_ilu = vec![R::default(); a.nrows()];
    amg_ilu.apply(PcSide::Left, &rhs, &mut x_ilu).unwrap();

    let diff = x_ilu
        .iter()
        .zip(&x_dense)
        .map(|(&a, &b)| {
            let delta = a - b;
            delta * delta
        })
        .sum::<R>()
        .sqrt();
    let norm = x_dense
        .iter()
        .map(|v| {
            let vv = *v;
            vv * vv
        })
        .sum::<R>()
        .sqrt();
    assert!(
        diff / norm <= R::from(1e-8),
        "relative diff = {}",
        diff / norm
    );
}

mod fixtures;
