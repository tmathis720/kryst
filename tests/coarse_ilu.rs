use faer::Mat;
use fixtures::csr_poisson_1d;
use kryst::preconditioner::amg::{AMGBuilder, CoarseSolve, RelaxPhase, RelaxType};
use kryst::preconditioner::{PcSide, Preconditioner};

#[test]
fn ilu_coarse_matches_dense() {
    let a = csr_poisson_1d(64);
    let rhs = vec![1.0; a.nrows()];

    let mut amg_dense = AMGBuilder::new()
        .relaxation_type(RelaxType::Jacobi)
        .grid_relax_type_all(RelaxType::Jacobi)
        .num_grid_sweeps(RelaxPhase::Coarsest, 0)
        .coarse_solve(CoarseSolve::DirectDense)
        .require_spd(false)
        .build(&Mat::<f64>::zeros(0, 0))
        .unwrap();
    amg_dense.setup(&a).unwrap();
    let mut x_dense = vec![0.0; a.nrows()];
    amg_dense.apply(PcSide::Left, &rhs, &mut x_dense).unwrap();

    let mut amg_ilu = AMGBuilder::new()
        .relaxation_type(RelaxType::Jacobi)
        .grid_relax_type_all(RelaxType::Jacobi)
        .num_grid_sweeps(RelaxPhase::Coarsest, 0)
        .coarse_solve(CoarseSolve::ILU)
        .require_spd(false)
        .build(&Mat::<f64>::zeros(0, 0))
        .unwrap();
    amg_ilu.setup(&a).unwrap();
    let mut x_ilu = vec![0.0; a.nrows()];
    amg_ilu.apply(PcSide::Left, &rhs, &mut x_ilu).unwrap();

    let diff = x_ilu
        .iter()
        .zip(&x_dense)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        .sqrt();
    let norm = x_dense.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(diff / norm <= 1e-8, "relative diff = {}", diff / norm);
}

mod fixtures;
