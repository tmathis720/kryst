#![cfg(feature = "backend-faer")]
use faer::Mat;
use fixtures::csr_poisson_1d;
use kryst::algebra::prelude::*;
use kryst::assert_vec_close;
use kryst::preconditioner::amg::{AMGBuilder, RelaxType};
use kryst::preconditioner::{PcSide, Preconditioner};

#[test]
fn keep_transpose_toggle_equivalence() {
    let a = csr_poisson_1d(32);
    let rhs = vec![S::one().real(); a.nrows()];

    let mut amg_keep = AMGBuilder::new()
        .relaxation_type(RelaxType::Jacobi)
        .grid_relax_type_all(RelaxType::Jacobi)
        .keep_transpose(true)
        .build(&Mat::<R>::zeros(0, 0))
        .unwrap();
    amg_keep.setup(&a).unwrap();
    let mut z_keep = vec![R::default(); a.nrows()];
    amg_keep.apply(PcSide::Left, &rhs, &mut z_keep).unwrap();

    let mut amg_drop = AMGBuilder::new()
        .relaxation_type(RelaxType::Jacobi)
        .grid_relax_type_all(RelaxType::Jacobi)
        .keep_transpose(false)
        .build(&Mat::<R>::zeros(0, 0))
        .unwrap();
    amg_drop.setup(&a).unwrap();
    let mut z_drop = vec![R::default(); a.nrows()];
    amg_drop.apply(PcSide::Left, &rhs, &mut z_drop).unwrap();

    let keep: Vec<S> = z_keep.iter().copied().map(S::from_real).collect();
    let drop: Vec<S> = z_drop.iter().copied().map(S::from_real).collect();
    assert_vec_close!("amg keep transpose", &keep, &drop);
}

mod fixtures;
