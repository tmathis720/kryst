use faer::Mat;
use fixtures::csr_poisson_1d;
use kryst::preconditioner::amg::{AMGBuilder, RelaxType};
use kryst::preconditioner::{PcSide, Preconditioner};

#[test]
fn keep_transpose_toggle_equivalence() {
    let a = csr_poisson_1d(32);
    let rhs = vec![1.0; a.nrows()];

    let mut amg_keep = AMGBuilder::new()
        .relaxation_type(RelaxType::Jacobi)
        .grid_relax_type_all(RelaxType::Jacobi)
        .keep_transpose(true)
        .build(&Mat::<f64>::zeros(0, 0))
        .unwrap();
    amg_keep.setup(&a).unwrap();
    let mut z_keep = vec![0.0; a.nrows()];
    amg_keep.apply(PcSide::Left, &rhs, &mut z_keep).unwrap();

    let mut amg_drop = AMGBuilder::new()
        .relaxation_type(RelaxType::Jacobi)
        .grid_relax_type_all(RelaxType::Jacobi)
        .keep_transpose(false)
        .build(&Mat::<f64>::zeros(0, 0))
        .unwrap();
    amg_drop.setup(&a).unwrap();
    let mut z_drop = vec![0.0; a.nrows()];
    amg_drop.apply(PcSide::Left, &rhs, &mut z_drop).unwrap();

    for i in 0..a.nrows() {
        assert!((z_keep[i] - z_drop[i]).abs() < 1e-8);
    }
}

mod fixtures;
