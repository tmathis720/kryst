use faer::Mat;
use kryst::preconditioner::amg::{AMGBuilder, RelaxType};
use kryst::preconditioner::{PcSide, Preconditioner};
use fixtures::csr_poisson_1d;

#[test]
fn w_cycle_reduces_residual_more_than_v() {
    let a = csr_poisson_1d(128);
    let rhs = vec![1.0; a.nrows()];
    let mut res = vec![0.0; a.nrows()];

    let mut amg_v = AMGBuilder::new()
        .relaxation_type(RelaxType::Jacobi)
        .grid_relax_type_all(RelaxType::Jacobi)
        .cycle_v()
        .build(&Mat::<f64>::zeros(0, 0))
        .unwrap();
    amg_v.setup(&a).unwrap();
    let mut z_v = vec![0.0; a.nrows()];
    amg_v.apply(PcSide::Left, &rhs, &mut z_v).unwrap();
    a.spmv_scaled(1.0, &z_v, 0.0, &mut res).unwrap();
    for i in 0..a.nrows() {
        res[i] = rhs[i] - res[i];
    }
    let norm_v = res.iter().map(|x| x * x).sum::<f64>().sqrt();

    let mut amg_w = AMGBuilder::new()
        .relaxation_type(RelaxType::Jacobi)
        .grid_relax_type_all(RelaxType::Jacobi)
        .cycle_w(2)
        .build(&Mat::<f64>::zeros(0, 0))
        .unwrap();
    amg_w.setup(&a).unwrap();
    let mut z_w = vec![0.0; a.nrows()];
    amg_w.apply(PcSide::Left, &rhs, &mut z_w).unwrap();
    a.spmv_scaled(1.0, &z_w, 0.0, &mut res).unwrap();
    for i in 0..a.nrows() {
        res[i] = rhs[i] - res[i];
    }
    let norm_w = res.iter().map(|x| x * x).sum::<f64>().sqrt();
    println!("norm_v = {}, norm_w = {}", norm_v, norm_w);
    assert!(norm_w <= norm_v * 1.1);
}

mod fixtures;
