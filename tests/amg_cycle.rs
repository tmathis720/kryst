#![cfg(all(feature = "backend-faer", not(feature = "complex")))]
use faer::Mat;
use fixtures::csr_poisson_1d;
use kryst::algebra::prelude::*;
use kryst::preconditioner::amg::{AMGBuilder, RelaxType};
use kryst::preconditioner::{PcSide, Preconditioner};

#[test]
fn w_cycle_reduces_residual_more_than_v() {
    let a = csr_poisson_1d(128);
    let rhs = vec![R::from(1.0); a.nrows()];
    let mut res = vec![R::default(); a.nrows()];

    let mut amg_v = AMGBuilder::new()
        .relaxation_type(RelaxType::Jacobi)
        .grid_relax_type_all(RelaxType::Jacobi)
        .cycle_v()
        .build(&Mat::<R>::zeros(0, 0))
        .unwrap();
    amg_v.setup(&a).unwrap();
    let mut z_v = vec![R::default(); a.nrows()];
    amg_v.apply(PcSide::Left, &rhs, &mut z_v).unwrap();
    a.spmv_scaled(R::from(1.0), &z_v, R::default(), &mut res)
        .unwrap();
    for i in 0..a.nrows() {
        res[i] = rhs[i] - res[i];
    }
    let norm_v = res
        .iter()
        .map(|x| {
            let xx = *x;
            xx * xx
        })
        .sum::<R>()
        .sqrt();

    let mut amg_w = AMGBuilder::new()
        .relaxation_type(RelaxType::Jacobi)
        .grid_relax_type_all(RelaxType::Jacobi)
        .cycle_w(2)
        .build(&Mat::<R>::zeros(0, 0))
        .unwrap();
    amg_w.setup(&a).unwrap();
    let mut z_w = vec![R::default(); a.nrows()];
    amg_w.apply(PcSide::Left, &rhs, &mut z_w).unwrap();
    a.spmv_scaled(R::from(1.0), &z_w, R::default(), &mut res)
        .unwrap();
    for i in 0..a.nrows() {
        res[i] = rhs[i] - res[i];
    }
    let norm_w = res
        .iter()
        .map(|x| {
            let xx = *x;
            xx * xx
        })
        .sum::<R>()
        .sqrt();
    println!("norm_v = {}, norm_w = {}", norm_v, norm_w);
    assert!(norm_w <= norm_v * R::from(1.1));
}

mod fixtures;

#[test]
fn amg_cycle_stats_include_work_estimate() {
    let a = csr_poisson_1d(64);
    let rhs = vec![R::from(1.0); a.nrows()];
    let mut amg = AMGBuilder::new()
        .logging_level(2)
        .relaxation_type(RelaxType::Jacobi)
        .grid_relax_type_all(RelaxType::Jacobi)
        .build(&Mat::<R>::zeros(0, 0))
        .unwrap();
    amg.setup(&a).unwrap();
    let mut z = vec![R::default(); a.nrows()];
    amg.apply(PcSide::Left, &rhs, &mut z).unwrap();
    let stats = amg.stats().expect("stats");
    assert!(stats.total_smoothing_work > 0.0);
}
