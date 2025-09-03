use std::time::Duration;

use faer::Mat;

use fixtures::csr_poisson_1d;
use kryst::preconditioner::amg::AMGBuilder;
use kryst::preconditioner::{PcSide, Preconditioner};

#[test]
fn amg_level_stats() {
    let a = csr_poisson_1d(8);
    let mut amg = AMGBuilder::new()
        .logging_level(1)
        .relaxation_type(kryst::preconditioner::amg::RelaxType::Jacobi)
        .grid_relax_type_all(kryst::preconditioner::amg::RelaxType::Jacobi)
        .build(&Mat::<f64>::zeros(0, 0))
        .unwrap();
    amg.setup(&a).unwrap();
    let stats = amg.stats().expect("stats");
    assert_eq!(stats.levels[0].n, a.nrows());
    for l in 1..stats.levels.len() {
        assert!(stats.levels[l - 1].n > stats.levels[l].n);
    }
    assert!((stats.levels[0].max_row_sum_a - 4.0).abs() < 1e-12);
}

#[test]
fn amg_cycle_timings_gated() {
    let a = csr_poisson_1d(8);
    let rhs = vec![1.0; a.nrows()];
    let mut z = vec![0.0; a.nrows()];

    let mut amg = AMGBuilder::new()
        .logging_level(0)
        .relaxation_type(kryst::preconditioner::amg::RelaxType::Jacobi)
        .grid_relax_type_all(kryst::preconditioner::amg::RelaxType::Jacobi)
        .build(&Mat::<f64>::zeros(0, 0))
        .unwrap();
    amg.setup(&a).unwrap();
    assert!(amg.stats().is_none());
    amg.apply(PcSide::Left, &rhs, &mut z).unwrap();
    assert!(amg.stats().is_none());

    let mut amg2 = AMGBuilder::new()
        .logging_level(2)
        .relaxation_type(kryst::preconditioner::amg::RelaxType::Jacobi)
        .grid_relax_type_all(kryst::preconditioner::amg::RelaxType::Jacobi)
        .build(&Mat::<f64>::zeros(0, 0))
        .unwrap();
    amg2.setup(&a).unwrap();
    let mut z2 = vec![0.0; a.nrows()];
    amg2.apply(PcSide::Left, &rhs, &mut z2).unwrap();
    let stats = amg2.stats().unwrap();
    let cyc = stats.last_cycle.expect("cycle timings");
    assert!(cyc.total_cycle > Duration::default());
}

mod fixtures;
