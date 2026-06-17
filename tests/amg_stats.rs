#![cfg(all(feature = "backend-faer", not(feature = "complex")))]
use std::time::Duration;

use faer::Mat;

use fixtures::csr_poisson_1d;
use kryst::algebra::prelude::*;
use kryst::assert_s_close;
use kryst::preconditioner::amg::{AMGBuilder, DistApplyStats};
use kryst::preconditioner::dist::{DistCoarseSolverRoute, DistCoarseStrategy};
use kryst::preconditioner::gamg::{Gamg, GamgConfig};
use kryst::preconditioner::{PcSide, Preconditioner};

#[test]
fn amg_level_stats() {
    let a = csr_poisson_1d(8);
    let mut amg = AMGBuilder::new()
        .logging_level(1)
        .relaxation_type(kryst::preconditioner::amg::RelaxType::Jacobi)
        .grid_relax_type_all(kryst::preconditioner::amg::RelaxType::Jacobi)
        .build(&Mat::<R>::zeros(0, 0))
        .unwrap();
    amg.setup(&a).unwrap();
    let stats = amg.stats().expect("stats");
    assert_eq!(stats.levels[0].n, a.nrows());
    for l in 1..stats.levels.len() {
        assert!(stats.levels[l - 1].n > stats.levels[l].n);
    }
    assert!(stats.operator_complexity >= 1.0);
    assert!(stats.grid_complexity >= 1.0);
    assert!(stats.total_nnz >= stats.levels[0].nnz_a);
    assert!(stats.total_smoothing_work >= 0.0);
    let observed = S::from_real(stats.levels[0].max_row_sum_a);
    let expected = S::from_real(R::from(4.0));
    assert_s_close!("amg level row sum", expected, observed);
}

#[test]
fn amg_cycle_timings_gated() {
    let a = csr_poisson_1d(8);
    let rhs = vec![S::one().real(); a.nrows()];
    let mut z = vec![R::default(); a.nrows()];

    let mut amg = AMGBuilder::new()
        .logging_level(0)
        .relaxation_type(kryst::preconditioner::amg::RelaxType::Jacobi)
        .grid_relax_type_all(kryst::preconditioner::amg::RelaxType::Jacobi)
        .build(&Mat::<R>::zeros(0, 0))
        .unwrap();
    amg.setup(&a).unwrap();
    assert!(amg.stats().is_none());
    amg.apply(PcSide::Left, &rhs, &mut z).unwrap();
    assert!(amg.stats().is_none());

    let mut amg2 = AMGBuilder::new()
        .logging_level(2)
        .relaxation_type(kryst::preconditioner::amg::RelaxType::Jacobi)
        .grid_relax_type_all(kryst::preconditioner::amg::RelaxType::Jacobi)
        .build(&Mat::<R>::zeros(0, 0))
        .unwrap();
    amg2.setup(&a).unwrap();
    let mut z2 = vec![R::default(); a.nrows()];
    amg2.apply(PcSide::Left, &rhs, &mut z2).unwrap();
    let stats = amg2.stats().unwrap();
    let cyc = stats.last_cycle.expect("cycle timings");
    assert!(cyc.total_cycle > Duration::default());
}

#[test]
fn amg_stats_include_dist_route_and_fallback_chain() {
    let a = csr_poisson_1d(8);
    let mut cfg = kryst::preconditioner::amg::AMGConfig::default();
    cfg.logging_level = 1;
    cfg.dist_coarse_strategy = DistCoarseStrategy::RootGather;
    cfg.dist_coarse_solver_route = DistCoarseSolverRoute::SuperLuDist;
    let mut amg = kryst::preconditioner::amg::AMG::with_config(cfg);
    amg.setup(&a).unwrap();
    let stats = amg.stats().expect("stats");
    assert_eq!(
        stats.selected_dist_coarse_route.as_deref(),
        Some("superlu_dist")
    );
    assert_eq!(
        stats.dist_route_fallback,
        vec![
            "superlu_dist".to_string(),
            "root_gather".to_string(),
            "local_prototype".to_string()
        ]
    );
}

#[test]
fn dist_apply_stats_expose_stable_route_labels() {
    let distributed = DistApplyStats {
        mode: DistCoarseStrategy::DistributedCsr,
        coarse_solver_route: DistCoarseSolverRoute::Auto,
        ..Default::default()
    };
    assert_eq!(distributed.mode_label(), "distributed_csr");
    assert_eq!(distributed.coarse_solver_route_label(), "distributed_csr");
    assert!(!distributed.uses_root_gather());
    assert!(!distributed.reports_distributed_support());
    assert!(!distributed.setup_uses_fine_matrix_gather());
    assert!(!distributed.apply_uses_root_vector_gather());

    let true_distributed = DistApplyStats {
        mode: DistCoarseStrategy::DistributedCsr,
        coarse_solver_route: DistCoarseSolverRoute::Auto,
        true_distributed_hierarchy: true,
        ..Default::default()
    };
    assert!(true_distributed.reports_distributed_support());

    let root = DistApplyStats {
        mode: DistCoarseStrategy::RootGather,
        coarse_solver_route: DistCoarseSolverRoute::Root,
        ..Default::default()
    };
    assert_eq!(root.mode_label(), "root_gather");
    assert_eq!(root.coarse_solver_route_label(), "root_gather");
    assert!(root.uses_root_gather());
    assert!(!root.reports_distributed_support());
    assert!(!root.setup_uses_fine_matrix_gather());
    assert!(!root.apply_uses_root_vector_gather());

    let root_after_setup = DistApplyStats {
        mode: DistCoarseStrategy::RootGather,
        coarse_solver_route: DistCoarseSolverRoute::Root,
        setup_gathered_fine_matrix: true,
        ..Default::default()
    };
    assert!(root_after_setup.setup_uses_fine_matrix_gather());

    let local = DistApplyStats {
        mode: DistCoarseStrategy::LocalPrototype,
        coarse_solver_route: DistCoarseSolverRoute::Local,
        ..Default::default()
    };
    assert_eq!(local.mode_label(), "local_prototype");
    assert_eq!(local.coarse_solver_route_label(), "local_prototype");
    assert!(!local.uses_root_gather());
    assert!(!local.reports_distributed_support());

    let superlu = DistApplyStats {
        mode: DistCoarseStrategy::SuperLuDist,
        coarse_solver_route: DistCoarseSolverRoute::Auto,
        ..Default::default()
    };
    assert_eq!(superlu.mode_label(), "superlu_dist");
    assert_eq!(superlu.coarse_solver_route_label(), "superlu_dist");
    assert!(!superlu.uses_root_gather());
    assert!(superlu.reports_distributed_support());
}

#[test]
fn gamg_route_policy_auto_uses_fallback_chain_during_setup() {
    let a = csr_poisson_1d(8);
    let opts = kryst::config::options::PcOptions {
        amg_dist_coarse_solver_route: Some("auto,superlu_dist,root,local".into()),
        pc_gamg_level_policies: Some(vec![
            "level=1,coarse_routes=auto,superlu_dist,root,local".into(),
        ]),
        ..Default::default()
    };
    let cfg = GamgConfig::try_from_opts(&opts).expect("gamg opts");
    let mut gamg = Gamg::with_config(cfg);
    gamg.setup(&a)
        .expect("gamg setup should resolve auto route");
}

mod fixtures;
