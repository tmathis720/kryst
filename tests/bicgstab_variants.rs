#![cfg(all(feature = "backend-faer", not(feature = "complex")))]

use faer::Mat;
use kryst::config::options::KspOptions;
use kryst::context::ksp_context::KspContext;
use kryst::parallel::{NoComm, UniverseComm};
use kryst::solver::LinearSolver;
use kryst::solver::{BiCgStabSolver, BiCgStabVariant};
use std::sync::Arc;

fn nonsym_tridiag(n: usize) -> Mat<f64> {
    let mut a = Mat::<f64>::zeros(n, n);
    for i in 0..n {
        a[(i, i)] = 4.0;
        if i > 0 {
            a[(i, i - 1)] = -1.0;
        }
        if i + 1 < n {
            a[(i, i + 1)] = 2.0;
        }
    }
    a
}

#[test]
fn bicgstab_lowsync_keeps_convergence_and_reduces_syncs() {
    let n = 36;
    let a = nonsym_tridiag(n);
    let b = vec![1.0; n];
    let comm = UniverseComm::NoComm(NoComm);

    let mut x_classic = vec![0.0; n];
    let mut classic = BiCgStabSolver::new(1e-9, 300);
    classic.set_variant(BiCgStabVariant::Classic);
    let mut ws_classic = kryst::context::ksp_context::Workspace::default();
    let stats_classic = classic
        .solve_f64(
            &a,
            None,
            &b,
            &mut x_classic,
            kryst::preconditioner::PcSide::Left,
            &comm,
            None,
            Some(&mut ws_classic),
        )
        .unwrap();

    let mut x_low = vec![0.0; n];
    let mut low = BiCgStabSolver::new(1e-9, 300);
    low.set_variant(BiCgStabVariant::LowSync);
    let mut ws_low = kryst::context::ksp_context::Workspace::default();
    let stats_low = low
        .solve_f64(
            &a,
            None,
            &b,
            &mut x_low,
            kryst::preconditioner::PcSide::Left,
            &comm,
            None,
            Some(&mut ws_low),
        )
        .unwrap();

    assert!(stats_low.final_residual <= 1e-7);
    assert!(
        stats_low.counters.num_global_reductions <= stats_classic.counters.num_global_reductions
    );
}

#[test]
fn bicgstab_variant_selectable_from_ksp_options() {
    let opts =
        KspOptions::from_args(&["-ksp_type", "bicgstab", "-ksp_bicgstab_variant", "lowsync"])
            .unwrap();
    let mut ksp = KspContext::new();
    ksp.set_from_options(&opts).unwrap();

    let n = 20;
    let a = nonsym_tridiag(n);
    let b = vec![1.0; n];
    let mut x = vec![0.0; n];
    ksp.set_operators(Arc::new(a), None);
    let stats = ksp.solve(&b, &mut x).unwrap();

    assert_eq!(
        stats.reduction_model.as_ref().map(|m| m.variant),
        Some("bicgstab-lowsync")
    );
}
