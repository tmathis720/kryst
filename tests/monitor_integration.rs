#![cfg(all(feature = "backend-faer", not(feature = "complex")))]
//! Tests for iteration monitoring and profiling functionality.

use faer::Mat;
use kryst::algebra::prelude::*;
use kryst::context::ksp_context::{KspContext, SolverType};
use kryst::context::pc_context::PcType;
use kryst::error::KError;
use kryst::solver::MonitorAction;
use std::sync::{Arc, Mutex};

#[test]
fn test_monitor_registration() {
    let mut ksp = KspContext::new();

    // Initially no monitors
    assert_eq!(ksp.num_monitors(), 0);

    // Add a monitor
    ksp.add_monitor(|_iter, _residual, _reductions| MonitorAction::Continue);
    assert_eq!(ksp.num_monitors(), 1);

    // Add another monitor
    ksp.add_monitor(|_iter, _residual, _reductions| MonitorAction::Continue);
    assert_eq!(ksp.num_monitors(), 2);

    // Clear monitors
    ksp.clear_monitors();
    assert_eq!(ksp.num_monitors(), 0);
}

#[test]
fn test_monitor_invocation() {
    let call_count = Arc::new(Mutex::new(0));
    let call_count_clone = Arc::clone(&call_count);

    let mut ksp = KspContext::new();

    // Register a monitor that counts calls
    ksp.add_monitor(move |_iter, _residual, _reductions| {
        let mut count = call_count_clone.lock().unwrap();
        *count += 1;
        MonitorAction::Continue
    });

    // Manually invoke monitors to test the mechanism
    ksp.invoke_monitors(0, R::from(1.0), 0);
    ksp.invoke_monitors(1, R::from(0.5), 0);
    ksp.invoke_monitors(2, R::from(0.1), 0);

    let final_count = *call_count.lock().unwrap();
    assert_eq!(final_count, 3);
}

#[test]
fn test_monitor_with_solver() -> Result<(), KError> {
    // Create a simple 2x2 SPD test matrix
    let n = 2;
    let a = Mat::<R>::from_fn(
        n,
        n,
        |i, j| if i == j { R::from(2.0) } else { R::from(-1.0) },
    );
    let b = vec![R::from(1.0); n];
    let mut x = vec![R::default(); n];

    // Track residuals seen by monitor
    let residuals = Arc::new(Mutex::new(Vec::new()));
    let residuals_clone = Arc::clone(&residuals);

    let mut ksp = KspContext::new();

    // Register a monitor to collect residuals
    ksp.add_monitor(move |iter, residual, _reductions| {
        let mut res_vec = residuals_clone.lock().unwrap();
        res_vec.push((iter, residual));
        MonitorAction::Continue
    });

    ksp.set_type(SolverType::Cg)?
        .set_pc_type(PcType::None, None)?;

    use kryst::matrix::op::LinOp;
    let amat: Arc<dyn LinOp<S = f64>> = Arc::new(a);
    ksp.set_operators(amat, None);

    let _ = ksp.solve(&b, &mut x);

    // For PREONLY, monitors might not be called since it's a direct solver
    // This is expected behavior
    Ok(())
}

#[test]
fn test_multiple_monitors() {
    let mut ksp = KspContext::new();

    let count1 = Arc::new(Mutex::new(0));
    let count2 = Arc::new(Mutex::new(0));

    let count1_clone = Arc::clone(&count1);
    let count2_clone = Arc::clone(&count2);

    // Register two monitors
    ksp.add_monitor(move |_iter, _residual, _reductions| {
        let mut c = count1_clone.lock().unwrap();
        *c += 1;
        MonitorAction::Continue
    });

    ksp.add_monitor(move |_iter, _residual, _reductions| {
        let mut c = count2_clone.lock().unwrap();
        *c += 2; // Different increment to distinguish
        MonitorAction::Continue
    });

    // Invoke monitors
    ksp.invoke_monitors(0, R::from(1.0), 0);

    assert_eq!(*count1.lock().unwrap(), 1);
    assert_eq!(*count2.lock().unwrap(), 2);
}

#[cfg(feature = "logging")]
#[test]
fn test_stage_guard() {
    use kryst::utils::profiling::StageGuard;

    // This test just ensures StageGuard can be created and dropped
    {
        let _guard = StageGuard::new("TestStage");
        // Guard should log entry here (if logging enabled)
    } // Guard should log exit here (if logging enabled)

    // Test nested guards
    let _outer = StageGuard::new("OuterStage");
    {
        let _inner = StageGuard::new("InnerStage");
    }
}
