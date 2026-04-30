use crate::solver::fgmres::{FgmresSolver, FgmresStagnationPolicy, FgmresVariant, PipelinePolicy};

#[test]
fn fgmres_classical_stagnation_disabled_does_not_force_restart() {
    let mut solver = FgmresSolver::new(1e-10, 20, 5);
    solver.variant = FgmresVariant::Classical;
    solver.stagnation_policy = FgmresStagnationPolicy::Disabled;
    let (_action, restart) = solver.stagnation_action(8);
    assert!(!restart);
}

#[test]
fn fgmres_classical_stagnation_pipeline_only_does_not_force_restart() {
    let mut solver = FgmresSolver::new(1e-10, 20, 5);
    solver.variant = FgmresVariant::Classical;
    solver.pipeline_policy = PipelinePolicy::FallbackToClassicalOnStagnation;
    solver.stagnation_policy = FgmresStagnationPolicy::PipelineFallbackOnly;
    let (_action, restart) = solver.stagnation_action(8);
    assert!(!restart);
}

#[test]
fn fgmres_stagnation_fallback_is_gated_by_min_inner_floor() {
    let mut solver = FgmresSolver::new(1e-6, 100, 20);
    solver.stagnation_policy = FgmresStagnationPolicy::RestartClassicalToo;
    solver.min_inner_before_fallback = 10;
    let (_action, restart) = solver.stagnation_action(4);
    assert!(!restart);
}

#[test]
fn fgmres_stagnation_fallback_can_trigger_after_min_inner_floor() {
    let mut solver = FgmresSolver::new(1e-6, 100, 20);
    solver.stagnation_policy = FgmresStagnationPolicy::RestartClassicalToo;
    solver.min_inner_before_fallback = 3;
    let (_action, restart) = solver.stagnation_action(6);
    assert!(restart);
}
