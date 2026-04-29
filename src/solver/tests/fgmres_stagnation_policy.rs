use crate::solver::fgmres::{
    FgmresSolver, FgmresStagnationPolicy, FgmresVariant, PipelinePolicy,
};

#[test]
fn fgmres_classical_stagnation_disabled_does_not_force_restart() {
    let mut solver = FgmresSolver::new(1e-10, 20, 5);
    solver.variant = FgmresVariant::Classical;
    solver.stagnation_policy = FgmresStagnationPolicy::Disabled;
    let (_action, restart) = solver.stagnation_action();
    assert!(!restart);
}

#[test]
fn fgmres_classical_stagnation_pipeline_only_does_not_force_restart() {
    let mut solver = FgmresSolver::new(1e-10, 20, 5);
    solver.variant = FgmresVariant::Classical;
    solver.pipeline_policy = PipelinePolicy::FallbackToClassicalOnStagnation;
    solver.stagnation_policy = FgmresStagnationPolicy::PipelineFallbackOnly;
    let (_action, restart) = solver.stagnation_action();
    assert!(!restart);
}
