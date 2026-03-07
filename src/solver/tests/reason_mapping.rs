#![cfg(not(feature = "complex"))]

use crate::error::KError;
use crate::utils::convergence::{
    ConvergedReason, FailureReasonKind, FailureStage, NestedPcFailure, ReasonEmitter,
    map_kerror_to_reason,
};

#[test]
fn reason_emitter_non_finite_maps_nan_and_inf() {
    assert_eq!(
        ReasonEmitter::non_finite(f64::NAN),
        Some(ConvergedReason::DivergedNan)
    );
    assert_eq!(
        ReasonEmitter::non_finite(f64::INFINITY),
        Some(ConvergedReason::DivergedInf)
    );
    assert_eq!(ReasonEmitter::non_finite(1.0), None);
}

#[test]
fn reason_emitter_breakdown_and_indefinite_helpers_are_stable() {
    assert_eq!(
        ReasonEmitter::breakdown(),
        ConvergedReason::DivergedBreakdown
    );
    assert_eq!(
        ReasonEmitter::breakdown_bicg(),
        ConvergedReason::DivergedBreakdownBiCG
    );
    assert_eq!(
        ReasonEmitter::indefinite_matrix(),
        ConvergedReason::DivergedIndefiniteMatrix
    );
    assert_eq!(
        ReasonEmitter::indefinite_pc(),
        ConvergedReason::DivergedIndefinitePC
    );
}

#[test]
fn kerror_stage_mapping_is_consistent_across_pc_setup_apply_and_indefiniteness() {
    assert_eq!(
        map_kerror_to_reason(&KError::PcFailed("setup".into()), FailureStage::Setup),
        Some(ConvergedReason::DivergedPcSetupFailed)
    );
    assert_eq!(
        map_kerror_to_reason(&KError::PcFailed("apply".into()), FailureStage::Solve),
        Some(ConvergedReason::DivergedPcFailed)
    );
    assert_eq!(
        map_kerror_to_reason(&KError::IndefiniteMatrix, FailureStage::Solve),
        Some(ConvergedReason::DivergedIndefiniteMatrix)
    );
    assert_eq!(
        map_kerror_to_reason(&KError::DivergedIndefinitePC, FailureStage::Solve),
        Some(ConvergedReason::DivergedIndefinitePC)
    );
    assert_eq!(
        map_kerror_to_reason(&KError::BreakdownOrIndefinite, FailureStage::Solve),
        Some(ConvergedReason::DivergedBreakdown)
    );
}

#[test]
fn reason_emitter_nested_metadata_preserves_nested_failure_context() {
    let nested = NestedPcFailure {
        component: "inner-ksp",
        reason: ConvergedReason::DivergedBreakdownBiCG,
        iterations: 7,
        final_norm: Some("||M^-1 r||=1e-3".into()),
        residual_history_summary: Some("7 entries".into()),
        detail: "inner preconditioner broke down".into(),
    };
    let cloned = ReasonEmitter::nested_pc_failure(
        &KError::NestedPcFailed(nested.clone()),
        FailureStage::Solve,
    )
    .expect("nested metadata should be present");

    assert_eq!(cloned, nested);
}

#[test]
fn reason_emitter_nested_metadata_builds_stage_specific_pc_failure() {
    let setup_failure = ReasonEmitter::nested_pc_failure(
        &KError::PcFailed("zero pivot".into()),
        FailureStage::Setup,
    )
    .expect("setup metadata");
    assert_eq!(setup_failure.reason, ConvergedReason::DivergedPcSetupFailed);
    assert!(setup_failure.detail.contains("stage=Setup"));

    let apply_failure = ReasonEmitter::nested_pc_failure(
        &KError::PcFailed("apply failed".into()),
        FailureStage::Solve,
    )
    .expect("apply metadata");
    assert_eq!(apply_failure.reason, ConvergedReason::DivergedPcFailed);
    assert!(apply_failure.detail.contains("stage=Solve"));
}

#[test]
fn failure_reason_kind_indefinite_variants_are_canonical() {
    assert_eq!(
        ConvergedReason::from_failure_kind(FailureReasonKind::IndefiniteMatrix),
        ConvergedReason::DivergedIndefiniteMatrix
    );
    assert_eq!(
        ConvergedReason::from_failure_kind(FailureReasonKind::IndefinitePc),
        ConvergedReason::DivergedIndefinitePC
    );
}
