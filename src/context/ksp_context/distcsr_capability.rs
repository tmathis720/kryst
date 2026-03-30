use crate::context::ksp_context::SolverType;
use crate::preconditioner::dist::{
    DistLocalApplyMode, DistRouteDecisionReport, DistRouteSelection, GlobalPcKind, LocalPcKind,
    MpiPcOptions,
};
use serde::Serialize;

#[derive(Clone, Copy, Debug)]
pub struct DistCsrCapabilityKey {
    pub solver_type: Option<SolverType>,
    pub global_pc: GlobalPcKind,
    pub local_pc: LocalPcKind,
    pub apply_mode: DistLocalApplyMode,
}

#[derive(Clone, Debug, Serialize)]
pub struct DistCsrCapabilityEntry {
    pub solver_type: String,
    pub global_pc: String,
    pub local_pc: String,
    pub apply_mode: String,
    pub native_global_candidate: bool,
    pub supports_native_distributed_mode: bool,
    pub supports_adapter_distributed_mode: bool,
    pub requested_distributed_mode: &'static str,
    pub requested_mode_supported: bool,
    pub requested_mode_failure_reason: Option<&'static str>,
    pub supports_native_apply_mode: bool,
    pub registry_rule: &'static str,
}

pub fn resolve_distcsr_capability(key: DistCsrCapabilityKey) -> DistCsrCapabilityEntry {
    let supports_native_distributed_mode =
        matches!(
            key.global_pc,
            GlobalPcKind::None | GlobalPcKind::BlockJacobi
        ) && key.local_pc.build_capabilities().native_local_apply;
    let supports_adapter_distributed_mode = matches!(
        key.global_pc,
        GlobalPcKind::None | GlobalPcKind::BlockJacobi
    ) && !key.apply_mode.requires_native();
    let requested_distributed_mode = if key.apply_mode.is_distributed_native() {
        "native_distributed"
    } else {
        "adapter_distributed"
    };
    let (requested_mode_supported, requested_mode_failure_reason) =
        if key.apply_mode.is_distributed_native() {
            if !key.local_pc.build_capabilities().native_local_apply {
                (false, Some("unsupported_local_pc"))
            } else if !matches!(
                key.global_pc,
                GlobalPcKind::None | GlobalPcKind::BlockJacobi
            ) {
                (false, Some("unsupported_global_pc"))
            } else {
                (true, None)
            }
        } else if !supports_adapter_distributed_mode {
            (false, Some("unsupported_global_pc"))
        } else if matches!(key.solver_type, Some(SolverType::Preonly))
            && key.apply_mode == DistLocalApplyMode::WrappedLocal
        {
            (false, Some("incompatible_solver_mode"))
        } else {
            (true, None)
        };
    let supports_native_apply_mode =
        key.apply_mode.is_distributed_native() && requested_mode_supported;

    let (native_global_candidate, registry_rule) = if !requested_mode_supported {
        match requested_mode_failure_reason {
            Some("unsupported_local_pc") => (false, "strict_mode_unsupported_local_pc"),
            Some("unsupported_global_pc") => (false, "strict_mode_unsupported_global_pc"),
            Some("incompatible_solver_mode") => (false, "ksp_solver_mode_incompatible"),
            _ => (false, "requested_mode_unsupported"),
        }
    } else if key.apply_mode.is_distributed_native() {
        match key.global_pc {
            GlobalPcKind::None | GlobalPcKind::BlockJacobi => {
                (true, "global_candidate_block_jacobi_native_apply")
            }
            GlobalPcKind::Asm | GlobalPcKind::Ras => (false, "explicit_global_pc_route"),
        }
    } else {
        (
            false,
            "global_candidate_block_jacobi_but_wrapped_local_apply",
        )
    };

    DistCsrCapabilityEntry {
        solver_type: key
            .solver_type
            .map(|solver| format!("{solver:?}"))
            .unwrap_or_else(|| "Unspecified".to_string()),
        global_pc: format!("{:?}", key.global_pc),
        local_pc: format!("{:?}", key.local_pc),
        apply_mode: key.apply_mode.communication_strategy_name().to_string(),
        native_global_candidate,
        supports_native_distributed_mode,
        supports_adapter_distributed_mode,
        requested_distributed_mode,
        requested_mode_supported,
        requested_mode_failure_reason,
        supports_native_apply_mode,
        registry_rule,
    }
}

pub fn build_dist_route_decision_report(
    mpi_opts: &MpiPcOptions,
    selected: DistRouteSelection,
    fallback_reason: Option<String>,
    fallback_chain_len: usize,
) -> DistRouteDecisionReport {
    let requested_mode = if mpi_opts.local_apply_mode.is_distributed_native() {
        "native_distributed"
    } else {
        "adapter_distributed"
    };
    DistRouteDecisionReport {
        requested_mode,
        selected_mode: selected.as_str(),
        fallback_reason,
        strict_local_apply: mpi_opts.local_apply_mode.requires_native(),
        native_required: mpi_opts.route_policy_budget.native_required,
        fallback_chain_len,
        max_allowed_fallbacks: mpi_opts.route_policy_budget.max_allowed_fallbacks,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DistCsrCapabilityKey, build_dist_route_decision_report, resolve_distcsr_capability,
    };
    use crate::context::ksp_context::SolverType;
    use crate::preconditioner::dist::{
        DistLocalApplyMode, DistRouteSelection, GlobalPcKind, LocalPcKind, MpiPcOptions,
    };

    #[test]
    fn wrapped_local_disables_native_global_candidate() {
        let cap = resolve_distcsr_capability(DistCsrCapabilityKey {
            solver_type: Some(SolverType::Gmres),
            global_pc: GlobalPcKind::None,
            local_pc: LocalPcKind::Ilu,
            apply_mode: DistLocalApplyMode::WrappedLocal,
        });
        assert!(!cap.native_global_candidate);
        assert_eq!(cap.requested_distributed_mode, "adapter_distributed");
        assert!(cap.requested_mode_supported);
        assert_eq!(cap.requested_mode_failure_reason, None);
        assert!(!cap.supports_native_apply_mode);
    }

    #[test]
    fn native_mode_enables_native_global_candidate_for_local_global_pc() {
        let cap = resolve_distcsr_capability(DistCsrCapabilityKey {
            solver_type: Some(SolverType::Gmres),
            global_pc: GlobalPcKind::BlockJacobi,
            local_pc: LocalPcKind::Ilu,
            apply_mode: DistLocalApplyMode::NativeLocalHalo,
        });
        assert!(cap.native_global_candidate);
        assert_eq!(cap.requested_distributed_mode, "native_distributed");
        assert!(cap.supports_native_distributed_mode);
        assert_eq!(cap.requested_mode_failure_reason, None);
        assert!(cap.supports_native_apply_mode);
    }

    #[test]
    fn explicit_global_pc_is_not_native_block_jacobi_candidate() {
        let cap = resolve_distcsr_capability(DistCsrCapabilityKey {
            solver_type: Some(SolverType::Gmres),
            global_pc: GlobalPcKind::Asm,
            local_pc: LocalPcKind::Ilu,
            apply_mode: DistLocalApplyMode::NativeHybrid,
        });
        assert!(!cap.native_global_candidate);
    }

    #[test]
    fn strict_native_supported_for_spai() {
        let cap = resolve_distcsr_capability(DistCsrCapabilityKey {
            solver_type: Some(SolverType::Gmres),
            global_pc: GlobalPcKind::BlockJacobi,
            local_pc: LocalPcKind::Spai,
            apply_mode: DistLocalApplyMode::NativeStrict,
        });
        assert!(cap.native_global_candidate);
        assert_eq!(cap.requested_distributed_mode, "native_distributed");
        assert!(cap.requested_mode_supported);
        assert_eq!(cap.requested_mode_failure_reason, None);
        assert_eq!(cap.registry_rule, "native_distcsr_capable");
    }

    #[test]
    fn wrapped_local_adapter_rejected_for_incompatible_global_wrapper() {
        let cap = resolve_distcsr_capability(DistCsrCapabilityKey {
            solver_type: Some(SolverType::Gmres),
            global_pc: GlobalPcKind::Asm,
            local_pc: LocalPcKind::Ilu,
            apply_mode: DistLocalApplyMode::WrappedLocal,
        });
        assert!(!cap.requested_mode_supported);
        assert_eq!(
            cap.requested_mode_failure_reason,
            Some("unsupported_global_pc")
        );
        assert_eq!(cap.registry_rule, "strict_mode_unsupported_global_pc");
    }

    #[test]
    fn wrapped_local_adapter_rejected_for_solver_mode_conflict() {
        let cap = resolve_distcsr_capability(DistCsrCapabilityKey {
            solver_type: Some(SolverType::Preonly),
            global_pc: GlobalPcKind::None,
            local_pc: LocalPcKind::Ilu,
            apply_mode: DistLocalApplyMode::WrappedLocal,
        });
        assert!(!cap.requested_mode_supported);
        assert_eq!(
            cap.requested_mode_failure_reason,
            Some("incompatible_solver_mode")
        );
        assert_eq!(cap.registry_rule, "ksp_solver_mode_incompatible");
    }

    #[test]
    fn route_report_tracks_requested_selected_and_budget_state() {
        let mut opts = MpiPcOptions::default();
        opts.local_apply_mode = DistLocalApplyMode::NativeStrict;
        opts.route_policy_budget.native_required = true;
        opts.route_policy_budget.max_allowed_fallbacks = Some(0);
        let report = build_dist_route_decision_report(
            &opts,
            DistRouteSelection::DistCsrNativeBlockJacobi,
            None,
            0,
        );
        assert_eq!(report.requested_mode, "native_distributed");
        assert_eq!(
            report.selected_mode,
            DistRouteSelection::DistCsrNativeBlockJacobi.as_str()
        );
        assert!(report.strict_local_apply);
        assert!(report.native_required);
        assert_eq!(report.max_allowed_fallbacks, Some(0));
    }
}
