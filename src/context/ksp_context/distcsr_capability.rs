use crate::context::ksp_context::SolverType;
use crate::preconditioner::dist::{DistLocalApplyMode, GlobalPcKind, LocalPcKind};
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
    pub supports_native_apply_mode: bool,
    pub registry_rule: &'static str,
}

pub fn resolve_distcsr_capability(key: DistCsrCapabilityKey) -> DistCsrCapabilityEntry {
    let supports_native_apply_mode = key.apply_mode.is_distributed_native();

    let (native_global_candidate, registry_rule) =
        match (key.global_pc, key.apply_mode, key.solver_type, key.local_pc) {
            (
                GlobalPcKind::None | GlobalPcKind::BlockJacobi,
                DistLocalApplyMode::WrappedLocal,
                _,
                _,
            ) => (
                false,
                "global_candidate_block_jacobi_but_wrapped_local_apply",
            ),
            (GlobalPcKind::None | GlobalPcKind::BlockJacobi, _, _, _) => {
                (true, "global_candidate_block_jacobi_native_apply")
            }
            (GlobalPcKind::Asm | GlobalPcKind::Ras, _, _, _) => (false, "explicit_global_pc_route"),
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
        supports_native_apply_mode,
        registry_rule,
    }
}

#[cfg(test)]
mod tests {
    use super::{DistCsrCapabilityKey, resolve_distcsr_capability};
    use crate::context::ksp_context::SolverType;
    use crate::preconditioner::dist::{DistLocalApplyMode, GlobalPcKind, LocalPcKind};

    #[test]
    fn wrapped_local_disables_native_global_candidate() {
        let cap = resolve_distcsr_capability(DistCsrCapabilityKey {
            solver_type: Some(SolverType::Gmres),
            global_pc: GlobalPcKind::None,
            local_pc: LocalPcKind::Ilu,
            apply_mode: DistLocalApplyMode::WrappedLocal,
        });
        assert!(!cap.native_global_candidate);
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
}
