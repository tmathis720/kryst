use crate::error::KError;
use serde::Serialize;
use std::str::FromStr;

/// High-level route selection policy for distributed preconditioners.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DistRoutePolicy {
    /// Prefer native distributed operator routes.
    Native,
    /// Force compatibility adapter routes (fallback-only mode).
    Adapted,
    /// Prefer root-gather routes for coarse/global correction phases.
    RootGather,
}

/// Structured fallback reasons for distributed route selection diagnostics.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DistRouteFallbackReason {
    AutoPromotedFromLocal,
    NativeSetupFailed,
    ConfiguredGlobalFallback,
    AdapterOnlyPolicy,
    RootGatherPolicy,
    MissingDistCsrOperator,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[derive(Default)]
pub struct DistRoutePolicyBudget {
    pub max_allowed_fallbacks: Option<usize>,
    pub native_required: bool,
}


#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct DistRouteDecisionReport {
    pub requested_mode: &'static str,
    pub selected_mode: &'static str,
    pub fallback_reason: Option<String>,
    pub strict_local_apply: bool,
    pub native_required: bool,
    pub fallback_chain_len: usize,
    pub max_allowed_fallbacks: Option<usize>,
}

impl DistRouteDecisionReport {
    pub fn fallback_budget_exceeded(&self) -> bool {
        self.max_allowed_fallbacks
            .is_some_and(|max| self.fallback_chain_len > max)
    }
}

pub fn validate_dist_route_policy_budget(
    report: &DistRouteDecisionReport,
    fallback_chain: &[String],
) -> Result<(), KError> {
    if report.native_required
        && report.selected_mode != DistRouteSelection::DistCsrNativeBlockJacobi.as_str()
    {
        return Err(KError::InvalidInput(format!(
            "distributed route policy requires native mode, but selected={} requested={} fallback_reason={}",
            report.selected_mode,
            report.requested_mode,
            report
                .fallback_reason
                .clone()
                .unwrap_or_else(|| "none".to_string())
        )));
    }

    if report.fallback_budget_exceeded() {
        return Err(KError::InvalidInput(format!(
            "distributed route fallback budget exceeded: used={} max={} chain={:?}",
            report.fallback_chain_len,
            report.max_allowed_fallbacks.unwrap_or(0),
            fallback_chain
        )));
    }
    Ok(())
}

impl DistRouteFallbackReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::AutoPromotedFromLocal => "auto_promoted_from_local",
            Self::NativeSetupFailed => "native_setup_failed",
            Self::ConfiguredGlobalFallback => "configured_global_fallback",
            Self::AdapterOnlyPolicy => "adapter_only_policy",
            Self::RootGatherPolicy => "root_gather_policy",
            Self::MissingDistCsrOperator => "missing_distcsr_operator",
        }
    }
}

impl Default for DistRoutePolicy {
    fn default() -> Self {
        Self::Native
    }
}

impl FromStr for DistRoutePolicy {
    type Err = KError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_lowercase().as_str() {
            "native" | "distributed_native" => Ok(Self::Native),
            "adapted" | "adapter" | "wrapped_local" => Ok(Self::Adapted),
            "root" | "root_gather" | "gather" => Ok(Self::RootGather),
            other => Err(KError::InvalidInput(format!(
                "invalid pc_dist_route mode: {other}"
            ))),
        }
    }
}

/// Canonical route labels emitted by distributed route selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DistRouteSelection {
    DistCsrNativeBlockJacobi,
    ConfiguredGlobal,
    LocalAdapter,
    RootGather,
}

impl DistRouteSelection {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::DistCsrNativeBlockJacobi => "distcsr_native_block_jacobi",
            Self::ConfiguredGlobal => "configured_global",
            Self::LocalAdapter => "local_adapter",
            Self::RootGather => "root_gather",
        }
    }
}

/// Structured reasons for why a route was accepted or rejected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DistRouteDecisionReason {
    DistCsrMissing,
    RoutePolicyAdapted,
    RoutePolicyRootGather,
    ExplicitGlobalRequested,
    LocalPcSupportsDistributed,
    NativeLocalApplyUnavailable,
    NativeRouteEligible,
}

impl DistRouteDecisionReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::DistCsrMissing => "distcsr_missing",
            Self::RoutePolicyAdapted => "route_policy_adapted",
            Self::RoutePolicyRootGather => "route_policy_root_gather",
            Self::ExplicitGlobalRequested => "explicit_global_requested",
            Self::LocalPcSupportsDistributed => "local_pc_supports_distributed",
            Self::NativeLocalApplyUnavailable => "native_local_apply_unavailable",
            Self::NativeRouteEligible => "native_route_eligible",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DistRouteDecision {
    pub selected: DistRouteSelection,
    pub accepted: Vec<DistRouteDecisionReason>,
    pub rejected: Vec<DistRouteDecisionReason>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DistRouteResolveInput {
    pub has_distcsr: bool,
    pub explicit_global: bool,
    pub native_global_candidate: bool,
    pub route_policy: DistRoutePolicy,
    pub local_only_pc: bool,
    pub local_apply_mode: DistLocalApplyMode,
}

pub fn resolve_dist_route(input: DistRouteResolveInput) -> DistRouteDecision {
    let mut accepted = Vec::new();
    let mut rejected = Vec::new();

    if !input.has_distcsr {
        rejected.push(DistRouteDecisionReason::DistCsrMissing);
    }

    if input.route_policy == DistRoutePolicy::Adapted {
        rejected.push(DistRouteDecisionReason::RoutePolicyAdapted);
    }
    if input.route_policy == DistRoutePolicy::RootGather {
        rejected.push(DistRouteDecisionReason::RoutePolicyRootGather);
    }
    if input.explicit_global {
        accepted.push(DistRouteDecisionReason::ExplicitGlobalRequested);
    }
    if !input.local_only_pc {
        accepted.push(DistRouteDecisionReason::LocalPcSupportsDistributed);
    }
    if !input.local_apply_mode.is_distributed_native() {
        rejected.push(DistRouteDecisionReason::NativeLocalApplyUnavailable);
    }

    let native_policy = !matches!(
        input.route_policy,
        DistRoutePolicy::Adapted | DistRoutePolicy::RootGather
    );
    let native_eligible = input.has_distcsr && input.native_global_candidate && native_policy;
    if native_eligible {
        accepted.push(DistRouteDecisionReason::NativeRouteEligible);
        return DistRouteDecision {
            selected: DistRouteSelection::DistCsrNativeBlockJacobi,
            accepted,
            rejected,
        };
    }

    if input.explicit_global {
        return DistRouteDecision {
            selected: DistRouteSelection::ConfiguredGlobal,
            accepted,
            rejected,
        };
    }

    if input.route_policy == DistRoutePolicy::RootGather {
        return DistRouteDecision {
            selected: DistRouteSelection::RootGather,
            accepted,
            rejected,
        };
    }

    DistRouteDecision {
        selected: DistRouteSelection::LocalAdapter,
        accepted,
        rejected,
    }
}

/// Controls whether distributed block-Jacobi uses a pure local wrapper
/// or a distributed-native apply path with neighborhood coupling exchange.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DistLocalApplyMode {
    /// Legacy behavior: apply local factors only.
    WrappedLocal,
    /// Distributed-native behavior with neighborhood halo exchange only.
    NativeLocalHalo,
    /// Distributed-native behavior with halo exchange and optional coarse coupling.
    NativeHybrid,
    /// Require distributed-native apply, failing setup when unavailable.
    NativeStrict,
}

impl Default for DistLocalApplyMode {
    fn default() -> Self {
        Self::NativeLocalHalo
    }
}

impl DistLocalApplyMode {
    pub fn communication_strategy_name(self) -> &'static str {
        match self {
            Self::WrappedLocal => "local",
            Self::NativeLocalHalo => "local-halo",
            Self::NativeHybrid => "hybrid",
            Self::NativeStrict => "strict",
        }
    }

    pub fn is_distributed_native(self) -> bool {
        matches!(
            self,
            Self::NativeLocalHalo | Self::NativeHybrid | Self::NativeStrict
        )
    }

    pub fn requires_native(self) -> bool {
        matches!(self, Self::NativeStrict)
    }
}

impl FromStr for DistLocalApplyMode {
    type Err = KError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_lowercase().as_str() {
            "local" | "local_wrapper" | "wrapped_local" | "wrapper" => Ok(Self::WrappedLocal),
            "distributed" | "distributed_native" | "native" | "local-halo" | "halo"
            | "distributed_halo" => Ok(Self::NativeLocalHalo),
            "hybrid" | "distributed_hybrid" | "native_hybrid" => Ok(Self::NativeHybrid),
            "distributed_strict" | "strict" | "native_strict" => Ok(Self::NativeStrict),
            other => Err(KError::InvalidInput(format!(
                "invalid pc_dist_local_apply mode: {other}"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DistLocalApplyMode, DistRouteDecisionReason, DistRouteDecisionReport,
        DistRouteFallbackReason, DistRoutePolicy, DistRouteResolveInput, DistRouteSelection,
        resolve_dist_route, validate_dist_route_policy_budget,
    };
    use std::str::FromStr;

    #[test]
    fn parse_modes_and_capabilities() {
        let wrapped = DistLocalApplyMode::from_str("wrapped_local").expect("wrapped");
        assert_eq!(wrapped, DistLocalApplyMode::WrappedLocal);
        assert!(!wrapped.is_distributed_native());
        assert!(!wrapped.requires_native());

        let native = DistLocalApplyMode::from_str("distributed_native").expect("native");
        assert_eq!(native, DistLocalApplyMode::NativeLocalHalo);
        assert!(native.is_distributed_native());
        assert!(!native.requires_native());

        let hybrid = DistLocalApplyMode::from_str("hybrid").expect("hybrid");
        assert_eq!(hybrid, DistLocalApplyMode::NativeHybrid);
        assert!(hybrid.is_distributed_native());
        assert!(!hybrid.requires_native());

        let strict = DistLocalApplyMode::from_str("strict").expect("strict");
        assert_eq!(strict, DistLocalApplyMode::NativeStrict);
        assert!(strict.is_distributed_native());
        assert!(strict.requires_native());
    }

    #[test]
    fn resolve_dist_route_native_and_policy_routes() {
        let native = resolve_dist_route(DistRouteResolveInput {
            has_distcsr: true,
            explicit_global: false,
            native_global_candidate: true,
            route_policy: DistRoutePolicy::Native,
            local_only_pc: true,
            local_apply_mode: DistLocalApplyMode::NativeLocalHalo,
        });
        assert_eq!(
            native.selected,
            DistRouteSelection::DistCsrNativeBlockJacobi
        );
        assert!(
            native
                .accepted
                .contains(&DistRouteDecisionReason::NativeRouteEligible)
        );

        let adapted = resolve_dist_route(DistRouteResolveInput {
            has_distcsr: true,
            explicit_global: false,
            native_global_candidate: true,
            route_policy: DistRoutePolicy::Adapted,
            local_only_pc: true,
            local_apply_mode: DistLocalApplyMode::WrappedLocal,
        });
        assert_eq!(adapted.selected, DistRouteSelection::LocalAdapter);
        assert!(
            adapted
                .rejected
                .contains(&DistRouteDecisionReason::RoutePolicyAdapted)
        );

        let root = resolve_dist_route(DistRouteResolveInput {
            has_distcsr: false,
            explicit_global: false,
            native_global_candidate: false,
            route_policy: DistRoutePolicy::RootGather,
            local_only_pc: true,
            local_apply_mode: DistLocalApplyMode::WrappedLocal,
        });
        assert_eq!(root.selected, DistRouteSelection::RootGather);
        assert!(
            root.rejected
                .contains(&DistRouteDecisionReason::RoutePolicyRootGather)
        );
    }

    #[test]
    fn fallback_reason_keys_are_stable() {
        assert_eq!(
            DistRouteFallbackReason::AutoPromotedFromLocal.as_str(),
            "auto_promoted_from_local"
        );
        assert_eq!(
            DistRouteFallbackReason::NativeSetupFailed.as_str(),
            "native_setup_failed"
        );
    }

    #[test]
    fn budget_check_enforces_native_only_policy() {
        let report = DistRouteDecisionReport {
            requested_mode: "native_distributed",
            selected_mode: DistRouteSelection::LocalAdapter.as_str(),
            fallback_reason: Some("adapter_only_policy".to_string()),
            strict_local_apply: false,
            native_required: true,
            fallback_chain_len: 1,
            max_allowed_fallbacks: Some(3),
        };
        let err = validate_dist_route_policy_budget(&report, &["adapter_only_policy".to_string()])
            .expect_err("native-required policy should fail for adapted route");
        assert!(err.to_string().contains("requires native mode"));
    }

    #[test]
    fn budget_check_enforces_max_fallbacks() {
        let report = DistRouteDecisionReport {
            requested_mode: "native_distributed",
            selected_mode: DistRouteSelection::DistCsrNativeBlockJacobi.as_str(),
            fallback_reason: None,
            strict_local_apply: false,
            native_required: false,
            fallback_chain_len: 2,
            max_allowed_fallbacks: Some(1),
        };
        let chain = vec![
            "auto_promoted_from_local".to_string(),
            "native_setup_failed".to_string(),
        ];
        let err = validate_dist_route_policy_budget(&report, &chain)
            .expect_err("fallback budget should be enforced");
        assert!(err.to_string().contains("fallback budget exceeded"));
    }

    #[test]
    fn budget_check_accepts_auto_fallback_within_budget() {
        let report = DistRouteDecisionReport {
            requested_mode: "native_distributed",
            selected_mode: DistRouteSelection::DistCsrNativeBlockJacobi.as_str(),
            fallback_reason: None,
            strict_local_apply: false,
            native_required: false,
            fallback_chain_len: 1,
            max_allowed_fallbacks: Some(2),
        };
        let chain = vec!["auto_promoted_from_local".to_string()];
        validate_dist_route_policy_budget(&report, &chain)
            .expect("auto fallback should succeed within budget");
    }
}
