use crate::error::KError;
use std::str::FromStr;

/// High-level route selection policy for distributed preconditioners.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DistRoutePolicy {
    /// Prefer native distributed operator routes.
    Native,
    /// Prefer compatibility adapter routes over native distributed kernels.
    Adapted,
    /// Prefer root-gather routes for coarse/global correction phases.
    RootGather,
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
    use super::DistLocalApplyMode;
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
}
