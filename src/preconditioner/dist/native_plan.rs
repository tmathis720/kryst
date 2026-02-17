use crate::error::KError;
use std::str::FromStr;

/// Controls whether distributed block-Jacobi uses a pure local wrapper
/// or a distributed-native apply path with neighborhood coupling exchange.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DistLocalApplyMode {
    /// Legacy behavior: apply local factors only.
    WrappedLocal,
    /// Distributed-native behavior: local factors plus halo-coupling correction.
    NativeDistributed,
    /// Require distributed-native apply, failing setup when unavailable.
    NativeStrict,
}

impl Default for DistLocalApplyMode {
    fn default() -> Self {
        Self::WrappedLocal
    }
}

impl DistLocalApplyMode {
    pub fn is_distributed_native(self) -> bool {
        matches!(self, Self::NativeDistributed | Self::NativeStrict)
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
            "distributed" | "distributed_native" | "native" => Ok(Self::NativeDistributed),
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
        assert_eq!(native, DistLocalApplyMode::NativeDistributed);
        assert!(native.is_distributed_native());
        assert!(!native.requires_native());

        let strict = DistLocalApplyMode::from_str("strict").expect("strict");
        assert_eq!(strict, DistLocalApplyMode::NativeStrict);
        assert!(strict.is_distributed_native());
        assert!(strict.requires_native());
    }
}
