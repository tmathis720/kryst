use crate::error::KError;
use std::str::FromStr;

/// Controls whether distributed block-Jacobi uses a pure local wrapper
/// or a distributed-native apply path with neighborhood coupling exchange.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DistLocalApplyMode {
    /// Legacy behavior: apply local factors only.
    LocalWrapper,
    /// Distributed-native behavior: local factors plus halo-coupling correction.
    DistributedNative,
}

impl Default for DistLocalApplyMode {
    fn default() -> Self {
        Self::LocalWrapper
    }
}

impl DistLocalApplyMode {
    pub fn is_distributed_native(self) -> bool {
        matches!(self, Self::DistributedNative)
    }
}

impl FromStr for DistLocalApplyMode {
    type Err = KError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_lowercase().as_str() {
            "local" | "local_wrapper" | "wrapper" => Ok(Self::LocalWrapper),
            "distributed" | "distributed_native" | "native" => Ok(Self::DistributedNative),
            other => Err(KError::InvalidInput(format!(
                "invalid pc_dist_local_apply mode: {other}"
            ))),
        }
    }
}
