use crate::error::KError;
use std::fmt;
use std::str::FromStr;

/// Unified coarse-level strategy for distributed preconditioners.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DistCoarseStrategy {
    /// No coarse correction (rank-local only).
    None,
    /// Gather to rank 0 and solve on the root.
    RootGather,
    /// Per-rank local prototype (optionally with halo correction).
    LocalPrototype,
    /// External distributed backend (e.g., SuperLU_DIST) when available.
    SuperLuDist,
}

impl DistCoarseStrategy {
    pub fn is_rank_local(self) -> bool {
        matches!(self, DistCoarseStrategy::None | DistCoarseStrategy::LocalPrototype)
    }

    pub fn is_collective(self) -> bool {
        matches!(self, DistCoarseStrategy::RootGather | DistCoarseStrategy::SuperLuDist)
    }
}

impl Default for DistCoarseStrategy {
    fn default() -> Self {
        DistCoarseStrategy::RootGather
    }
}

impl fmt::Display for DistCoarseStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            DistCoarseStrategy::None => "none",
            DistCoarseStrategy::RootGather => "root_gather",
            DistCoarseStrategy::LocalPrototype => "local_prototype",
            DistCoarseStrategy::SuperLuDist => "superlu_dist",
        };
        write!(f, "{label}")
    }
}

impl FromStr for DistCoarseStrategy {
    type Err = KError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_lowercase().as_str() {
            "none" | "off" => Ok(DistCoarseStrategy::None),
            "root" | "root_gather" | "gather" => Ok(DistCoarseStrategy::RootGather),
            "local" | "local_prototype" | "prototype" => Ok(DistCoarseStrategy::LocalPrototype),
            "superlu_dist" | "superludist" => Ok(DistCoarseStrategy::SuperLuDist),
            other => Err(KError::InvalidInput(format!(
                "invalid dist coarse strategy: {other}"
            ))),
        }
    }
}
