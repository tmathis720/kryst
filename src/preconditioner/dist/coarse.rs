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

/// Explicit coarse ownership policy for distributed coarse-grid operators/solves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DistCoarsePolicy {
    /// Gather coarse system/solve state to root rank only.
    RootGather,
    /// Replicate coarse system state on all ranks.
    Replicated,
    /// Route through an external distributed backend.
    External,
}

impl Default for DistCoarsePolicy {
    fn default() -> Self {
        Self::RootGather
    }
}

impl FromStr for DistCoarsePolicy {
    type Err = KError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_lowercase().as_str() {
            "root" | "root_gather" | "gather" => Ok(Self::RootGather),
            "replicated" | "replicate" | "all" => Ok(Self::Replicated),
            "external" | "backend" | "superlu_dist" => Ok(Self::External),
            other => Err(KError::InvalidInput(format!(
                "invalid dist coarse policy: {other}"
            ))),
        }
    }
}

/// Policy for repartitioning the coarse grid in distributed AMG setups.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DistCoarseRepartition {
    /// Preserve the fine-grid ownership partition.
    Keep,
    /// Build an even contiguous row partition over all ranks.
    Uniform,
    /// Route setup through a root-owned coarse partition.
    Root,
}

impl Default for DistCoarseRepartition {
    fn default() -> Self {
        Self::Keep
    }
}

impl FromStr for DistCoarseRepartition {
    type Err = KError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_lowercase().as_str() {
            "keep" | "fine" | "inherit" => Ok(Self::Keep),
            "uniform" | "block" | "contiguous" => Ok(Self::Uniform),
            "hybrid" | "local" => Ok(Self::Uniform),
            "root" | "root_owned" => Ok(Self::Root),
            other => Err(KError::InvalidInput(format!(
                "invalid dist coarse repartition policy: {other}"
            ))),
        }
    }
}

/// Explicit route for the coarse solve backend in distributed AMG.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DistCoarseSolverRoute {
    /// Choose the backend implied by other settings.
    Auto,
    /// Force gathered root-level solve path.
    Root,
    /// Force local prototype coarse path (fallback/prototyping).
    Local,
    /// Force SuperLU_DIST coarse routing when available.
    SuperLuDist,
}

impl Default for DistCoarseSolverRoute {
    fn default() -> Self {
        Self::Auto
    }
}

impl FromStr for DistCoarseSolverRoute {
    type Err = KError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "root" | "root_gather" | "gather" => Ok(Self::Root),
            "local" | "local_prototype" | "hybrid" => Ok(Self::Local),
            "superlu_dist" | "superludist" => Ok(Self::SuperLuDist),
            other => Err(KError::InvalidInput(format!(
                "invalid dist coarse solver route: {other}"
            ))),
        }
    }
}

impl DistCoarseStrategy {
    pub fn is_rank_local(self) -> bool {
        matches!(
            self,
            DistCoarseStrategy::None | DistCoarseStrategy::LocalPrototype
        )
    }

    pub fn is_collective(self) -> bool {
        matches!(
            self,
            DistCoarseStrategy::RootGather | DistCoarseStrategy::SuperLuDist
        )
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
            "local" | "local_prototype" | "prototype" | "hybrid" => {
                Ok(DistCoarseStrategy::LocalPrototype)
            }
            "superlu_dist" | "superludist" => Ok(DistCoarseStrategy::SuperLuDist),
            other => Err(KError::InvalidInput(format!(
                "invalid dist coarse strategy: {other}"
            ))),
        }
    }
}
