use std::sync::Arc;

use crate::matrix::dist::halo::HaloPlan;
use crate::parallel::UniverseComm;

/// Metadata scaffolding for distributed multigrid hierarchies.
///
/// This captures the pieces needed to keep multilevel operators and halo plans
/// distributed per rank (row partitions per level plus coarse ownership maps).
/// The intent is to let AMG cycles remain local while coarse correction traffic
/// flows through explicit halo exchanges instead of root gathers.
#[derive(Clone)]
pub struct DistHierarchyMeta {
    pub comm: UniverseComm,
    pub row_part: Arc<Vec<usize>>,
    pub level_row_parts: Vec<Arc<Vec<usize>>>,
    pub level_halos: Vec<Arc<HaloPlan>>,
    pub coarse_owners: Vec<usize>,
    pub coarse_offsets: Vec<usize>,
    pub coarse_global_size: usize,
}
