//! Minimal in-crate graph partitioners and helpers for ASM.

/// Contiguous band partition of `0..n` into `nparts` parts.
/// Each part gets roughly `ceil(n / nparts)` rows.
pub fn contiguous_partition(n: usize, nparts: usize) -> Vec<usize> {
    let p = nparts.max(1);
    let chunk = n.div_ceil(p);
    let mut owner_of = vec![0usize; n];
    for (i, owner) in owner_of.iter_mut().enumerate() {
        *owner = (i / chunk).min(p - 1);
    }
    owner_of
}

/// Greedy row-nnz balanced partition. Optional `nnz_per_row` guides balancing.
/// Falls back to contiguous when `nnz_per_row` is not provided.
pub fn greedy_nnz_balanced_partition(n: usize, nparts: usize, nnz_per_row: Option<&[usize]>) -> Vec<usize> {
    if nnz_per_row.is_none() { return contiguous_partition(n, nparts); }
    let nnz = nnz_per_row.unwrap();
    let p = nparts.max(1);
    let mut owner_of = vec![0usize; n];
    let mut loads = vec![0usize; p];
    for (i, owner) in owner_of.iter_mut().enumerate() {
        let pid = loads
            .iter()
            .enumerate()
            .min_by_key(|(_, v)| **v)
            .map(|(i, _)| i)
            .unwrap();
        *owner = pid;
        loads[pid] += nnz[i];
    }
    owner_of
}
