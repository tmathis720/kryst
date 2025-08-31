//! Minimal in-crate graph partitioners and helpers for ASM.

/// Contiguous band partition of `0..n` into `nparts` parts.
/// Each part gets roughly `ceil(n / nparts)` rows.
pub fn contiguous_partition(n: usize, nparts: usize) -> Vec<usize> {
    let p = nparts.max(1);
    let chunk = (n + p - 1) / p;
    let mut owner_of = vec![0usize; n];
    for i in 0..n {
        owner_of[i] = (i / chunk).min(p - 1);
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
    for i in 0..n {
        let pid = loads
            .iter()
            .enumerate()
            .min_by_key(|(_, v)| **v)
            .map(|(i, _)| i)
            .unwrap();
        owner_of[i] = pid;
        loads[pid] += nnz[i];
    }
    owner_of
}
