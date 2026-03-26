use super::Global;
use crate::matrix::dist_csr::DistCsrOp;
use crate::parallel::{Comm, UniverseComm};

/// Partition `n_global` rows among all ranks in `comm`.
///
/// Returns a vector `row_starts` of length `p + 1` where `p = comm.size()`.
/// Row `k` on rank `r` spans `row_starts[r] .. row_starts[r+1]`.
#[deprecated(
    since = "1.1.0",
    note = "Use DistCsrOp::partition_rows_balanced; legacy parcsr partition helpers are compatibility-only and planned for removal after 2026-12-31"
)]
pub fn partition_rows(n_global: Global, comm: &UniverseComm) -> Vec<Global> {
    DistCsrOp::partition_rows_balanced(n_global as usize, comm)
        .into_iter()
        .map(|v| v as Global)
        .collect()
}

/// Core implementation taking the number of partitions `p` directly.
#[deprecated(
    since = "1.1.0",
    note = "Use DistCsrOp::partition_rows_balanced; legacy parcsr partition helpers are compatibility-only and planned for removal after 2026-12-31"
)]
pub fn partition_rows_size(n_global: Global, p: usize) -> Vec<Global> {
    assert!(p > 0, "number of partitions must be positive");
    let comm_size = p;
    let base = n_global as usize / comm_size;
    let rem = n_global as usize % comm_size;
    let mut starts = Vec::with_capacity(comm_size + 1);
    let mut s: usize = 0;
    for k in 0..comm_size {
        starts.push(s as Global);
        s += base + usize::from(k < rem);
    }
    starts.push(n_global);
    starts
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_props(part: &[Global], n_global: Global) {
        assert_eq!(part.first().copied(), Some(0));
        assert_eq!(part.last().copied(), Some(n_global));
        assert!(part.windows(2).all(|w| w[0] <= w[1]));
        let total: Global = part.windows(2).map(|w| w[1] - w[0]).sum();
        assert_eq!(total, n_global);
    }

    #[test]
    fn partitions_even() {
        let part = partition_rows_size(8, 4);
        assert_eq!(part, vec![0, 2, 4, 6, 8]);
        check_props(&part, 8);
    }

    #[test]
    fn partitions_uneven() {
        let part = partition_rows_size(10, 3);
        assert_eq!(part, vec![0, 4, 7, 10]);
        check_props(&part, 10);
    }

    #[test]
    fn partitions_zero_rows() {
        let part = partition_rows_size(0, 4);
        assert_eq!(part, vec![0, 0, 0, 0, 0]);
        check_props(&part, 0);
    }
}
