use crate::parallel::{Comm, UniverseComm};
use super::Global;

/// Partition `n_global` rows among all ranks in `comm`.
///
/// Returns a vector `row_starts` of length `p + 1` where `p = comm.size()`.
/// Row `k` on rank `r` spans `row_starts[r] .. row_starts[r+1]`.
pub fn partition_rows(n_global: Global, comm: &UniverseComm) -> Vec<Global> {
    partition_rows_size(n_global, comm.size())
}

/// Core implementation taking the number of partitions `p` directly.
pub fn partition_rows_size(n_global: Global, p: usize) -> Vec<Global> {
    assert!(p > 0, "number of partitions must be positive");
    let base = n_global / p as Global;
    let rem = n_global % p as Global;
    let mut starts = Vec::with_capacity(p + 1);
    let mut s: Global = 0;
    for k in 0..p {
        starts.push(s);
        s += base + if (k as Global) < rem { 1 } else { 0 };
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
