use super::*;
use crate::preconditioner::amg::coarsen::{build_aggregates, AggAlgo, AggOpts};
use crate::preconditioner::amg::strength::Strength;
use std::collections::VecDeque;

fn path_strength(n: usize) -> Strength {
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::new();
    row_ptr.push(0);
    for i in 0..n {
        if i > 0 {
            col_idx.push(i - 1);
        }
        if i + 1 < n {
            col_idx.push(i + 1);
        }
        row_ptr.push(col_idx.len());
    }
    Strength { row_ptr, col_idx }
}

fn bfs_dist(s: &Strength, start: usize, goal: usize) -> usize {
    if start == goal {
        return 0;
    }
    let n = s.row_ptr.len() - 1;
    let mut dist = vec![usize::MAX; n];
    let mut q = VecDeque::new();
    dist[start] = 0;
    q.push_back(start);
    while let Some(u) = q.pop_front() {
        let d = dist[u];
        let rs = s.row_ptr[u];
        let re = s.row_ptr[u + 1];
        for &v in &s.col_idx[rs..re] {
            if dist[v] == usize::MAX {
                dist[v] = d + 1;
                if v == goal {
                    return dist[v];
                }
                q.push_back(v);
            }
        }
    }
    usize::MAX
}

#[test]
fn mis_independence_and_determinism() {
    let s = path_strength(10);
    for algo in [AggAlgo::PMIS, AggAlgo::HMIS] {
        for &k in &[1usize, 2usize] {
            let opts = AggOpts {
                mis_k: k,
                cap_per_row: None,
            };
            let (agg1, _) = build_aggregates(&s, algo, &opts);
            let (agg2, _) = build_aggregates(&s, algo, &opts);
            assert_eq!(agg1, agg2);
            let n_coarse = 1 + agg1.iter().copied().max().unwrap_or(0);
            let mut seeds = Vec::new();
            let mut seen = vec![false; n_coarse];
            for (i, &a) in agg1.iter().enumerate() {
                if !seen[a] {
                    seeds.push(i);
                    seen[a] = true;
                }
                assert!(a < n_coarse);
            }
            for i in 0..seeds.len() {
                for j in (i + 1)..seeds.len() {
                    let d = bfs_dist(&s, seeds[i], seeds[j]);
                    assert!(d > k, "seeds {}/{} too close: {}", seeds[i], seeds[j], d);
                }
            }
        }
    }
}
