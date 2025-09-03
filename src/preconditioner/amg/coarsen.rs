use super::strength::Strength;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AggAlgo {
    PMIS,
    HMIS,
    Falgout,
    RSGreedy,
}

pub struct AggOpts {
    pub mis_k: usize,
    pub cap_per_row: Option<usize>,
}

/// Deterministic MIS-k on an undirected graph.
fn mis_k(s: &Strength, k: usize, prio: &[u64]) -> Vec<bool> {
    assert!(k >= 1);
    let n = s.row_ptr.len() - 1;
    let mut removed = vec![false; n];
    let mut seed = vec![false; n];
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by(|&a, &b| prio[b].cmp(&prio[a]).then_with(|| b.cmp(&a)));
    let mut q = Vec::<usize>::new();
    let mut vis = vec![usize::MAX; n];
    let mut tick = 0usize;
    for &u in &order {
        if removed[u] {
            continue;
        }
        seed[u] = true;
        tick += 1;
        q.clear();
        vis[u] = 0;
        q.push(u);
        while let Some(x) = q.pop() {
            removed[x] = true;
            let depth = vis[x];
            if depth == k {
                continue;
            }
            let rs = s.row_ptr[x];
            let re = s.row_ptr[x + 1];
            for &v in &s.col_idx[rs..re] {
                if vis[v] != tick && vis[v] > depth + 1 {
                    vis[v] = depth + 1;
                    q.push(v);
                }
            }
        }
    }
    seed
}

#[inline]
fn prio_degree(i: usize, deg: usize) -> u64 {
    ((deg as u64) << 32) | (u32::MAX as u64 - i as u64)
}

#[inline]
fn prio_hmis(i: usize, deg1: usize, deg2: usize) -> u64 {
    ((deg2 as u64) << 42) | ((deg1 as u64) << 32) | (u32::MAX as u64 - i as u64)
}

fn aggregates_from_seeds(s: &Strength, is_seed: &[bool]) -> Vec<usize> {
    let n = s.row_ptr.len() - 1;
    let mut agg = vec![usize::MAX; n];
    let mut next = 0usize;
    for i in 0..n {
        if is_seed[i] {
            agg[i] = next;
            next += 1;
        }
    }
    for i in 0..n {
        if agg[i] != usize::MAX {
            continue;
        }
        let rs = s.row_ptr[i];
        let re = s.row_ptr[i + 1];
        let mut best: Option<usize> = None;
        for &j in &s.col_idx[rs..re] {
            if is_seed[j] {
                best = Some(match best {
                    Some(b) => b.min(j),
                    None => j,
                });
            }
        }
        if let Some(seed) = best {
            agg[i] = agg[seed];
            continue;
        }
        'twohop: {
            for &j in &s.col_idx[rs..re] {
                let rs2 = s.row_ptr[j];
                let re2 = s.row_ptr[j + 1];
                for &k in &s.col_idx[rs2..re2] {
                    if is_seed[k] {
                        agg[i] = agg[k];
                        break 'twohop;
                    }
                }
            }
        }
        if agg[i] == usize::MAX {
            agg[i] = {
                let id = next;
                next += 1;
                id
            };
        }
    }
    agg
}

/// Build aggregates from a strength graph. Returns fine -> aggregate id.
pub fn build_aggregates(s_in: &Strength, algo: AggAlgo, opts: &AggOpts) -> (Vec<usize>, Vec<bool>) {
    let mut s = s_in.symmetrize();
    if let Some(k) = opts.cap_per_row {
        s = s.capped(k);
    }
    let n = s.row_ptr.len() - 1;

    if matches!(algo, AggAlgo::RSGreedy) {
        return rs_greedy(&s);
    }

    let mut deg1 = vec![0usize; n];
    let mut deg2 = vec![0usize; n];
    for i in 0..n {
        let rs = s.row_ptr[i];
        let re = s.row_ptr[i + 1];
        let di = re - rs;
        deg1[i] = di;
        let mut sum = 0usize;
        for &j in &s.col_idx[rs..re] {
            sum += s.row_ptr[j + 1] - s.row_ptr[j];
        }
        deg2[i] = sum;
    }
    let mut prio = vec![0u64; n];
    match algo {
        AggAlgo::PMIS | AggAlgo::Falgout => {
            for i in 0..n {
                prio[i] = prio_degree(i, deg1[i]);
            }
        }
        AggAlgo::HMIS => {
            for i in 0..n {
                prio[i] = prio_hmis(i, deg1[i], deg2[i]);
            }
        }
        AggAlgo::RSGreedy => {}
    }
    let mut is_seed = mis_k(&s, opts.mis_k.max(1), &prio);

    if matches!(algo, AggAlgo::Falgout) {
        for i in 0..n {
            if is_seed[i] {
                continue;
            }
            let rs = s.row_ptr[i];
            let re = s.row_ptr[i + 1];
            let mut has = false;
            for &j in &s.col_idx[rs..re] {
                if is_seed[j] {
                    has = true;
                    break;
                }
            }
            if !has {
                let mut best = i;
                let mut bestp = prio[i];
                for &j in &s.col_idx[rs..re] {
                    if prio[j] > bestp {
                        best = j;
                        bestp = prio[j];
                    }
                }
                is_seed[best] = true;
            }
        }
    }

    let agg = aggregates_from_seeds(&s, &is_seed);
    (agg, is_seed)
}

// Legacy greedy aggregator kept for compatibility.
fn rs_greedy(s: &Strength) -> (Vec<usize>, Vec<bool>) {
    let n = s.row_ptr.len() - 1;
    let mut agg = vec![usize::MAX; n];
    let mut is_seed = vec![false; n];
    let mut next = 0usize;
    let max_sz = 4usize;
    let mut order: Vec<(usize, usize)> = (0..n)
        .map(|i| (s.row_ptr[i + 1] - s.row_ptr[i], i))
        .collect();
    order.sort_by(|a, b| b.0.cmp(&a.0));
    for &(_, seed) in &order {
        if agg[seed] != usize::MAX {
            continue;
        }
        agg[seed] = next;
        is_seed[seed] = true;
        let rs = s.row_ptr[seed];
        let re = s.row_ptr[seed + 1];
        let mut neigh: Vec<(usize, usize)> = s.col_idx[rs..re]
            .iter()
            .copied()
            .filter(|&j| agg[j] == usize::MAX)
            .map(|j| (s.row_ptr[j + 1] - s.row_ptr[j], j))
            .collect();
        neigh.sort_by(|a, b| b.0.cmp(&a.0));
        let mut count = 1usize;
        for &(_, j) in &neigh {
            if count >= max_sz {
                break;
            }
            agg[j] = next;
            count += 1;
        }
        next += 1;
    }
    for i in 0..n {
        if agg[i] == usize::MAX {
            agg[i] = {
                let id = next;
                next += 1;
                id
            };
            is_seed[i] = true;
        }
    }
    (agg, is_seed)
}
