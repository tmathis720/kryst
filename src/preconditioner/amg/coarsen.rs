use super::strength::Strength;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AggAlgo { PMIS, HMIS, RSGreedy }

/// Build aggregates from a strength graph. Returns fine -> aggregate id.
pub fn build_aggregates(s: &Strength, algo: AggAlgo) -> Vec<usize> {
    match algo {
        AggAlgo::RSGreedy => rs_greedy(s),
        AggAlgo::PMIS | AggAlgo::HMIS => rs_greedy(s), // placeholder: robust greedy default
    }
}

/// Simple greedy aggregation: each unassigned node seeds an aggregate and we
/// attach its strongest neighbors up to a small cap.
fn rs_greedy(s: &Strength) -> Vec<usize> {
    let n = s.row_ptr.len() - 1;
    let mut agg = vec![usize::MAX; n];
    let mut next = 0usize;
    let max_sz = 4usize;

    // Order nodes by degree descending to seed aggregates on hubs first
    let mut order: Vec<(usize, usize)> = (0..n)
        .map(|i| (s.row_ptr[i + 1] - s.row_ptr[i], i))
        .collect();
    order.sort_by(|a, b| b.0.cmp(&a.0));

    for &(_, seed) in &order {
        if agg[seed] != usize::MAX { continue; }
        agg[seed] = next;
        // gather neighbors by degree as a simple heuristic
        let rs = s.row_ptr[seed]; let re = s.row_ptr[seed + 1];
        let mut neigh: Vec<(usize, usize)> = s.col_idx[rs..re]
            .iter().copied()
            .filter(|&j| agg[j] == usize::MAX)
            .map(|j| (s.row_ptr[j + 1] - s.row_ptr[j], j))
            .collect();
        neigh.sort_by(|a, b| b.0.cmp(&a.0));
        let mut count = 1usize;
        for &(_, j) in &neigh {
            if count >= max_sz { break; }
            agg[j] = next; count += 1;
        }
        next += 1;
    }

    // Any singleton remainers get their own aggregate
    for i in 0..n { if agg[i] == usize::MAX { agg[i] = { let id = next; next += 1; id }; } }
    agg
}

