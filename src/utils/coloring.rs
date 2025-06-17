//! Distance-2 graph coloring utility for block-Jacobi and multicolored preconditioners.
//! See Saad §10.7, §12.4 for background.

use std::collections::HashSet;

/// Extract adjacency list from a matrix pattern: adj[i] = { j | A[i,j] ≠ 0 or A[j,i] ≠ 0 }
pub fn extract_adjacency<F>(n: usize, is_nz: F) -> Vec<Vec<usize>>
where
    F: Fn(usize, usize) -> bool,
{
    let mut adj = vec![Vec::new(); n];
    for i in 0..n {
        for j in 0..n {
            if i != j && (is_nz(i, j) || is_nz(j, i)) {
                adj[i].push(j);
            }
        }
    }
    adj
}

/// Build distance-2 neighbor sets: dist2[i] = adj[i] ∪ (⋃_{j∈adj[i]} adj[j])
pub fn distance2_neighbors(adj: &[Vec<usize>]) -> Vec<HashSet<usize>> {
    let n = adj.len();
    let mut dist2 = vec![HashSet::new(); n];
    for i in 0..n {
        for &j in &adj[i] {
            dist2[i].insert(j);
            for &k in &adj[j] {
                dist2[i].insert(k);
            }
        }
        dist2[i].insert(i); // include self
    }
    dist2
}

/// Greedy distance-2 coloring. Returns colors[i] = color assigned to node i.
pub fn greedy_distance2_coloring(dist2: &[HashSet<usize>]) -> Vec<usize> {
    let n = dist2.len();
    let mut color_of = vec![None; n];
    for i in 0..n {
        let mut banned = HashSet::new();
        for &k in &dist2[i] {
            if let Some(c) = color_of[k] {
                banned.insert(c);
            }
        }
        let c = (0..).find(|c| !banned.contains(c)).unwrap();
        color_of[i] = Some(c);
    }
    color_of.into_iter().map(|c| c.unwrap()).collect()
}

/// Convenience: color a matrix given a sparsity predicate.
/// Returns a color assignment for each node.
pub fn color_graph<F>(n: usize, is_nz: F) -> Vec<usize>
where
    F: Fn(usize, usize) -> bool,
{
    let adj = extract_adjacency(n, &is_nz);
    let dist2 = distance2_neighbors(&adj);
    greedy_distance2_coloring(&dist2)
}

/// Build blocks from a color assignment: blocks[c] = indices with color c
pub fn build_blocks_from_colors(colors: &[usize]) -> Vec<Vec<usize>> {
    let num_colors = colors.iter().copied().max().map(|c| c + 1).unwrap_or(0);
    let mut blocks = vec![Vec::new(); num_colors];
    for (i, &c) in colors.iter().enumerate() {
        blocks[c].push(i);
    }
    blocks
}
