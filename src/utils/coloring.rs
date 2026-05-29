//! Distance-2 graph coloring utility for block-Jacobi and multicolored preconditioners.
//! See Saad §10.7, §12.4 for background.

use std::collections::HashSet;

use crate::matrix::sparse::CsrMatrix;

/// Extract adjacency list from a matrix pattern: adj\[i\] = { j | A\[i,j\] ≠ 0 or A\[j,i\] ≠ 0 }
pub fn extract_adjacency<F>(n: usize, is_nz: F) -> Vec<Vec<usize>>
where
    F: Fn(usize, usize) -> bool,
{
    let mut adj = vec![Vec::new(); n];
    for (i, row) in adj.iter_mut().enumerate() {
        for j in 0..n {
            if i != j && (is_nz(i, j) || is_nz(j, i)) {
                row.push(j);
            }
        }
    }
    adj
}

/// Build distance-2 neighbor sets: dist2\[i\] = adj\[i\] ∪ (⋃_{j∈adj\[i\]} adj\[j\])
pub fn distance2_neighbors(adj: &[Vec<usize>]) -> Vec<HashSet<usize>> {
    let n = adj.len();
    let mut dist2 = vec![HashSet::new(); n];
    for (i, neighbors) in adj.iter().enumerate() {
        for &j in neighbors {
            dist2[i].insert(j);
            for &k in &adj[j] {
                dist2[i].insert(k);
            }
        }
        dist2[i].insert(i); // include self
    }
    dist2
}

/// Greedy distance-2 coloring. Returns colors\[i\] = color assigned to node i.
pub fn greedy_distance2_coloring(dist2: &[HashSet<usize>]) -> Vec<usize> {
    let n = dist2.len();
    let mut color_of = vec![None; n];
    for (i, neighbors) in dist2.iter().enumerate() {
        let mut banned = HashSet::new();
        for &k in neighbors {
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

/// Build blocks from a color assignment: blocks\[c\] = indices with color c
pub fn build_blocks_from_colors(colors: &[usize]) -> Vec<Vec<usize>> {
    let num_colors = colors.iter().copied().max().map(|c| c + 1).unwrap_or(0);
    let mut blocks = vec![Vec::new(); num_colors];
    for (i, &c) in colors.iter().enumerate() {
        blocks[c].push(i);
    }
    blocks
}

/// Compute a distance-2 coloring for a CSR matrix graph (based on sparsity).
///
/// The resulting colors are suitable for parallel Gauss–Seidel-style sweeps
/// where nodes in the same color have no direct adjacency.
pub fn csr_distance2_coloring<T>(a: &CsrMatrix<T>) -> Vec<usize> {
    let n = a.nrows().min(a.ncols());
    let mut adj_sets = vec![HashSet::new(); n];
    for i in 0..n {
        let (cols, _) = a.row(i);
        for &j in cols {
            if j < n && j != i {
                adj_sets[i].insert(j);
                adj_sets[j].insert(i);
            }
        }
    }
    let adj: Vec<Vec<usize>> = adj_sets
        .into_iter()
        .map(|set| set.into_iter().collect())
        .collect();
    let dist2 = distance2_neighbors(&adj);
    greedy_distance2_coloring(&dist2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_adjacency_empty() {
        let n = 3;
        let adj = extract_adjacency(n, |_, _| false);
        assert_eq!(adj.len(), 3);
        assert!(adj[0].is_empty());
        assert!(adj[1].is_empty());
        assert!(adj[2].is_empty());
    }

    #[test]
    fn test_extract_adjacency_simple() {
        // Matrix with pattern:
        // [1 1 0]
        // [1 1 1]
        // [0 1 1]
        let n = 3;
        let is_nz = |i: usize, j: usize| -> bool {
            matches!(
                (i, j),
                (0, 0) | (0, 1) | (1, 0) | (1, 1) | (1, 2) | (2, 1) | (2, 2)
            )
        };

        let adj = extract_adjacency(n, is_nz);
        assert_eq!(adj.len(), 3);

        // Node 0 is connected to node 1
        assert!(adj[0].contains(&1));
        assert_eq!(adj[0].len(), 1);

        // Node 1 is connected to nodes 0 and 2
        assert!(adj[1].contains(&0));
        assert!(adj[1].contains(&2));
        assert_eq!(adj[1].len(), 2);

        // Node 2 is connected to node 1
        assert!(adj[2].contains(&1));
        assert_eq!(adj[2].len(), 1);
    }

    #[test]
    fn test_distance2_neighbors_simple() {
        // Simple chain: 0-1-2
        let adj = vec![
            vec![1],    // 0 connected to 1
            vec![0, 2], // 1 connected to 0, 2
            vec![1],    // 2 connected to 1
        ];

        let dist2 = distance2_neighbors(&adj);

        // Node 0: neighbors at distance ≤ 2 are {0, 1, 2}
        assert!(dist2[0].contains(&0)); // self
        assert!(dist2[0].contains(&1)); // direct neighbor
        assert!(dist2[0].contains(&2)); // distance-2 neighbor
        assert_eq!(dist2[0].len(), 3);

        // Node 1: neighbors at distance ≤ 2 are {0, 1, 2}
        assert!(dist2[1].contains(&0)); // direct neighbor
        assert!(dist2[1].contains(&1)); // self
        assert!(dist2[1].contains(&2)); // direct neighbor
        assert_eq!(dist2[1].len(), 3);

        // Node 2: neighbors at distance ≤ 2 are {0, 1, 2}
        assert!(dist2[2].contains(&0)); // distance-2 neighbor
        assert!(dist2[2].contains(&1)); // direct neighbor
        assert!(dist2[2].contains(&2)); // self
        assert_eq!(dist2[2].len(), 3);
    }

    #[test]
    fn test_distance2_neighbors_isolated() {
        // Two isolated nodes: 0  1
        let adj = vec![
            vec![], // 0 has no neighbors
            vec![], // 1 has no neighbors
        ];

        let dist2 = distance2_neighbors(&adj);

        // Node 0: only itself
        assert!(dist2[0].contains(&0));
        assert_eq!(dist2[0].len(), 1);

        // Node 1: only itself
        assert!(dist2[1].contains(&1));
        assert_eq!(dist2[1].len(), 1);
    }

    #[test]
    fn test_greedy_distance2_coloring_isolated() {
        // Two isolated nodes can have the same color
        let dist2 = vec![
            [0].iter().cloned().collect(), // Node 0: only itself
            [1].iter().cloned().collect(), // Node 1: only itself
        ];

        let colors = greedy_distance2_coloring(&dist2);
        assert_eq!(colors.len(), 2);
        assert_eq!(colors[0], 0);
        assert_eq!(colors[1], 0); // Same color since they don't conflict
    }

    #[test]
    fn test_greedy_distance2_coloring_chain() {
        // Chain: 0-1-2, all nodes need different colors
        let dist2 = vec![
            [0, 1, 2].iter().cloned().collect(), // Node 0
            [0, 1, 2].iter().cloned().collect(), // Node 1
            [0, 1, 2].iter().cloned().collect(), // Node 2
        ];

        let colors = greedy_distance2_coloring(&dist2);
        assert_eq!(colors.len(), 3);

        // All colors should be different
        assert_ne!(colors[0], colors[1]);
        assert_ne!(colors[1], colors[2]);
        assert_ne!(colors[0], colors[2]);
    }

    #[test]
    fn test_color_graph_integration() {
        // Test the full pipeline with a simple 2x2 diagonal matrix
        let n = 2;
        let is_nz = |i: usize, j: usize| i == j; // Only diagonal elements

        let colors = color_graph(n, is_nz);
        assert_eq!(colors.len(), 2);

        // Diagonal nodes don't conflict, so they can have the same color
        assert_eq!(colors[0], colors[1]);
    }

    #[test]
    fn test_color_graph_tridiagonal() {
        // Tridiagonal matrix pattern
        let n = 4;
        let is_nz = |i: usize, j: usize| i == j || (i + 1 == j) || (j + 1 == i);

        let colors = color_graph(n, is_nz);
        assert_eq!(colors.len(), 4);

        // Adjacent nodes should have different colors
        for i in 0..n - 1 {
            assert_ne!(colors[i], colors[i + 1]);
        }
    }

    #[test]
    fn test_build_blocks_from_colors() {
        let colors = vec![0, 1, 0, 2, 1];
        let blocks = build_blocks_from_colors(&colors);

        assert_eq!(blocks.len(), 3); // 3 colors: 0, 1, 2

        // Color 0: indices 0, 2
        assert_eq!(blocks[0], vec![0, 2]);

        // Color 1: indices 1, 4
        assert_eq!(blocks[1], vec![1, 4]);

        // Color 2: index 3
        assert_eq!(blocks[2], vec![3]);
    }

    #[test]
    fn test_build_blocks_from_colors_empty() {
        let colors: Vec<usize> = vec![];
        let blocks = build_blocks_from_colors(&colors);
        assert!(blocks.is_empty());
    }

    #[test]
    fn test_build_blocks_single_color() {
        let colors = vec![0, 0, 0];
        let blocks = build_blocks_from_colors(&colors);

        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0], vec![0, 1, 2]);
    }

    #[test]
    fn test_extract_adjacency_symmetric() {
        // Test that adjacency is symmetric for symmetric matrices
        let n = 3;
        let is_nz = |i: usize, j: usize| (i + j) % 2 == 0; // Some pattern

        let adj = extract_adjacency(n, is_nz);

        // Verify symmetry: if j ∈ adj[i], then i ∈ adj[j]
        for i in 0..n {
            for &j in &adj[i] {
                assert!(
                    adj[j].contains(&i),
                    "Adjacency not symmetric: {} -> {} but not {} -> {}",
                    i,
                    j,
                    j,
                    i
                );
            }
        }
    }

    #[test]
    fn test_coloring_properties() {
        // Test coloring on a small complete graph
        let n = 3;
        let is_nz = |i: usize, j: usize| i != j; // Complete graph (all off-diagonal elements)

        let colors = color_graph(n, is_nz);
        assert_eq!(colors.len(), 3);

        // In a complete graph, all nodes need different colors
        assert_ne!(colors[0], colors[1]);
        assert_ne!(colors[1], colors[2]);
        assert_ne!(colors[0], colors[2]);
    }

    #[test]
    fn test_distance2_self_inclusion() {
        // Verify that distance-2 sets always include the node itself
        let adj = vec![vec![1, 2], vec![0], vec![0]];

        let dist2 = distance2_neighbors(&adj);

        for i in 0..adj.len() {
            assert!(
                dist2[i].contains(&i),
                "Node {} not in its own distance-2 set",
                i
            );
        }
    }
}
