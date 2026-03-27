use kryst::matrix::sparse::CsrMatrix;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use std::collections::BTreeSet;

/// 5-point Poisson on an n x n grid with Dirichlet BCs.
pub fn poisson2d_csr(n: usize) -> CsrMatrix<f64> {
    let nn = n * n;
    let mut row_ptr = Vec::with_capacity(nn + 1);
    let mut col_idx = Vec::with_capacity(5 * nn);
    let mut vals = Vec::with_capacity(5 * nn);
    row_ptr.push(0);
    for i in 0..n {
        for j in 0..n {
            let row = i * n + j;
            // Collect entries then sort by column to satisfy faer CSR invariants.
            let mut entries: [(usize, f64); 5] = [(usize::MAX, 0.0); 5];
            let mut len = 0usize;
            let mut diag = 0.0;
            if j > 0 {
                entries[len] = (row - 1, -1.0);
                len += 1;
                diag += 1.0;
            }
            if j + 1 < n {
                entries[len] = (row + 1, -1.0);
                len += 1;
                diag += 1.0;
            }
            if i > 0 {
                entries[len] = (row - n, -1.0);
                len += 1;
                diag += 1.0;
            }
            if i + 1 < n {
                entries[len] = (row + n, -1.0);
                len += 1;
                diag += 1.0;
            }
            // include diagonal
            entries[len] = (row, diag);
            len += 1;
            // Sort the first `len` elements by column index
            entries[..len].sort_unstable_by_key(|&(c, _)| c);
            for k in 0..len {
                col_idx.push(entries[k].0);
                vals.push(entries[k].1);
            }
            row_ptr.push(col_idx.len());
        }
    }
    CsrMatrix::from_csr(nn, nn, row_ptr, col_idx, vals)
}

/// Build a simple injection prolongator for 2x coarsening and its transpose restriction.
/// Fine grid n x n (n must be even). Each fine node maps to one coarse node.
pub fn poisson2d_prolong_restrict(n: usize) -> (CsrMatrix<f64>, CsrMatrix<f64>) {
    assert!(n % 2 == 0, "poisson2d_prolong_restrict: n must be even");
    let nf = n * n;
    let nc = (n / 2) * (n / 2);

    let mut p_row_ptr = Vec::with_capacity(nf + 1);
    let mut p_col_idx = Vec::with_capacity(nf);
    let mut p_vals = Vec::with_capacity(nf);
    p_row_ptr.push(0);
    for i in 0..n {
        for j in 0..n {
            // coarse index (i/2, j/2)
            let ci = i / 2;
            let cj = j / 2;
            let c = ci * (n / 2) + cj;
            p_col_idx.push(c);
            p_vals.push(1.0);
            p_row_ptr.push(p_row_ptr.last().unwrap() + 1);
        }
    }
    let p = CsrMatrix::from_csr(nf, nc, p_row_ptr, p_col_idx, p_vals);

    // R = P^T
    let mut counts = vec![0usize; nc + 1];
    for &cj in p.col_idx() {
        counts[cj + 1] += 1;
    }
    for i in 0..nc {
        counts[i + 1] += counts[i];
    }
    let mut r_row_ptr = counts.clone();
    let mut r_col_idx = vec![0usize; p.col_idx().len()];
    let mut r_vals = vec![0.0f64; p.values().len()];
    let mut next = r_row_ptr.clone();
    for i in 0..nf {
        for t in p.row_ptr()[i]..p.row_ptr()[i + 1] {
            let j = p.col_idx()[t];
            let dst = next[j];
            r_col_idx[dst] = i;
            r_vals[dst] = p.values()[t];
            next[j] += 1;
        }
    }
    let r = CsrMatrix::from_csr(nc, nf, r_row_ptr, r_col_idx, r_vals);
    (p, r)
}

/// Provide (A, P, R) triplet for RAP benchmarks on 2D Poisson.
pub fn rap_triplet_poisson2d(n: usize) -> (CsrMatrix<f64>, CsrMatrix<f64>, CsrMatrix<f64>) {
    let a = poisson2d_csr(n);
    let (p, r) = poisson2d_prolong_restrict(n);
    (a, p, r)
}

/// Random sparse matrix with average degree ~avg_deg and deterministic seed.
/// Not strictly power-law, but skewed degrees with duplicates removed and CSR sorted.
pub fn random_powerlaw_like(n: usize, avg_deg: usize, seed: u64) -> CsrMatrix<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::with_capacity(n * avg_deg);
    let mut vals = Vec::with_capacity(n * avg_deg);
    row_ptr.push(0);

    for i in 0..n {
        let base = rng.random_range((avg_deg / 2).max(1)..=(avg_deg * 3 / 2).max(2));
        let burst = if rng.random::<f64>() < 0.05 {
            rng.random_range(avg_deg..=4 * avg_deg)
        } else {
            0
        };
        let deg = (base + burst).min(n - 1);

        let mut set: BTreeSet<usize> = BTreeSet::new();
        // Ensure diagonal present to avoid singular row
        set.insert(i);
        while set.len() < deg {
            let j = rng.random_range(0..n);
            set.insert(j);
        }
        for &j in set.iter() {
            col_idx.push(j);
            // Values in [0.5, 1.5], negative off-diagonals with small probability
            let mut v = 0.5 + rng.random::<f64>();
            if j != i && rng.random::<f64>() < 0.2 {
                v = -v;
            }
            vals.push(v);
        }
        row_ptr.push(col_idx.len());
    }
    CsrMatrix::from_csr(n, n, row_ptr, col_idx, vals)
}

/// Block-diagonal with light overlap between neighboring blocks.
/// Returns (A, blocks) where blocks is a list of index sets.
pub fn blocky_csr(
    n_blocks: usize,
    block_size: usize,
    overlap: usize,
) -> (CsrMatrix<f64>, Vec<Vec<usize>>) {
    let n = n_blocks * block_size;
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::new();
    let mut vals = Vec::new();
    row_ptr.push(0);

    for b in 0..n_blocks {
        let start = b * block_size;
        let end = start + block_size;
        for i in start..end {
            // simple tri-diagonal inside block
            if i > start {
                col_idx.push(i - 1);
                vals.push(-1.0);
            }
            col_idx.push(i);
            vals.push(2.0);
            if i + 1 < end {
                col_idx.push(i + 1);
                vals.push(-1.0);
            }

            // overlap edges to next block
            if overlap > 0 && b + 1 < n_blocks {
                let next_start = (b + 1) * block_size;
                for k in 0..overlap.min(block_size) {
                    let j = next_start + k;
                    col_idx.push(j);
                    vals.push(-0.1);
                }
            }
            row_ptr.push(col_idx.len());
        }
    }

    let a = CsrMatrix::from_csr(n, n, row_ptr, col_idx, vals);
    // Blocks index sets
    let mut blocks = Vec::with_capacity(n_blocks);
    for b in 0..n_blocks {
        let start = b * block_size;
        blocks.push((start..start + block_size).collect());
    }
    (a, blocks)
}
