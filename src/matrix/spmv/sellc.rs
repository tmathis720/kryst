//! SELL-C-σ storage conversion and SIMD-friendly kernels.
//!
//! The SELL-C-σ format groups rows into fixed-height slices (`C`) while sorting
//! within a sliding window of size `σ` to reduce row length variance. This
//! produces near-contiguous access patterns that map well to portable SIMD.

use wide::{f64x2, f64x4};

/// Dense row-blocked storage derived from a CSR matrix.
#[derive(Clone, Debug)]
pub struct SellCStorage {
    /// Slice height.
    pub c: usize,
    /// Local sorting window.
    pub sigma: usize,
    /// Prefix sum over the number of column groups per slice.
    pub slice_ptr: Vec<usize>,
    /// Interleaved column indices laid out column-major within each slice.
    pub col_idx: Vec<usize>,
    /// Interleaved numerical values matching [`col_idx`].
    pub vals: Vec<f64>,
    /// Number of rows represented by the structure.
    pub rows: usize,
    /// Number of columns.
    pub cols: usize,
    /// Mapping from slice-local row to original row index (padded with
    /// `usize::MAX` for vacant slots).
    pub row_perm: Vec<usize>,
}

impl SellCStorage {
    /// Returns the number of slices stored in this structure.
    #[inline]
    pub fn slices(&self) -> usize {
        self.slice_ptr.len().saturating_sub(1)
    }
}

/// Converts a CSR matrix into SELL-C-σ storage.
pub fn csr_to_sellc(
    m: usize,
    n: usize,
    row_ptr: &[usize],
    col_idx: &[usize],
    vals: &[f64],
    c: usize,
    sigma: usize,
) -> SellCStorage {
    assert_eq!(row_ptr.len(), m + 1);
    assert_eq!(col_idx.len(), vals.len());
    assert!(c > 0, "slice height must be positive");
    assert!(sigma >= c, "sigma must be at least the slice height");

    let nnz = col_idx.len();
    let mut row_lengths = vec![0usize; m];
    for i in 0..m {
        row_lengths[i] = row_ptr[i + 1] - row_ptr[i];
    }

    // Build the permutation according to the σ-window heuristic.
    let mut permuted_rows: Vec<usize> = (0..m).collect();
    let mut start = 0;
    while start < m {
        let end = (start + sigma).min(m);
        permuted_rows[start..end]
            .sort_by(|&lhs, &rhs| row_lengths[rhs].cmp(&row_lengths[lhs]).then(lhs.cmp(&rhs)));
        start = end;
    }

    let nslices = (m + c - 1) / c;
    let mut slice_ptr = Vec::with_capacity(nslices + 1);
    slice_ptr.push(0);
    let mut col_storage = Vec::with_capacity(nnz);
    let mut val_storage = Vec::with_capacity(nnz);
    let mut row_perm = Vec::with_capacity(nslices * c);

    let mut offset = 0usize;
    for _ in 0..nslices {
        let rows_in_slice = permuted_rows.len().saturating_sub(offset).min(c);
        if rows_in_slice == 0 {
            row_perm.extend(std::iter::repeat(usize::MAX).take(c));
            slice_ptr.push(*slice_ptr.last().unwrap());
            continue;
        }
        let slice_rows = &permuted_rows[offset..offset + rows_in_slice];
        for &r in slice_rows {
            row_perm.push(r);
        }
        // Pad the permutation with sentinels so row indices align with `c`.
        if rows_in_slice < c {
            row_perm.extend(std::iter::repeat(usize::MAX).take(c - rows_in_slice));
        }

        let mut max_len = 0usize;
        for &r in slice_rows {
            max_len = max_len.max(row_lengths[r]);
        }

        for t in 0..max_len {
            for &r in slice_rows {
                let rs = row_ptr[r];
                let len = row_lengths[r];
                if t < len {
                    col_storage.push(col_idx[rs + t]);
                    val_storage.push(vals[rs + t]);
                } else {
                    col_storage.push(0);
                    val_storage.push(0.0);
                }
            }
            if rows_in_slice < c {
                col_storage.extend(std::iter::repeat(0).take(c - rows_in_slice));
                val_storage.extend(std::iter::repeat(0.0).take(c - rows_in_slice));
            }
        }

        slice_ptr.push(slice_ptr.last().copied().unwrap() + max_len);
        offset += rows_in_slice;
    }

    SellCStorage {
        c,
        sigma,
        slice_ptr,
        col_idx: col_storage,
        vals: val_storage,
        rows: m,
        cols: n,
        row_perm,
    }
}

#[inline]
fn load2(vals: &[f64]) -> f64x2 {
    f64x2::new([vals[0], vals[1]])
}

#[inline]
fn load4(vals: &[f64]) -> f64x4 {
    f64x4::new([vals[0], vals[1], vals[2], vals[3]])
}

#[inline]
unsafe fn gather2(x: &[f64], idx: &[usize]) -> f64x2 {
    let x0 = unsafe { *x.get_unchecked(idx[0]) };
    let x1 = unsafe { *x.get_unchecked(idx[1]) };
    f64x2::new([x0, x1])
}

#[inline]
unsafe fn gather4(x: &[f64], idx: &[usize]) -> f64x4 {
    let x0 = unsafe { *x.get_unchecked(idx[0]) };
    let x1 = unsafe { *x.get_unchecked(idx[1]) };
    let x2 = unsafe { *x.get_unchecked(idx[2]) };
    let x3 = unsafe { *x.get_unchecked(idx[3]) };
    f64x4::new([x0, x1, x2, x3])
}

fn spmv_scaled_sellc_lane2(
    storage: &SellCStorage,
    alpha: f64,
    x: &[f64],
    beta: f64,
    y: &mut [f64],
) {
    if beta == 0.0 {
        y[..storage.rows].fill(0.0);
    } else if beta != 1.0 {
        for yi in &mut y[..storage.rows] {
            *yi *= beta;
        }
    }

    let mut base = 0usize;
    for si in 0..storage.slices() {
        let r_begin = si * storage.c;
        let rows_remaining = storage.rows.saturating_sub(r_begin);
        let rows_in_slice = rows_remaining.min(storage.c);
        if rows_in_slice == 0 {
            break;
        }
        let groups = storage.slice_ptr[si + 1] - storage.slice_ptr[si];
        if groups == 0 {
            base += groups * storage.c;
            continue;
        }
        let mut acc = vec![0.0f64; rows_in_slice];

        for g in 0..groups {
            let offset = base + g * storage.c;
            let cols = &storage.col_idx[offset..offset + storage.c];
            let vals = &storage.vals[offset..offset + storage.c];

            let mut r = 0usize;
            while r + 1 < rows_in_slice {
                let idx = [cols[r], cols[r + 1]];
                let val_vec = load2(&vals[r..r + 2]);
                let x_vec = unsafe { gather2(x, &idx) };
                let prod = (val_vec * x_vec).to_array();
                for lane in 0..2 {
                    acc[r + lane] += prod[lane];
                }
                r += 2;
            }
            while r < rows_in_slice {
                acc[r] += vals[r] * x[cols[r]];
                r += 1;
            }
        }

        for local in 0..rows_in_slice {
            let row = storage.row_perm[r_begin + local];
            if row != usize::MAX {
                y[row] += alpha * acc[local];
            }
        }

        base += groups * storage.c;
    }
}

fn spmv_scaled_sellc_lane4(
    storage: &SellCStorage,
    alpha: f64,
    x: &[f64],
    beta: f64,
    y: &mut [f64],
) {
    if beta == 0.0 {
        y[..storage.rows].fill(0.0);
    } else if beta != 1.0 {
        for yi in &mut y[..storage.rows] {
            *yi *= beta;
        }
    }

    let mut base = 0usize;
    for si in 0..storage.slices() {
        let r_begin = si * storage.c;
        let rows_remaining = storage.rows.saturating_sub(r_begin);
        let rows_in_slice = rows_remaining.min(storage.c);
        if rows_in_slice == 0 {
            break;
        }
        let groups = storage.slice_ptr[si + 1] - storage.slice_ptr[si];
        if groups == 0 {
            base += groups * storage.c;
            continue;
        }
        let mut acc = vec![0.0f64; rows_in_slice];

        for g in 0..groups {
            let offset = base + g * storage.c;
            let cols = &storage.col_idx[offset..offset + storage.c];
            let vals = &storage.vals[offset..offset + storage.c];

            let mut r = 0usize;
            while r + 3 < rows_in_slice {
                let idx = [cols[r], cols[r + 1], cols[r + 2], cols[r + 3]];
                let val_vec = load4(&vals[r..r + 4]);
                let x_vec = unsafe { gather4(x, &idx) };
                let prod = (val_vec * x_vec).to_array();
                for lane in 0..4 {
                    acc[r + lane] += prod[lane];
                }
                r += 4;
            }
            while r < rows_in_slice {
                acc[r] += vals[r] * x[cols[r]];
                r += 1;
            }
        }

        for local in 0..rows_in_slice {
            let row = storage.row_perm[r_begin + local];
            if row != usize::MAX {
                y[row] += alpha * acc[local];
            }
        }

        base += groups * storage.c;
    }
}

/// SIMD-friendly SELL-C kernel that computes `y = alpha * A * x + beta * y`.
pub fn spmv_scaled_sellc(
    storage: &SellCStorage,
    alpha: f64,
    x: &[f64],
    beta: f64,
    y: &mut [f64],
    lanes: usize,
) {
    match lanes {
        4 => spmv_scaled_sellc_lane4(storage, alpha, x, beta, y),
        _ => spmv_scaled_sellc_lane2(storage, alpha, x, beta, y),
    }
}
