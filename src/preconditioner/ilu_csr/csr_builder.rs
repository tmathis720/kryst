#[allow(unused_imports)]
use crate::algebra::prelude::*;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

#[derive(Clone, Debug)]
pub struct CsrRowBuilder {
    pub cols: Vec<usize>,
    pub vals: Vec<S>,
}

impl Default for CsrRowBuilder {
    fn default() -> Self {
        Self {
            cols: Vec::new(),
            vals: Vec::new(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct CsrBuilder {
    nrows: usize,
    pub rows: Vec<CsrRowBuilder>,
}

impl CsrBuilder {
    pub fn new(n: usize) -> Self {
        Self {
            nrows: n,
            rows: (0..n).map(|_| CsrRowBuilder::default()).collect(),
        }
    }

    #[inline]
    pub fn push(&mut self, i: usize, j: usize, v: S) {
        self.rows[i].cols.push(j);
        self.rows[i].vals.push(v);
    }

    pub fn row(&self, i: usize) -> (&[usize], &[S]) {
        let r = &self.rows[i];
        (&r.cols, &r.vals)
    }
}

impl CsrBuilder {
    pub fn finalize_sorted_unique(self, reproducible: bool) -> (Vec<usize>, Vec<usize>, Vec<S>) {
        let mut row_ptr = Vec::with_capacity(self.nrows + 1);
        let mut col_idx: Vec<usize> = Vec::new();
        let mut vals: Vec<S> = Vec::new();
        row_ptr.push(0);
        for mut r in self.rows.into_iter() {
            let mut pairs: Vec<(usize, S)> = r.cols.drain(..).zip(r.vals.drain(..)).collect();
            if reproducible {
                pairs.sort_by(|a, b| a.0.cmp(&b.0));
            } else {
                pairs.sort_unstable_by(|a, b| a.0.cmp(&b.0));
            }
            let mut last_col: Option<usize> = None;
            let mut last_val: S = S::zero();
            for (c, v) in pairs {
                if let Some(lc) = last_col {
                    if lc == c {
                        last_val = last_val + v;
                        continue;
                    } else {
                        col_idx.push(lc);
                        vals.push(last_val);
                    }
                }
                last_col = Some(c);
                last_val = v;
            }
            if let Some(lc) = last_col {
                col_idx.push(lc);
                vals.push(last_val);
            }
            row_ptr.push(col_idx.len());
        }
        (row_ptr, col_idx, vals)
    }
}

/// Metadata associated with candidate fill entries when building ILU rows.
pub enum Meta<R> {
    /// No extra metadata (ILU0)
    None,
    /// Level information used by ILU(k)
    Level(u32),
    /// Magnitude for ILUT
    Magn(R),
}

/// Trait for building a single sparse row under a given ILU policy.
pub trait RowBuilder<S: KrystScalar> {
    /// Reset the builder for a new row without allocating.
    fn clear(&mut self);
    /// Update an existing column value. The column is guaranteed to be present
    /// in the row structure for ILU(0) style builders.
    fn push_existing(&mut self, col: usize, val: S);
    /// Attempt to insert a candidate fill with associated metadata.
    fn try_insert(&mut self, col: usize, meta: Meta<S::Real>, val: S);
    /// Finalize the row by appending its columns and values to the provided
    /// arrays. The arrays are expected to be empty and are not cleared by the
    /// builder.
    fn finalize_into(&mut self, row_cols: &mut Vec<usize>, row_vals: &mut Vec<S>);
}

/// ILU(0) row builder that only updates existing structure slots.
pub struct Ilu0Row<'a, S> {
    cols: &'a [usize],
    vals: &'a mut [S],
}

impl<'a, S: KrystScalar> Ilu0Row<'a, S> {
    pub fn new(cols: &'a [usize], vals: &'a mut [S]) -> Self {
        Self { cols, vals }
    }
}

impl<'a, S: KrystScalar> RowBuilder<S> for Ilu0Row<'a, S> {
    fn clear(&mut self) {}

    fn push_existing(&mut self, col: usize, val: S) {
        if let Ok(pos) = self.cols.binary_search(&col) {
            self.vals[pos] = val;
        }
    }

    fn try_insert(&mut self, _col: usize, _meta: Meta<S::Real>, _val: S) {}

    fn finalize_into(&mut self, _row_cols: &mut Vec<usize>, _row_vals: &mut Vec<S>) {}
}

/// ILU(k) row builder using fixed-size arrays for candidate fills.
pub struct IlukRow<S: KrystScalar, const CAP: usize> {
    pub used: usize,
    pub max_level: u32,
    pub cand_col: [usize; CAP],
    pub cand_lev: [u32; CAP],
    pub cand_val: [S; CAP],
}

impl<S: KrystScalar, const CAP: usize> IlukRow<S, CAP> {
    pub fn new(max_level: u32) -> Self {
        Self {
            used: 0,
            max_level,
            cand_col: [0; CAP],
            cand_lev: [0; CAP],
            cand_val: [S::zero(); CAP],
        }
    }

    #[inline]
    fn upsert(&mut self, col: usize, lev: u32, val: S) {
        for i in 0..self.used {
            if self.cand_col[i] == col {
                self.cand_val[i] = self.cand_val[i] + val;
                if lev < self.cand_lev[i] {
                    self.cand_lev[i] = lev;
                }
                return;
            }
        }
        if lev > self.max_level {
            return;
        }
        if self.used < CAP {
            self.cand_col[self.used] = col;
            self.cand_lev[self.used] = lev;
            self.cand_val[self.used] = val;
            self.used += 1;
            return;
        }
        // Replace worst entry if the new one is better
        let mut worst = 0usize;
        for i in 1..CAP {
            let worse = self.cand_lev[i] > self.cand_lev[worst]
                || (self.cand_lev[i] == self.cand_lev[worst]
                    && self.cand_val[i].abs() < self.cand_val[worst].abs());
            if worse {
                worst = i;
            }
        }
        let better = lev < self.cand_lev[worst]
            || (lev == self.cand_lev[worst] && val.abs() > self.cand_val[worst].abs());
        if better {
            self.cand_col[worst] = col;
            self.cand_lev[worst] = lev;
            self.cand_val[worst] = val;
        }
    }
}

impl<S: KrystScalar, const CAP: usize> RowBuilder<S> for IlukRow<S, CAP> {
    fn clear(&mut self) {
        self.used = 0;
    }

    fn push_existing(&mut self, col: usize, val: S) {
        self.upsert(col, 0, val);
    }

    fn try_insert(&mut self, col: usize, meta: Meta<S::Real>, val: S) {
        let lev = match meta {
            Meta::Level(l) => l,
            _ => 0,
        };
        self.upsert(col, lev, val);
    }

    fn finalize_into(&mut self, row_cols: &mut Vec<usize>, row_vals: &mut Vec<S>) {
        // simple insertion sort
        for i in 1..self.used {
            let mut j = i;
            while j > 0 && self.cand_col[j - 1] > self.cand_col[j] {
                self.cand_col.swap(j - 1, j);
                self.cand_lev.swap(j - 1, j);
                self.cand_val.swap(j - 1, j);
                j -= 1;
            }
        }
        row_cols.extend_from_slice(&self.cand_col[..self.used]);
        row_vals.extend_from_slice(&self.cand_val[..self.used]);
        self.used = 0;
    }
}

#[derive(Clone)]
struct HeapElem<S: KrystScalar> {
    mag: S::Real,
    col: usize,
    val: S,
}

impl<S: KrystScalar> PartialEq for HeapElem<S> {
    fn eq(&self, other: &Self) -> bool {
        self.mag == other.mag
    }
}

impl<S: KrystScalar> Eq for HeapElem<S> {}

impl<S: KrystScalar> PartialOrd for HeapElem<S> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.mag.partial_cmp(&other.mag)
    }
}

impl<S: KrystScalar> Ord for HeapElem<S> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

/// ILUT row builder using a magnitude-based capped heap.
pub struct IlutRow<S: KrystScalar> {
    p: usize,
    drop_tol: S::Real,
    heap: BinaryHeap<Reverse<HeapElem<S>>>,
}

impl<S: KrystScalar> IlutRow<S> {
    pub fn new(p: usize, drop_tol: S::Real) -> Self {
        Self {
            p,
            drop_tol,
            heap: BinaryHeap::new(),
        }
    }
}

impl<S: KrystScalar> RowBuilder<S> for IlutRow<S> {
    fn clear(&mut self) {
        self.heap.clear();
    }

    fn push_existing(&mut self, col: usize, val: S) {
        self.try_insert(col, Meta::None, val);
    }

    fn try_insert(&mut self, col: usize, _meta: Meta<S::Real>, val: S) {
        let mag = val.abs();
        if mag < self.drop_tol {
            return;
        }
        let elem = Reverse(HeapElem { mag, col, val });
        if self.heap.len() < self.p {
            self.heap.push(elem);
        } else if let Some(peek) = self.heap.peek()
            && mag > peek.0.mag
        {
            let _ = self.heap.pop();
            self.heap.push(elem);
        }
    }

    fn finalize_into(&mut self, row_cols: &mut Vec<usize>, row_vals: &mut Vec<S>) {
        let mut elems: Vec<_> = self.heap.drain().map(|r| r.0).collect();
        elems.sort_by_key(|e| e.col);
        for e in elems {
            row_cols.push(e.col);
            row_vals.push(e.val);
        }
    }
}
