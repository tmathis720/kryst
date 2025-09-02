use num_traits::Zero;

#[derive(Clone, Debug)]
pub struct CsrRowBuilder<T> {
    pub cols: Vec<usize>,
    pub vals: Vec<T>,
}

impl<T> Default for CsrRowBuilder<T> {
    fn default() -> Self {
        Self { cols: Vec::new(), vals: Vec::new() }
    }
}

#[derive(Clone, Debug)]
pub struct CsrBuilder<T> {
    nrows: usize,
    pub rows: Vec<CsrRowBuilder<T>>,
}

impl<T> CsrBuilder<T> {
    pub fn new(n: usize) -> Self {
        Self { nrows: n, rows: (0..n).map(|_| CsrRowBuilder::<T>::default()).collect() }
    }

    #[inline]
    pub fn push(&mut self, i: usize, j: usize, v: T) {
        self.rows[i].cols.push(j);
        self.rows[i].vals.push(v);
    }

    pub fn row(&self, i: usize) -> (&[usize], &[T]) {
        let r = &self.rows[i];
        (&r.cols, &r.vals)
    }
}

impl<T> CsrBuilder<T>
where
    T: Zero + Copy + std::ops::AddAssign + PartialOrd,
{
    pub fn finalize_sorted_unique(self, reproducible: bool) -> (Vec<usize>, Vec<usize>, Vec<T>) {
        let mut row_ptr = Vec::with_capacity(self.nrows + 1);
        let mut col_idx: Vec<usize> = Vec::new();
        let mut vals: Vec<T> = Vec::new();
        row_ptr.push(0);
        for mut r in self.rows.into_iter() {
            let mut pairs: Vec<(usize, T)> = r.cols.drain(..).zip(r.vals.drain(..)).collect();
            if reproducible {
                pairs.sort_by(|a, b| a.0.cmp(&b.0));
            } else {
                pairs.sort_unstable_by(|a, b| a.0.cmp(&b.0));
            }
            let mut last_col: Option<usize> = None;
            let mut last_val: T = T::zero();
            for (c, v) in pairs {
                if let Some(lc) = last_col {
                    if lc == c {
                        last_val += v;
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
