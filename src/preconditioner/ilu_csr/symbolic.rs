use crate::algebra::scalar::{KrystScalar, S};

#[derive(Clone, Debug)]
pub struct RowWork {
    pub mark: Vec<i32>,  // n, initialized to -1
    pub idx: Vec<usize>, // positions used in this row
    pub val: Vec<S>,     // values aligned with idx
}

pub fn ensure_rowwork(w: &mut RowWork, n: usize) {
    if w.mark.len() < n {
        w.mark.resize(n, -1);
    }
    // Do not clear mark here; we track used entries in idx
    w.idx.clear();
    w.val.clear();
}

#[inline]
pub fn find_or_insert(w: &mut RowWork, j: usize) -> usize {
    let k = w.mark[j];
    if k >= 0 {
        k as usize
    } else {
        let pos = w.idx.len();
        w.idx.push(j);
        w.val.push(S::zero());
        w.mark[j] = pos as i32;
        pos
    }
}

pub fn clear_rowwork(w: &mut RowWork) {
    for &j in &w.idx {
        w.mark[j] = -1;
    }
    w.idx.clear();
    w.val.clear();
}
