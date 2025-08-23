use crate::error::KError;
use crate::matrix::op::LinOp;
use crate::preconditioner::{PcSide, Preconditioner};
use std::sync::Mutex;

/// A simple compositional preconditioner:
/// y = P_k( ... P_2(P_1(x)) ... ) for all PcSide variants.
/// This models M^{-1} ≈ P_k ∘ ... ∘ P_1.
///
pub struct PcChain {
    stages: Vec<Box<dyn Preconditioner>>,
    scratch: Mutex<ChainScratch>,
}

#[derive(Default)]
struct ChainScratch {
    buf1: Vec<f64>,
    buf2: Vec<f64>,
}

impl PcChain {
    pub fn new(stages: Vec<Box<dyn Preconditioner>>) -> Self {
        Self { stages, scratch: Mutex::new(ChainScratch::default()) }
    }

    #[inline]
    fn ensure_bufs(s: &mut ChainScratch, n: usize) {
        if s.buf1.len() != n { s.buf1.resize(n, 0.0); }
        if s.buf2.len() != n { s.buf2.resize(n, 0.0); }
    }

    pub fn len(&self) -> usize { self.stages.len() }
    pub fn is_empty(&self) -> bool { self.stages.is_empty() }
}

impl Preconditioner for PcChain {
    fn setup(&mut self, a: &dyn LinOp<S = f64>) -> Result<(), KError> {
        for st in self.stages.iter_mut() {
            st.setup(a)?;
        }
        Ok(())
    }

    fn apply(&self, side: PcSide, x: &[f64], y: &mut [f64]) -> Result<(), KError> {
        if self.stages.is_empty() {
            y.copy_from_slice(x);
            return Ok(());
        }
        if self.stages.len() == 1 {
            return self.stages[0].apply(side, x, y);
        }

        let n = x.len();
        let mut s = self.scratch.lock().unwrap();
        Self::ensure_bufs(&mut s, n);
        let ChainScratch { buf1, buf2 } = &mut *s;

        self.stages[0].apply(side, x, buf1)?;
        let mut in_is_buf1 = true;
        let m = self.stages.len();
        for st in self.stages.iter().skip(1).take(m - 2) {
            if in_is_buf1 {
                st.apply(side, &*buf1, buf2)?;
            } else {
                st.apply(side, &*buf2, buf1)?;
            }
            in_is_buf1 = !in_is_buf1;
        }
        let last = self.stages.last().unwrap();
        if in_is_buf1 {
            last.apply(side, &*buf1, y)?;
        } else {
            last.apply(side, &*buf2, y)?;
        }
        Ok(())
    }

    fn apply_mut(&mut self, side: PcSide, x: &[f64], y: &mut [f64]) -> Result<(), KError> {
        if self.stages.is_empty() {
            y.copy_from_slice(x);
            return Ok(());
        }
        if self.stages.len() == 1 {
            return self.stages[0].apply_mut(side, x, y);
        }

        let n = x.len();
        let mut s = self.scratch.lock().unwrap();
        Self::ensure_bufs(&mut s, n);
        let ChainScratch { buf1, buf2 } = &mut *s;

        self.stages[0].apply_mut(side, x, buf1)?;
        let mut in_is_buf1 = true;
        let m = self.stages.len();
        for st in self.stages.iter_mut().skip(1).take(m - 2) {
            if in_is_buf1 {
                st.apply_mut(side, &*buf1, buf2)?;
            } else {
                st.apply_mut(side, &*buf2, buf1)?;
            }
            in_is_buf1 = !in_is_buf1;
        }
        let last = self.stages.last_mut().unwrap();
        if in_is_buf1 {
            last.apply_mut(side, &*buf1, y)?;
        } else {
            last.apply_mut(side, &*buf2, y)?;
        }
        Ok(())
    }

    fn supports_numeric_update(&self) -> bool {
        self.stages.iter().all(|s| s.supports_numeric_update())
    }

    fn update_numeric(&mut self, a: &dyn LinOp<S = f64>) -> Result<(), KError> {
        for st in self.stages.iter_mut() {
            if st.supports_numeric_update() {
                st.update_numeric(a)?;
            } else {
                st.update_symbolic(a)?;
            }
        }
        Ok(())
    }

    fn update_symbolic(&mut self, a: &dyn LinOp<S = f64>) -> Result<(), KError> {
        for st in self.stages.iter_mut() {
            st.update_symbolic(a)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
