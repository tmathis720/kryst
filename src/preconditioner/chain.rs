//! # PcChain
//!
//! Compose PCs sequentially. Common use: cheap smoother before ILU.
//!
//! ```no_run
//! # use kryst::context::pc_context::PcFactory;
//! # use faer::Mat;
//! let specs = PcFactory::create_pc_chain_from_str("jacobi->ilut", None).unwrap();
//! // later, when P is known:
//! # let p = Mat::<f64>::zeros(10,10);
//! let chain = PcFactory::construct_deferred_pc_chain(specs, &p).unwrap();
//! ```

use crate::error::KError;
use crate::matrix::convert::materialize_linop_with_hint;
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
        Self {
            stages,
            scratch: Mutex::new(ChainScratch::default()),
        }
    }

    #[inline]
    fn ensure_bufs(s: &mut ChainScratch, n: usize) {
        if s.buf1.len() != n {
            s.buf1.resize(n, 0.0);
        }
        if s.buf2.len() != n {
            s.buf2.resize(n, 0.0);
        }
    }

    pub fn len(&self) -> usize {
        self.stages.len()
    }
    pub fn is_empty(&self) -> bool {
        self.stages.is_empty()
    }
}

impl Preconditioner for PcChain {
    fn setup(&mut self, a: &dyn LinOp<S = f64>) -> Result<(), KError> {
        for st in self.stages.iter_mut() {
            let hint = st.required_format();
            let tol = st.preferred_drop_tol_for_format().unwrap_or(0.0);
            let view = materialize_linop_with_hint(a, hint, tol)?;
            st.setup(view.as_ref())?;
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
            let hint = st.required_format();
            let tol = st.preferred_drop_tol_for_format().unwrap_or(0.0);
            let view = materialize_linop_with_hint(a, hint, tol)?;
            if st.supports_numeric_update() {
                st.update_numeric(view.as_ref())?;
            } else {
                st.update_symbolic(view.as_ref())?;
            }
        }
        Ok(())
    }

    fn update_symbolic(&mut self, a: &dyn LinOp<S = f64>) -> Result<(), KError> {
        for st in self.stages.iter_mut() {
            let hint = st.required_format();
            let tol = st.preferred_drop_tol_for_format().unwrap_or(0.0);
            let view = materialize_linop_with_hint(a, hint, tol)?;
            st.update_symbolic(view.as_ref())?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
