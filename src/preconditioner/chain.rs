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
use std::cell::RefCell;

thread_local! {
    static TLS_BUF: RefCell<Vec<f64>> = RefCell::new(Vec::new());
}

/// A simple compositional preconditioner:
/// y = P_k( ... P_2(P_1(x)) ... ) for all PcSide variants.
/// This models M^{-1} ≈ P_k ∘ ... ∘ P_1.
///
pub struct PcChain {
    stages: Vec<Box<dyn Preconditioner>>,
}

impl PcChain {
    pub fn new(stages: Vec<Box<dyn Preconditioner>>) -> Self {
        Self { stages }
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
        // Best-effort pre-size TLS buffer for apply hot path
        let (n, _) = a.dims();
        TLS_BUF.with(|b| b.borrow_mut().resize(n, 0.0));
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
        TLS_BUF.with(|b| -> Result<(), KError> {
            let mut tmp = b.borrow_mut();
            if tmp.len() < x.len() {
                tmp.resize(x.len(), 0.0);
            }
            tmp.copy_from_slice(x);
            for st in &self.stages {
                st.apply(side, &*tmp, y)?;
                tmp.copy_from_slice(y);
            }
            Ok(())
        })
    }

    fn apply_mut(&mut self, side: PcSide, x: &[f64], y: &mut [f64]) -> Result<(), KError> {
        if self.stages.is_empty() {
            y.copy_from_slice(x);
            return Ok(());
        }
        if self.stages.len() == 1 {
            return self.stages[0].apply_mut(side, x, y);
        }
        TLS_BUF.with(|b| -> Result<(), KError> {
            let mut tmp = b.borrow_mut();
            if tmp.len() < x.len() {
                tmp.resize(x.len(), 0.0);
            }
            tmp.copy_from_slice(x);
            for st in self.stages.iter_mut() {
                st.apply_mut(side, &*tmp, y)?;
                tmp.copy_from_slice(y);
            }
            Ok(())
        })
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
