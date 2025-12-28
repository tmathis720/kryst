//! # PcChain
//!
//! Compose PCs sequentially. Common use: cheap smoother before ILU.
//!
//! ```no_run
//! # use kryst::context::pc_context::PcFactory;
//! # use kryst::matrix::MatShell;
//! # use kryst::algebra::prelude::S;
//! # use std::sync::Arc;
//! let specs = PcFactory::create_pc_chain_from_str("jacobi->ilut", None).unwrap();
//! // later, when P is known:
//! # let p = MatShell::<S>::new(10, 10, |x, y| y.copy_from_slice(x));
//! # let p = Arc::new(p);
//! let chain = PcFactory::construct_deferred_pc_chain(specs, p.as_ref()).unwrap();
//! ```

#[cfg(feature = "complex")]
use crate::algebra::bridge::BridgeScratch;
#[allow(unused_imports)]
use crate::algebra::prelude::*;
use crate::error::KError;
use crate::matrix::backend::materialize_ref;
use crate::matrix::op::LinOp;
#[cfg(feature = "complex")]
use crate::ops::kpc::KPreconditioner;
#[cfg(feature = "complex")]
use crate::preconditioner::bridge::{
    apply_pc_mut_s as bridge_apply_pc_mut_s, apply_pc_s as bridge_apply_pc_s,
};
use crate::preconditioner::{PcSide, Preconditioner};
use std::cell::RefCell;

thread_local! {
    static TLS_BUF: RefCell<Vec<S>> = const { RefCell::new(Vec::new()) };
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
    fn setup(&mut self, a: &dyn LinOp<S = S>) -> Result<(), KError> {
        for st in self.stages.iter_mut() {
            let want = st.required_format();
            let tol = st.preferred_drop_tol_for_format().unwrap_or_default();
            if want.is_any() || a.format() == want {
                st.setup(a)?;
            } else {
                let view = materialize_ref(a, want, tol)?;
                st.setup(view.as_ref())?;
            }
        }
        // Best-effort pre-size TLS buffer for apply hot path
        let (n, _) = a.dims();
        TLS_BUF.with(|b| b.borrow_mut().resize(n, S::zero()));
        Ok(())
    }

    fn apply(&self, side: PcSide, x: &[S], y: &mut [S]) -> Result<(), KError> {
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
                tmp.resize(x.len(), S::zero());
            }
            tmp.copy_from_slice(x);
            for st in &self.stages {
                st.apply(side, &tmp, y)?;
                tmp.copy_from_slice(y);
            }
            Ok(())
        })
    }

    fn apply_mut(&mut self, side: PcSide, x: &[S], y: &mut [S]) -> Result<(), KError> {
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
                tmp.resize(x.len(), S::zero());
            }
            tmp.copy_from_slice(x);
            for st in self.stages.iter_mut() {
                st.apply_mut(side, &tmp, y)?;
                tmp.copy_from_slice(y);
            }
            Ok(())
        })
    }

    fn supports_numeric_update(&self) -> bool {
        self.stages.iter().all(|s| s.supports_numeric_update())
    }

    fn update_numeric(&mut self, a: &dyn LinOp<S = S>) -> Result<(), KError> {
        for st in self.stages.iter_mut() {
            let want = st.required_format();
            let tol = st.preferred_drop_tol_for_format().unwrap_or_default();
            if want.is_any() || a.format() == want {
                if st.supports_numeric_update() {
                    st.update_numeric(a)?;
                } else {
                    st.update_symbolic(a)?;
                }
            } else {
                let view = materialize_ref(a, want, tol)?;
                if st.supports_numeric_update() {
                    st.update_numeric(view.as_ref())?;
                } else {
                    st.update_symbolic(view.as_ref())?;
                }
            }
        }
        Ok(())
    }

    fn update_symbolic(&mut self, a: &dyn LinOp<S = S>) -> Result<(), KError> {
        for st in self.stages.iter_mut() {
            let want = st.required_format();
            let tol = st.preferred_drop_tol_for_format().unwrap_or_default();
            if want.is_any() || a.format() == want {
                st.update_symbolic(a)?;
            } else {
                let view = materialize_ref(a, want, tol)?;
                st.update_symbolic(view.as_ref())?;
            }
        }
        Ok(())
    }
}

#[cfg(feature = "complex")]
impl KPreconditioner for PcChain {
    type Scalar = S;

    #[inline]
    fn dims(&self) -> (usize, usize) {
        Preconditioner::dims(self)
    }

    fn apply_s(
        &self,
        side: PcSide,
        x: &[S],
        y: &mut [S],
        scratch: &mut BridgeScratch,
    ) -> Result<(), KError> {
        bridge_apply_pc_s(self, side, x, y, scratch)
    }

    fn apply_mut_s(
        &mut self,
        side: PcSide,
        x: &[S],
        y: &mut [S],
        scratch: &mut BridgeScratch,
    ) -> Result<(), KError> {
        bridge_apply_pc_mut_s(self, side, x, y, scratch)
    }

    fn on_restart_s(&mut self, outer_iter: usize, residual_norm: R) -> Result<(), KError> {
        Preconditioner::on_restart(self, outer_iter, residual_norm)
    }
}

#[cfg(test)]
mod tests;
