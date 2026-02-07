use crate::algebra::scalar::S;
use crate::error::KError;
use crate::matrix::op::LinOp;
use crate::preconditioner::{PcSide, Preconditioner};
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

pub trait ShellApply: Send + Sync {
    fn apply(&self, side: PcSide, x: &[S], y: &mut [S]) -> Result<(), KError>;
}

impl<F> ShellApply for F
where
    F: Fn(PcSide, &[S], &mut [S]) -> Result<(), KError> + Send + Sync,
{
    fn apply(&self, side: PcSide, x: &[S], y: &mut [S]) -> Result<(), KError> {
        (self)(side, x, y)
    }
}

static REGISTRY: Lazy<RwLock<HashMap<String, Arc<dyn ShellApply>>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));

pub fn register_shell_callback(name: impl Into<String>, callback: Arc<dyn ShellApply>) {
    REGISTRY
        .write()
        .expect("shell callback registry poisoned")
        .insert(name.into(), callback);
}

pub struct ShellPc {
    callback_name: Option<String>,
    callback: Option<Arc<dyn ShellApply>>,
}

impl ShellPc {
    pub fn new(callback_name: Option<String>) -> Self {
        Self {
            callback_name,
            callback: None,
        }
    }
}

impl Preconditioner for ShellPc {
    fn setup(&mut self, _a: &dyn LinOp<S = S>) -> Result<(), KError> {
        if let Some(name) = self.callback_name.as_ref() {
            self.callback = REGISTRY
                .read()
                .expect("shell callback registry poisoned")
                .get(name)
                .cloned();
        }
        Ok(())
    }

    fn apply(&self, side: PcSide, x: &[S], y: &mut [S]) -> Result<(), KError> {
        if let Some(cb) = self.callback.as_ref() {
            return cb.apply(side, x, y);
        }
        if x.len() != y.len() {
            return Err(KError::InvalidInput(
                "shell pc input/output length mismatch".into(),
            ));
        }
        y.copy_from_slice(x);
        Ok(())
    }
}
