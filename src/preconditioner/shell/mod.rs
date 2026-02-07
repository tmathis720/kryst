use crate::algebra::scalar::S;
use crate::error::KError;
use crate::matrix::op::LinOp;
use crate::preconditioner::{PcSide, Preconditioner};
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

pub trait ShellContext: Send + Sync + std::any::Any {}

impl<T: Send + Sync + std::any::Any> ShellContext for T {}

pub trait ShellContextFactory: Send + Sync {
    fn create(&self) -> Box<dyn ShellContext>;
}

impl<F> ShellContextFactory for F
where
    F: Fn() -> Box<dyn ShellContext> + Send + Sync,
{
    fn create(&self) -> Box<dyn ShellContext> {
        (self)()
    }
}

pub trait ShellApply: Send + Sync {
    fn apply(
        &self,
        side: PcSide,
        x: &[S],
        y: &mut [S],
        ctx: &mut dyn ShellContext,
    ) -> Result<(), KError>;
}

impl<F> ShellApply for F
where
    F: Fn(PcSide, &[S], &mut [S]) -> Result<(), KError> + Send + Sync,
{
    fn apply(
        &self,
        side: PcSide,
        x: &[S],
        y: &mut [S],
        _ctx: &mut dyn ShellContext,
    ) -> Result<(), KError> {
        (self)(side, x, y)
    }
}

impl<F> ShellApply for F
where
    F: Fn(PcSide, &[S], &mut [S], &mut dyn ShellContext) -> Result<(), KError> + Send + Sync,
{
    fn apply(
        &self,
        side: PcSide,
        x: &[S],
        y: &mut [S],
        ctx: &mut dyn ShellContext,
    ) -> Result<(), KError> {
        (self)(side, x, y, ctx)
    }
}

pub trait ShellSetup: Send + Sync {
    fn setup(&self, a: &dyn LinOp<S = S>, ctx: &mut dyn ShellContext) -> Result<(), KError>;
}

impl<F> ShellSetup for F
where
    F: Fn(&dyn LinOp<S = S>) -> Result<(), KError> + Send + Sync,
{
    fn setup(&self, a: &dyn LinOp<S = S>, _ctx: &mut dyn ShellContext) -> Result<(), KError> {
        (self)(a)
    }
}

impl<F> ShellSetup for F
where
    F: Fn(&dyn LinOp<S = S>, &mut dyn ShellContext) -> Result<(), KError> + Send + Sync,
{
    fn setup(&self, a: &dyn LinOp<S = S>, ctx: &mut dyn ShellContext) -> Result<(), KError> {
        (self)(a, ctx)
    }
}

pub trait ShellDestroy: Send + Sync {
    fn destroy(&self, ctx: &mut dyn ShellContext) -> Result<(), KError>;
}

impl<F> ShellDestroy for F
where
    F: Fn() -> Result<(), KError> + Send + Sync,
{
    fn destroy(&self, _ctx: &mut dyn ShellContext) -> Result<(), KError> {
        (self)()
    }
}

impl<F> ShellDestroy for F
where
    F: Fn(&mut dyn ShellContext) -> Result<(), KError> + Send + Sync,
{
    fn destroy(&self, ctx: &mut dyn ShellContext) -> Result<(), KError> {
        (self)(ctx)
    }
}

static APPLY_REGISTRY: Lazy<RwLock<HashMap<String, Arc<dyn ShellApply>>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));
static SETUP_REGISTRY: Lazy<RwLock<HashMap<String, Arc<dyn ShellSetup>>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));
static DESTROY_REGISTRY: Lazy<RwLock<HashMap<String, Arc<dyn ShellDestroy>>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));
static CONTEXT_REGISTRY: Lazy<RwLock<HashMap<String, Arc<dyn ShellContextFactory>>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));

pub fn register_shell_callback(name: impl Into<String>, callback: Arc<dyn ShellApply>) {
    APPLY_REGISTRY
        .write()
        .expect("shell callback registry poisoned")
        .insert(name.into(), callback);
}

pub fn register_shell_setup(name: impl Into<String>, callback: Arc<dyn ShellSetup>) {
    SETUP_REGISTRY
        .write()
        .expect("shell setup registry poisoned")
        .insert(name.into(), callback);
}

pub fn register_shell_destroy(name: impl Into<String>, callback: Arc<dyn ShellDestroy>) {
    DESTROY_REGISTRY
        .write()
        .expect("shell destroy registry poisoned")
        .insert(name.into(), callback);
}

pub fn register_shell_context(name: impl Into<String>, factory: Arc<dyn ShellContextFactory>) {
    CONTEXT_REGISTRY
        .write()
        .expect("shell context registry poisoned")
        .insert(name.into(), factory);
}

pub struct ShellPc {
    callback_name: Option<String>,
    setup_name: Option<String>,
    destroy_name: Option<String>,
    context_name: Option<String>,
    callback: Option<Arc<dyn ShellApply>>,
    setup: Option<Arc<dyn ShellSetup>>,
    destroy: Option<Arc<dyn ShellDestroy>>,
    context_factory: Option<Arc<dyn ShellContextFactory>>,
    context: Mutex<Option<Box<dyn ShellContext>>>,
}

impl ShellPc {
    pub fn new(
        callback_name: Option<String>,
        setup_name: Option<String>,
        destroy_name: Option<String>,
        context_name: Option<String>,
    ) -> Self {
        Self {
            callback_name,
            setup_name,
            destroy_name,
            context_name,
            callback: None,
            setup: None,
            destroy: None,
            context_factory: None,
            context: Mutex::new(None),
        }
    }

    fn ensure_context(
        &self,
        factory: Option<Arc<dyn ShellContextFactory>>,
    ) -> Result<std::sync::MutexGuard<'_, Option<Box<dyn ShellContext>>>, KError> {
        let mut guard = self
            .context
            .lock()
            .map_err(|_| KError::SolveError("shell pc context mutex poisoned".into()))?;
        if guard.is_none() {
            let ctx = factory.map(|f| f.create()).unwrap_or_else(|| Box::new(()));
            *guard = Some(ctx);
        }
        Ok(guard)
    }

    fn shell_error(stage: &str, err: KError) -> KError {
        KError::PcFailed(format!("shell pc {stage} failed: {err}"))
    }
}

impl Preconditioner for ShellPc {
    fn setup(&mut self, a: &dyn LinOp<S = S>) -> Result<(), KError> {
        let factory = if let Some(name) = self.context_name.as_ref() {
            let registry = CONTEXT_REGISTRY
                .read()
                .expect("shell context registry poisoned");
            Some(registry.get(name).cloned().ok_or_else(|| {
                KError::InvalidInput(format!("shell context not registered: {name}"))
            })?)
        } else {
            None
        };
        self.context_factory = factory.clone();

        if let Some(name) = self.callback_name.as_ref() {
            self.callback = APPLY_REGISTRY
                .read()
                .expect("shell callback registry poisoned")
                .get(name)
                .cloned()
                .ok_or_else(|| {
                    KError::InvalidInput(format!("shell callback not registered: {name}"))
                })?;
        }
        if let Some(name) = self.setup_name.as_ref() {
            self.setup = SETUP_REGISTRY
                .read()
                .expect("shell setup registry poisoned")
                .get(name)
                .cloned()
                .ok_or_else(|| KError::InvalidInput(format!("shell setup not registered: {name}")))?;
        }
        if let Some(name) = self.destroy_name.as_ref() {
            self.destroy = DESTROY_REGISTRY
                .read()
                .expect("shell destroy registry poisoned")
                .get(name)
                .cloned()
                .ok_or_else(|| {
                    KError::InvalidInput(format!("shell destroy not registered: {name}"))
                })?;
        }

        let mut guard = self
            .context
            .lock()
            .map_err(|_| KError::SolveError("shell pc context mutex poisoned".into()))?;
        if guard.is_none() {
            let ctx = factory
                .as_ref()
                .map(|f| f.create())
                .unwrap_or_else(|| Box::new(()));
            *guard = Some(ctx);
        }

        if let Some(setup) = self.setup.as_ref() {
            let mut guard = self.ensure_context(factory)?;
            let ctx = guard.as_mut().expect("shell context missing");
            setup
                .setup(a, ctx)
                .map_err(|err| Self::shell_error("setup", err))?;
        }
        Ok(())
    }

    fn apply(&self, side: PcSide, x: &[S], y: &mut [S]) -> Result<(), KError> {
        if let Some(cb) = self.callback.as_ref() {
            let mut guard = self.ensure_context(self.context_factory.clone())?;
            let ctx = guard.as_mut().expect("shell context missing");
            return cb
                .apply(side, x, y, ctx)
                .map_err(|err| Self::shell_error("apply", err));
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

impl Drop for ShellPc {
    fn drop(&mut self) {
        let Some(destroy) = self.destroy.as_ref() else {
            return;
        };
        let mut guard = match self.context.lock() {
            Ok(guard) => guard,
            Err(_) => {
                log::warn!("shell pc context mutex poisoned during drop");
                return;
            }
        };
        let Some(ctx) = guard.as_mut() else {
            return;
        };
        if let Err(err) = destroy.destroy(ctx) {
            log::warn!("shell pc destroy hook failed: {err}");
        }
        *guard = None;
    }
}
