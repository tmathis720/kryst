use crate::algebra::scalar::S;
use crate::error::KError;
use crate::matrix::op::LinOp;
use crate::preconditioner::{Op, PcCaps, PcSide, Preconditioner};
use crate::utils::convergence::{
    ConvergedReason, FailureReasonKind, FailureStage, NestedPcFailure,
};
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

pub trait ShellContext: Send + Sync + std::any::Any {}

impl<T: Send + Sync + std::any::Any> ShellContext for T {}

pub fn shell_context_downcast_ref<T: ShellContext + 'static>(
    ctx: &dyn ShellContext,
) -> Result<&T, KError> {
    let any = ctx as &dyn std::any::Any;
    if any.is::<T>() {
        return Ok(any
            .downcast_ref::<T>()
            .expect("checked context downcast should succeed"));
    }
    if any.is::<Box<dyn ShellContext>>() {
        let boxed = any
            .downcast_ref::<Box<dyn ShellContext>>()
            .expect("checked boxed shell context downcast should succeed");
        return shell_context_downcast_ref::<T>(boxed.as_ref());
    }
    Err(KError::InvalidInput(format!(
        "shell context type mismatch: expected {}",
        std::any::type_name::<T>()
    )))
}

pub fn shell_context_downcast_mut<T: ShellContext + 'static>(
    ctx: &mut dyn ShellContext,
) -> Result<&mut T, KError> {
    let any = ctx as &mut dyn std::any::Any;
    if any.is::<T>() {
        return Ok(any
            .downcast_mut::<T>()
            .expect("checked context downcast should succeed"));
    }
    if any.is::<Box<dyn ShellContext>>() {
        let boxed = any
            .downcast_mut::<Box<dyn ShellContext>>()
            .expect("checked boxed shell context downcast should succeed");
        return shell_context_downcast_mut::<T>(boxed.as_mut());
    }
    Err(KError::InvalidInput(format!(
        "shell context type mismatch: expected {}",
        std::any::type_name::<T>()
    )))
}

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

pub struct ShellApplyFn<F>(F);

impl<F> ShellApply for ShellApplyFn<F>
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
        (self.0)(side, x, y)
    }
}

pub struct ShellApplyWithContext<F>(F);

impl<F> ShellApply for ShellApplyWithContext<F>
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
        (self.0)(side, x, y, ctx)
    }
}

pub trait ShellSetup: Send + Sync {
    fn setup(&self, a: &dyn LinOp<S = S>, ctx: &mut dyn ShellContext) -> Result<(), KError>;
}

pub struct ShellSetupFn<F>(F);

impl<F> ShellSetup for ShellSetupFn<F>
where
    F: Fn(&dyn LinOp<S = S>) -> Result<(), KError> + Send + Sync,
{
    fn setup(&self, a: &dyn LinOp<S = S>, _ctx: &mut dyn ShellContext) -> Result<(), KError> {
        (self.0)(a)
    }
}

pub struct ShellSetupWithContext<F>(F);

impl<F> ShellSetup for ShellSetupWithContext<F>
where
    F: Fn(&dyn LinOp<S = S>, &mut dyn ShellContext) -> Result<(), KError> + Send + Sync,
{
    fn setup(&self, a: &dyn LinOp<S = S>, ctx: &mut dyn ShellContext) -> Result<(), KError> {
        (self.0)(a, ctx)
    }
}

pub trait ShellDestroy: Send + Sync {
    fn destroy(&self, ctx: &mut dyn ShellContext) -> Result<(), KError>;
}

pub struct ShellDestroyFn<F>(F);

impl<F> ShellDestroy for ShellDestroyFn<F>
where
    F: Fn() -> Result<(), KError> + Send + Sync,
{
    fn destroy(&self, _ctx: &mut dyn ShellContext) -> Result<(), KError> {
        (self.0)()
    }
}

pub struct ShellDestroyWithContext<F>(F);

impl<F> ShellDestroy for ShellDestroyWithContext<F>
where
    F: Fn(&mut dyn ShellContext) -> Result<(), KError> + Send + Sync,
{
    fn destroy(&self, ctx: &mut dyn ShellContext) -> Result<(), KError> {
        (self.0)(ctx)
    }
}

static APPLY_REGISTRY: Lazy<RwLock<HashMap<String, Arc<dyn ShellApply>>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));
static APPLY_TRANSPOSE_REGISTRY: Lazy<RwLock<HashMap<String, Arc<dyn ShellApply>>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));
static APPLY_CONJ_TRANSPOSE_REGISTRY: Lazy<RwLock<HashMap<String, Arc<dyn ShellApply>>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));
static APPLY_SYMMETRIC_REGISTRY: Lazy<RwLock<HashMap<String, Arc<dyn ShellApply>>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));
static APPLY_SYMMETRIC_LEFT_REGISTRY: Lazy<RwLock<HashMap<String, Arc<dyn ShellApply>>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));
static APPLY_SYMMETRIC_RIGHT_REGISTRY: Lazy<RwLock<HashMap<String, Arc<dyn ShellApply>>>> =
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

pub fn register_shell_apply_transpose(name: impl Into<String>, callback: Arc<dyn ShellApply>) {
    APPLY_TRANSPOSE_REGISTRY
        .write()
        .expect("shell transpose callback registry poisoned")
        .insert(name.into(), callback);
}

pub fn register_shell_apply_conjugate_transpose(
    name: impl Into<String>,
    callback: Arc<dyn ShellApply>,
) {
    APPLY_CONJ_TRANSPOSE_REGISTRY
        .write()
        .expect("shell conjugate-transpose callback registry poisoned")
        .insert(name.into(), callback);
}

pub fn register_shell_apply_symmetric(name: impl Into<String>, callback: Arc<dyn ShellApply>) {
    APPLY_SYMMETRIC_REGISTRY
        .write()
        .expect("shell symmetric callback registry poisoned")
        .insert(name.into(), callback);
}

pub fn register_shell_apply_symmetric_left(name: impl Into<String>, callback: Arc<dyn ShellApply>) {
    APPLY_SYMMETRIC_LEFT_REGISTRY
        .write()
        .expect("shell symmetric-left callback registry poisoned")
        .insert(name.into(), callback);
}

pub fn register_shell_apply_symmetric_right(
    name: impl Into<String>,
    callback: Arc<dyn ShellApply>,
) {
    APPLY_SYMMETRIC_RIGHT_REGISTRY
        .write()
        .expect("shell symmetric-right callback registry poisoned")
        .insert(name.into(), callback);
}

pub fn register_shell_apply_typed<T, F>(name: impl Into<String>, callback: F)
where
    T: ShellContext + 'static,
    F: Fn(PcSide, &[S], &mut [S], &mut T) -> Result<(), KError> + Send + Sync + 'static,
{
    register_shell_callback(name, shell_apply_with_typed_context(callback));
}

pub fn register_shell_apply_transpose_typed<T, F>(name: impl Into<String>, callback: F)
where
    T: ShellContext + 'static,
    F: Fn(PcSide, &[S], &mut [S], &mut T) -> Result<(), KError> + Send + Sync + 'static,
{
    register_shell_apply_transpose(name, shell_apply_with_typed_context(callback));
}

pub fn register_shell_apply_conjugate_transpose_typed<T, F>(name: impl Into<String>, callback: F)
where
    T: ShellContext + 'static,
    F: Fn(PcSide, &[S], &mut [S], &mut T) -> Result<(), KError> + Send + Sync + 'static,
{
    register_shell_apply_conjugate_transpose(name, shell_apply_with_typed_context(callback));
}

pub fn register_shell_apply_symmetric_typed<T, F>(name: impl Into<String>, callback: F)
where
    T: ShellContext + 'static,
    F: Fn(PcSide, &[S], &mut [S], &mut T) -> Result<(), KError> + Send + Sync + 'static,
{
    register_shell_apply_symmetric(name, shell_apply_with_typed_context(callback));
}

pub fn register_shell_apply_symmetric_left_typed<T, F>(name: impl Into<String>, callback: F)
where
    T: ShellContext + 'static,
    F: Fn(PcSide, &[S], &mut [S], &mut T) -> Result<(), KError> + Send + Sync + 'static,
{
    register_shell_apply_symmetric_left(name, shell_apply_with_typed_context(callback));
}

pub fn register_shell_apply_symmetric_right_typed<T, F>(name: impl Into<String>, callback: F)
where
    T: ShellContext + 'static,
    F: Fn(PcSide, &[S], &mut [S], &mut T) -> Result<(), KError> + Send + Sync + 'static,
{
    register_shell_apply_symmetric_right(name, shell_apply_with_typed_context(callback));
}

pub fn shell_apply<F>(callback: F) -> Arc<dyn ShellApply>
where
    F: Fn(PcSide, &[S], &mut [S]) -> Result<(), KError> + Send + Sync + 'static,
{
    Arc::new(ShellApplyFn(callback))
}

pub fn shell_apply_with_context<F>(callback: F) -> Arc<dyn ShellApply>
where
    F: Fn(PcSide, &[S], &mut [S], &mut dyn ShellContext) -> Result<(), KError>
        + Send
        + Sync
        + 'static,
{
    Arc::new(ShellApplyWithContext(callback))
}

pub fn register_shell_setup(name: impl Into<String>, callback: Arc<dyn ShellSetup>) {
    SETUP_REGISTRY
        .write()
        .expect("shell setup registry poisoned")
        .insert(name.into(), callback);
}

pub fn shell_setup<F>(callback: F) -> Arc<dyn ShellSetup>
where
    F: Fn(&dyn LinOp<S = S>) -> Result<(), KError> + Send + Sync + 'static,
{
    Arc::new(ShellSetupFn(callback))
}

pub fn shell_setup_with_context<F>(callback: F) -> Arc<dyn ShellSetup>
where
    F: Fn(&dyn LinOp<S = S>, &mut dyn ShellContext) -> Result<(), KError> + Send + Sync + 'static,
{
    Arc::new(ShellSetupWithContext(callback))
}

pub fn register_shell_destroy(name: impl Into<String>, callback: Arc<dyn ShellDestroy>) {
    DESTROY_REGISTRY
        .write()
        .expect("shell destroy registry poisoned")
        .insert(name.into(), callback);
}

pub fn shell_destroy<F>(callback: F) -> Arc<dyn ShellDestroy>
where
    F: Fn() -> Result<(), KError> + Send + Sync + 'static,
{
    Arc::new(ShellDestroyFn(callback))
}

pub fn shell_destroy_with_context<F>(callback: F) -> Arc<dyn ShellDestroy>
where
    F: Fn(&mut dyn ShellContext) -> Result<(), KError> + Send + Sync + 'static,
{
    Arc::new(ShellDestroyWithContext(callback))
}

pub fn register_shell_context(name: impl Into<String>, factory: Arc<dyn ShellContextFactory>) {
    CONTEXT_REGISTRY
        .write()
        .expect("shell context registry poisoned")
        .insert(name.into(), factory);
}

pub fn shell_context_factory<T, F>(factory: F) -> Arc<dyn ShellContextFactory>
where
    T: ShellContext + 'static,
    F: Fn() -> T + Send + Sync + 'static,
{
    Arc::new(move || Box::new(factory()) as Box<dyn ShellContext>)
}

pub fn register_shell_context_typed<T, F>(name: impl Into<String>, factory: F)
where
    T: ShellContext + 'static,
    F: Fn() -> T + Send + Sync + 'static,
{
    register_shell_context(name, shell_context_factory(factory));
}

pub fn register_shell_context_shared<T>(name: impl Into<String>, shared: Arc<T>)
where
    T: Send + Sync + 'static,
{
    register_shell_context(
        name,
        shell_context_factory(move || shared.clone() as Arc<T>),
    );
}

pub fn register_shell_context_shared_rwlock<T>(name: impl Into<String>, shared: Arc<RwLock<T>>)
where
    T: Send + Sync + 'static,
{
    register_shell_context(
        name,
        shell_context_factory(move || shared.clone() as Arc<RwLock<T>>),
    );
}

pub fn shell_apply_with_typed_context<T, F>(callback: F) -> Arc<dyn ShellApply>
where
    T: ShellContext + 'static,
    F: Fn(PcSide, &[S], &mut [S], &mut T) -> Result<(), KError> + Send + Sync + 'static,
{
    shell_apply_with_context(move |side, x, y, ctx| {
        callback(side, x, y, shell_context_downcast_mut::<T>(ctx)?)
    })
}

pub fn shell_setup_with_typed_context<T, F>(callback: F) -> Arc<dyn ShellSetup>
where
    T: ShellContext + 'static,
    F: Fn(&dyn LinOp<S = S>, &mut T) -> Result<(), KError> + Send + Sync + 'static,
{
    shell_setup_with_context(move |a, ctx| callback(a, shell_context_downcast_mut::<T>(ctx)?))
}

pub fn shell_destroy_with_typed_context<T, F>(callback: F) -> Arc<dyn ShellDestroy>
where
    T: ShellContext + 'static,
    F: Fn(&mut T) -> Result<(), KError> + Send + Sync + 'static,
{
    shell_destroy_with_context(move |ctx| callback(shell_context_downcast_mut::<T>(ctx)?))
}

pub struct ShellPc {
    callback_name: Option<String>,
    callback_transpose_name: Option<String>,
    callback_conjugate_transpose_name: Option<String>,
    callback_symmetric_name: Option<String>,
    callback_symmetric_left_name: Option<String>,
    callback_symmetric_right_name: Option<String>,
    setup_name: Option<String>,
    destroy_name: Option<String>,
    context_name: Option<String>,
    callback: Option<Arc<dyn ShellApply>>,
    callback_transpose: Option<Arc<dyn ShellApply>>,
    callback_conjugate_transpose: Option<Arc<dyn ShellApply>>,
    callback_symmetric: Option<Arc<dyn ShellApply>>,
    callback_symmetric_left: Option<Arc<dyn ShellApply>>,
    callback_symmetric_right: Option<Arc<dyn ShellApply>>,
    setup: Option<Arc<dyn ShellSetup>>,
    destroy: Option<Arc<dyn ShellDestroy>>,
    context_factory: Option<Arc<dyn ShellContextFactory>>,
    context: Mutex<Option<Box<dyn ShellContext>>>,
}

impl ShellPc {
    pub fn new(
        callback_name: Option<String>,
        callback_transpose_name: Option<String>,
        callback_conjugate_transpose_name: Option<String>,
        callback_symmetric_name: Option<String>,
        callback_symmetric_left_name: Option<String>,
        callback_symmetric_right_name: Option<String>,
        setup_name: Option<String>,
        destroy_name: Option<String>,
        context_name: Option<String>,
    ) -> Self {
        Self {
            callback_name,
            callback_transpose_name,
            callback_conjugate_transpose_name,
            callback_symmetric_name,
            callback_symmetric_left_name,
            callback_symmetric_right_name,
            setup_name,
            destroy_name,
            context_name,
            callback: None,
            callback_transpose: None,
            callback_conjugate_transpose: None,
            callback_symmetric: None,
            callback_symmetric_left: None,
            callback_symmetric_right: None,
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

    fn shell_error(stage: FailureStage, hook: &'static str, err: KError) -> KError {
        let reason = match stage {
            FailureStage::Setup => ConvergedReason::from_failure_kind(FailureReasonKind::PcSetup),
            FailureStage::Solve => ConvergedReason::from_failure_kind(FailureReasonKind::PcApply),
        };
        let nested_detail = match &err {
            KError::NestedPcFailed(inner) => format!(
                " inner_component={} inner_reason={} inner_detail={}",
                inner.component, inner.reason, inner.detail
            ),
            _ => String::new(),
        };
        KError::NestedPcFailed(NestedPcFailure {
            component: "pc_shell",
            reason,
            iterations: 0,
            final_norm: None,
            residual_history_summary: None,
            detail: format!("stage={stage:?} hook={hook} nested_error={err}{nested_detail}"),
        })
    }

    pub fn with_typed_context<T, F, R>(&self, callback: F) -> Result<R, KError>
    where
        T: ShellContext + 'static,
        F: FnOnce(&mut T) -> Result<R, KError>,
    {
        let mut guard = self.ensure_context(self.context_factory.clone())?;
        let ctx = guard.as_mut().expect("shell context missing").as_mut();
        let typed = shell_context_downcast_mut::<T>(ctx)?;
        callback(typed)
    }

    pub fn with_typed_context_ref<T, F, R>(&self, callback: F) -> Result<R, KError>
    where
        T: ShellContext + 'static,
        F: FnOnce(&T) -> Result<R, KError>,
    {
        let guard = self
            .context
            .lock()
            .map_err(|_| KError::SolveError("shell pc context mutex poisoned".into()))?;
        let Some(ctx) = guard.as_ref() else {
            return Err(KError::InvalidInput(
                "shell context is not initialized".into(),
            ));
        };
        let typed = shell_context_downcast_ref::<T>(ctx.as_ref())?;
        callback(typed)
    }

    pub fn with_shared_context<T, F, R>(&self, callback: F) -> Result<R, KError>
    where
        T: Send + Sync + 'static,
        F: FnOnce(&Arc<T>) -> Result<R, KError>,
    {
        self.with_typed_context_ref::<Arc<T>, _, _>(callback)
    }

    pub fn with_shared_context_rwlock<T, F, R>(&self, callback: F) -> Result<R, KError>
    where
        T: Send + Sync + 'static,
        F: FnOnce(&Arc<RwLock<T>>) -> Result<R, KError>,
    {
        self.with_typed_context_ref::<Arc<RwLock<T>>, _, _>(callback)
    }

    fn invoke_apply(
        &self,
        callback: Option<&Arc<dyn ShellApply>>,
        hook: &'static str,
        stage: FailureStage,
        side: PcSide,
        x: &[S],
        y: &mut [S],
    ) -> Result<(), KError> {
        if let Some(cb) = callback {
            let mut guard = self.ensure_context(self.context_factory.clone())?;
            let ctx = guard.as_mut().expect("shell context missing");
            return cb
                .apply(side, x, y, ctx)
                .map_err(|err| Self::shell_error(stage, hook, err));
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
            self.callback = Some(
                APPLY_REGISTRY
                    .read()
                    .expect("shell callback registry poisoned")
                    .get(name)
                    .cloned()
                    .ok_or_else(|| {
                        KError::InvalidInput(format!("shell callback not registered: {name}"))
                    })?,
            );
        }
        if let Some(name) = self.setup_name.as_ref() {
            self.setup = Some(
                SETUP_REGISTRY
                    .read()
                    .expect("shell setup registry poisoned")
                    .get(name)
                    .cloned()
                    .ok_or_else(|| {
                        KError::InvalidInput(format!("shell setup not registered: {name}"))
                    })?,
            );
        }
        if let Some(name) = self.callback_transpose_name.as_ref() {
            self.callback_transpose = Some(
                APPLY_TRANSPOSE_REGISTRY
                    .read()
                    .expect("shell transpose callback registry poisoned")
                    .get(name)
                    .cloned()
                    .ok_or_else(|| {
                        KError::InvalidInput(format!(
                            "shell transpose callback not registered: {name}"
                        ))
                    })?,
            );
        }
        if let Some(name) = self.callback_conjugate_transpose_name.as_ref() {
            self.callback_conjugate_transpose = Some(
                APPLY_CONJ_TRANSPOSE_REGISTRY
                    .read()
                    .expect("shell conjugate-transpose callback registry poisoned")
                    .get(name)
                    .cloned()
                    .ok_or_else(|| {
                        KError::InvalidInput(format!(
                            "shell conjugate-transpose callback not registered: {name}"
                        ))
                    })?,
            );
        }
        if let Some(name) = self.callback_symmetric_name.as_ref() {
            self.callback_symmetric = Some(
                APPLY_SYMMETRIC_REGISTRY
                    .read()
                    .expect("shell symmetric callback registry poisoned")
                    .get(name)
                    .cloned()
                    .ok_or_else(|| {
                        KError::InvalidInput(format!(
                            "shell symmetric callback not registered: {name}"
                        ))
                    })?,
            );
        }
        if let Some(name) = self.callback_symmetric_left_name.as_ref() {
            self.callback_symmetric_left = Some(
                APPLY_SYMMETRIC_LEFT_REGISTRY
                    .read()
                    .expect("shell symmetric-left callback registry poisoned")
                    .get(name)
                    .cloned()
                    .ok_or_else(|| {
                        KError::InvalidInput(format!(
                            "shell symmetric-left callback not registered: {name}"
                        ))
                    })?,
            );
        }
        if let Some(name) = self.callback_symmetric_right_name.as_ref() {
            self.callback_symmetric_right = Some(
                APPLY_SYMMETRIC_RIGHT_REGISTRY
                    .read()
                    .expect("shell symmetric-right callback registry poisoned")
                    .get(name)
                    .cloned()
                    .ok_or_else(|| {
                        KError::InvalidInput(format!(
                            "shell symmetric-right callback not registered: {name}"
                        ))
                    })?,
            );
        }
        if let Some(name) = self.destroy_name.as_ref() {
            self.destroy = Some(
                DESTROY_REGISTRY
                    .read()
                    .expect("shell destroy registry poisoned")
                    .get(name)
                    .cloned()
                    .ok_or_else(|| {
                        KError::InvalidInput(format!("shell destroy not registered: {name}"))
                    })?,
            );
        }

        let mut guard = self
            .context
            .lock()
            .map_err(|_| KError::SolveError("shell pc context mutex poisoned".into()))?;
        match factory.as_ref() {
            Some(factory) => {
                *guard = Some(factory.create());
            }
            None if guard.is_none() => {
                *guard = Some(Box::new(()));
            }
            None => {}
        }

        if let Some(setup) = self.setup.as_ref() {
            let ctx = guard.as_mut().expect("shell context missing");
            setup
                .setup(a, ctx)
                .map_err(|err| Self::shell_error(FailureStage::Setup, "setup", err))?;
        }
        Ok(())
    }

    fn apply(&self, side: PcSide, x: &[S], y: &mut [S]) -> Result<(), KError> {
        if matches!(side, PcSide::Symmetric) {
            if self.callback_symmetric_left.is_some() {
                self.invoke_apply(
                    self.callback_symmetric_left.as_ref(),
                    "apply_symmetric_left",
                    FailureStage::Solve,
                    PcSide::Left,
                    x,
                    y,
                )?;
                let tmp = y.to_vec();
                return self.invoke_apply(
                    self.callback_symmetric_right
                        .as_ref()
                        .or(self.callback_symmetric_left.as_ref()),
                    "apply_symmetric_right",
                    FailureStage::Solve,
                    PcSide::Right,
                    &tmp,
                    y,
                );
            }
            if self.callback_symmetric.is_some() {
                return self.invoke_apply(
                    self.callback_symmetric.as_ref(),
                    "apply_symmetric",
                    FailureStage::Solve,
                    side,
                    x,
                    y,
                );
            }
        }
        self.invoke_apply(
            self.callback.as_ref(),
            "apply",
            FailureStage::Solve,
            side,
            x,
            y,
        )
    }

    fn apply_op(&self, op: Op, x: &[S], y: &mut [S]) -> Result<(), KError> {
        match op {
            Op::NoTrans => self.apply(PcSide::Left, x, y),
            Op::Trans => self.invoke_apply(
                self.callback_transpose.as_ref().or(self.callback.as_ref()),
                "apply_transpose",
                FailureStage::Solve,
                PcSide::Left,
                x,
                y,
            ),
            Op::ConjTrans => self.invoke_apply(
                self.callback_conjugate_transpose
                    .as_ref()
                    .or(self.callback_transpose.as_ref())
                    .or(self.callback.as_ref()),
                "apply_conjugate_transpose",
                FailureStage::Solve,
                PcSide::Left,
                x,
                y,
            ),
        }
    }

    fn capabilities(&self) -> PcCaps {
        let supports_transpose = self.callback_transpose.is_some() || self.callback.is_some();
        let supports_conj_trans = self.callback_conjugate_transpose.is_some()
            || self.callback_transpose.is_some()
            || self.callback.is_some();
        PcCaps {
            supports_transpose,
            supports_conj_trans,
            ..PcCaps::default()
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::options::{KspOptions, PcOptions};
    use crate::context::ksp_context::{KspContext, SolverType};
    use crate::matrix::op::LinOp;
    use std::sync::{Arc, Mutex};

    #[derive(Default)]
    struct TestOp;

    impl LinOp for TestOp {
        type S = S;

        fn dims(&self) -> (usize, usize) {
            (2, 2)
        }

        fn matvec(&self, x: &[S], y: &mut [S]) {
            y.copy_from_slice(x);
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    #[derive(Debug, Default)]
    struct HookCtx {
        log: Vec<&'static str>,
        apply_calls: usize,
        trans_calls: usize,
        sym_calls: usize,
        destroy_calls: usize,
    }

    #[test]
    fn shell_hooks_invoke_in_order_and_propagate_context() {
        let tag = "shell_hook_order";
        register_shell_context_typed(format!("{tag}_ctx"), HookCtx::default);

        register_shell_setup(format!("{tag}_setup"), shell_setup(|_a| Ok(())));

        register_shell_callback(
            format!("{tag}_apply"),
            shell_apply_with_typed_context(|_side, x, y, ctx: &mut HookCtx| {
                ctx.apply_calls += 1;
                ctx.log.push("apply");
                y.copy_from_slice(x);
                Ok(())
            }),
        );

        register_shell_apply_transpose(
            format!("{tag}_transpose"),
            shell_apply_with_typed_context(|_side, x, y, ctx: &mut HookCtx| {
                ctx.trans_calls += 1;
                ctx.log.push("transpose");
                y.copy_from_slice(x);
                Ok(())
            }),
        );
        register_shell_apply_conjugate_transpose(
            format!("{tag}_conj_transpose"),
            shell_apply_with_typed_context(|_side, x, y, ctx: &mut HookCtx| {
                ctx.trans_calls += 1;
                ctx.log.push("conjugate_transpose");
                y.copy_from_slice(x);
                Ok(())
            }),
        );

        register_shell_apply_symmetric(
            format!("{tag}_symmetric"),
            shell_apply_with_typed_context(|_side, x, y, ctx: &mut HookCtx| {
                ctx.sym_calls += 1;
                ctx.log.push("symmetric");
                y.copy_from_slice(x);
                Ok(())
            }),
        );

        let final_log = Arc::new(Mutex::new(Vec::new()));
        let final_log_clone = Arc::clone(&final_log);
        register_shell_destroy(
            format!("{tag}_destroy"),
            shell_destroy_with_typed_context(move |ctx: &mut HookCtx| {
                ctx.destroy_calls += 1;
                ctx.log.push("destroy");
                *final_log_clone.lock().expect("final log mutex poisoned") = ctx.log.clone();
                Ok(())
            }),
        );

        {
            let mut pc = ShellPc::new(
                Some(format!("{tag}_apply")),
                Some(format!("{tag}_transpose")),
                Some(format!("{tag}_conj_transpose")),
                Some(format!("{tag}_symmetric")),
                None,
                None,
                Some(format!("{tag}_setup")),
                Some(format!("{tag}_destroy")),
                Some(format!("{tag}_ctx")),
            );
            pc.setup(&TestOp).expect("setup should succeed");

            let x = vec![1.0, 2.0];
            let mut y = vec![0.0, 0.0];
            pc.apply(PcSide::Left, &x, &mut y)
                .expect("forward apply should succeed");
            pc.apply_op(Op::Trans, &x, &mut y)
                .expect("transpose apply should succeed");
            pc.apply_op(Op::ConjTrans, &x, &mut y)
                .expect("conjugate-transpose apply should succeed");
            pc.apply(PcSide::Symmetric, &x, &mut y)
                .expect("symmetric apply should succeed");

            let mut guard = pc.context.lock().expect("context mutex poisoned");
            let ctx = guard.as_mut().expect("context should exist").as_mut();
            let typed =
                shell_context_downcast_mut::<HookCtx>(ctx).expect("context type should match");
            assert_eq!(typed.apply_calls, 1);
            assert_eq!(typed.trans_calls, 2);
            assert_eq!(typed.sym_calls, 1);
            drop(guard);
        }

        let log = final_log.lock().expect("final log mutex poisoned").clone();
        assert_eq!(
            log,
            vec![
                "apply",
                "transpose",
                "conjugate_transpose",
                "symmetric",
                "destroy"
            ]
        );
    }

    #[test]
    fn shell_pc_typed_context_helpers_surface_values() {
        let tag = "shell_typed_context_api";
        register_shell_context_typed(format!("{tag}_ctx"), || HookCtx {
            apply_calls: 3,
            ..HookCtx::default()
        });

        let mut pc = ShellPc::new(
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            Some(format!("{tag}_ctx")),
        );
        pc.setup(&TestOp).expect("setup should succeed");

        let calls = pc
            .with_typed_context_ref::<HookCtx, _, _>(|ctx| Ok(ctx.apply_calls))
            .expect("typed immutable access should succeed");
        assert_eq!(calls, 3);

        pc.with_typed_context::<HookCtx, _, _>(|ctx| {
            ctx.apply_calls += 1;
            Ok(())
        })
        .expect("typed mutable access should succeed");

        let calls = pc
            .with_typed_context_ref::<HookCtx, _, _>(|ctx| Ok(ctx.apply_calls))
            .expect("typed immutable access should succeed");
        assert_eq!(calls, 4);
    }

    #[test]
    fn shared_context_helpers_expose_arc_state() {
        let tag = "shell_shared_ctx";
        let shared = Arc::new(RwLock::new(HookCtx::default()));
        register_shell_context_shared_rwlock(format!("{tag}_ctx"), shared.clone());
        let mut pc = ShellPc::new(
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            Some(format!("{tag}_ctx")),
        );
        pc.setup(&TestOp).expect("setup should succeed");
        pc.with_shared_context_rwlock::<HookCtx, _, _>(|ctx| {
            let mut guard = ctx.write().expect("rwlock poisoned");
            guard.apply_calls += 2;
            Ok(())
        })
        .expect("shared rwlock context access should succeed");
        assert_eq!(shared.read().expect("rwlock poisoned").apply_calls, 2);
    }

    #[test]
    fn typed_context_helpers_reject_wrong_type() {
        let mut ctx: Box<dyn ShellContext> = Box::new(7usize);
        let err = match shell_context_downcast_mut::<HookCtx>(ctx.as_mut()) {
            Ok(_) => panic!("expected downcast error"),
            Err(err) => err,
        };
        assert!(format!("{err}").contains("HookCtx"));
    }

    #[test]
    fn symmetric_left_right_callbacks_chain_in_order() {
        #[derive(Default)]
        struct Ctx {
            trace: Vec<&'static str>,
        }
        let tag = "shell_symmetric_lr";
        register_shell_context_typed(format!("{tag}_ctx"), Ctx::default);
        register_shell_apply_symmetric_left_typed(
            format!("{tag}_left"),
            |_side, x, y, ctx: &mut Ctx| {
                ctx.trace.push("left");
                for (yi, xi) in y.iter_mut().zip(x.iter()) {
                    *yi = *xi * 2.0;
                }
                Ok(())
            },
        );
        register_shell_apply_symmetric_right_typed(
            format!("{tag}_right"),
            |_side, x, y, ctx: &mut Ctx| {
                ctx.trace.push("right");
                for (yi, xi) in y.iter_mut().zip(x.iter()) {
                    *yi = *xi + 1.0;
                }
                Ok(())
            },
        );

        let mut pc = ShellPc::new(
            None,
            None,
            None,
            None,
            Some(format!("{tag}_left")),
            Some(format!("{tag}_right")),
            None,
            None,
            Some(format!("{tag}_ctx")),
        );
        pc.setup(&TestOp).expect("setup should succeed");
        let mut y = vec![0.0, 0.0];
        pc.apply(PcSide::Symmetric, &[1.0, 2.0], &mut y)
            .expect("symmetric split apply should succeed");
        assert_eq!(y, vec![3.0, 5.0]);
        pc.with_typed_context_ref::<Ctx, _, _>(|ctx| {
            assert_eq!(ctx.trace, vec!["left", "right"]);
            Ok(())
        })
        .expect("context read should succeed");
    }

    #[test]
    fn shell_error_preserves_nested_failure_diagnostics() {
        let inner = KError::NestedPcFailed(NestedPcFailure {
            component: "pc_ksp",
            reason: ConvergedReason::DivergedPcFailed,
            iterations: 4,
            final_norm: None,
            residual_history_summary: None,
            detail: "inner exploded".into(),
        });
        let wrapped = ShellPc::shell_error(FailureStage::Solve, "apply", inner);
        let KError::NestedPcFailed(failure) = wrapped else {
            panic!("expected nested failure");
        };
        assert_eq!(failure.reason, ConvergedReason::DivergedPcFailed);
        assert!(failure.detail.contains("inner_component=pc_ksp"));
        assert!(failure.detail.contains("inner_detail=inner exploded"));
    }

    #[test]
    fn transpose_and_symmetric_fallback_to_apply_when_unregistered() {
        let mut pc = ShellPc::new(None, None, None, None, None, None, None, None, None);
        pc.setup(&TestOp).expect("setup should succeed");

        let x = vec![1.0, 2.0];
        let mut y = vec![0.0, 0.0];
        pc.apply_op(Op::Trans, &x, &mut y)
            .expect("transpose fallback should succeed");
        assert_eq!(y, x);
        pc.apply_op(Op::ConjTrans, &x, &mut y)
            .expect("conjugate-transpose fallback should succeed");
        assert_eq!(y, x);
        pc.apply(PcSide::Symmetric, &x, &mut y)
            .expect("symmetric fallback should succeed");
        assert_eq!(y, x);
    }

    #[test]
    fn ksp_context_reports_shell_symmetric_hook_failures() {
        let tag = "shell_sym_failure";
        register_shell_apply_typed(format!("{tag}_apply"), |_side, x, y, _ctx: &mut ()| {
            y.copy_from_slice(x);
            Ok(())
        });
        register_shell_apply_symmetric_typed(
            format!("{tag}_sym"),
            |_side, _x, _y, _ctx: &mut ()| Err(KError::SolveError("sym boom".into())),
        );

        let mut ksp = KspContext::new();
        ksp.set_type(SolverType::Richardson)
            .expect("solver selection should succeed");
        let ksp_opts = KspOptions {
            maxits: Some(2),
            rtol: Some(1e-12),
            pc_side: Some("symmetric".into()),
            ..Default::default()
        };
        let pc_opts = PcOptions {
            pc_type: Some("shell".into()),
            pc_shell_apply: Some(format!("{tag}_apply")),
            pc_shell_apply_symmetric: Some(format!("{tag}_sym")),
            ..Default::default()
        };
        ksp.set_from_all_options(&ksp_opts, &pc_opts)
            .expect("options should apply");
        ksp.set_operators(Arc::new(TestOp), None);

        let b = vec![1.0, 2.0];
        let mut x = vec![0.0, 0.0];
        let stats = ksp.solve(&b, &mut x).expect("solve should return stats");
        assert_eq!(stats.reason, ConvergedReason::DivergedPcFailed);
        let failure = stats
            .nested_pc_failure
            .as_ref()
            .expect("nested failure metadata should be present");
        assert_eq!(failure.component, "pc_shell");
        assert!(failure.detail.contains("hook=apply_symmetric"));
    }
}
