use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use super::op::{LinOp, StructureId, ValuesId};

/// Matrix-free "shell" operator.
pub struct MatShell<S> {
    m: usize,
    n: usize,
    mv: Arc<dyn Fn(&[S], &mut [S]) + Send + Sync>,
    mvt: Option<Arc<dyn Fn(&[S], &mut [S]) + Send + Sync>>,
    sid: AtomicU64,
    vid: AtomicU64,
}

impl MatShell<f64> {
    pub fn new(
        m: usize,
        n: usize,
        mv: impl Fn(&[f64], &mut [f64]) + Send + Sync + 'static,
    ) -> Self {
        Self {
            m,
            n,
            mv: Arc::new(mv),
            mvt: None,
            sid: AtomicU64::new(1),
            vid: AtomicU64::new(1),
        }
    }

    pub fn with_transpose(
        mut self,
        mvt: impl Fn(&[f64], &mut [f64]) + Send + Sync + 'static,
    ) -> Self {
        self.mvt = Some(Arc::new(mvt));
        self
    }

    pub fn bump_values(&self) {
        self.vid.fetch_add(1, Ordering::Relaxed);
    }

    pub fn bump_structure(&self) {
        self.sid.fetch_add(1, Ordering::Relaxed);
    }
}

impl LinOp for MatShell<f64> {
    type S = f64;

    fn dims(&self) -> (usize, usize) {
        (self.m, self.n)
    }

    fn matvec(&self, x: &[f64], y: &mut [f64]) {
        (self.mv)(x, y)
    }

    fn supports_transpose(&self) -> bool {
        self.mvt.is_some()
    }

    fn t_matvec(&self, x: &[f64], y: &mut [f64]) {
        if let Some(f) = &self.mvt {
            f(x, y);
        } else {
            panic!("LinOp::t_matvec called but supports_transpose() == false");
        }
    }

    fn structure_id(&self) -> StructureId {
        StructureId(self.sid.load(Ordering::Relaxed))
    }
    fn values_id(&self) -> ValuesId {
        ValuesId(self.vid.load(Ordering::Relaxed))
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
