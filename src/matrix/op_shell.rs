use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use super::op::{LinOp, LinOpF64, StructureId, ValuesId};
use crate::algebra::prelude::KrystScalar;
use crate::error::KError;

/// Matrix-free "shell" operator.
pub struct MatShell<S: KrystScalar> {
    m: usize,
    n: usize,
    mv: Arc<dyn Fn(&[S], &mut [S]) + Send + Sync>,
    mvt: Option<Arc<dyn Fn(&[S], &mut [S]) + Send + Sync>>,
    sid: AtomicU64,
    vid: AtomicU64,
}

impl<S: KrystScalar> MatShell<S> {
    pub fn new(
        m: usize,
        n: usize,
        mv: impl Fn(&[S], &mut [S]) + Send + Sync + 'static,
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
        mvt: impl Fn(&[S], &mut [S]) + Send + Sync + 'static,
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

impl<S: KrystScalar> LinOp for MatShell<S> {
    type S = S;

    fn dims(&self) -> (usize, usize) {
        (self.m, self.n)
    }

    fn matvec(&self, x: &[S], y: &mut [S]) {
        debug_assert_eq!(x.len(), self.n, "MatShell::matvec x.len mismatch");
        debug_assert_eq!(y.len(), self.m, "MatShell::matvec y.len mismatch");
        (self.mv)(x, y)
    }

    fn try_matvec(&self, x: &[S], y: &mut [S]) -> Result<(), KError> {
        if x.len() != self.n || y.len() != self.m {
            return Err(KError::InvalidInput(format!(
                "MatShell::matvec dimension mismatch: A={}x{}, x.len()={}, y.len()={}",
                self.m,
                self.n,
                x.len(),
                y.len()
            )));
        }
        (self.mv)(x, y);
        Ok(())
    }

    fn supports_transpose(&self) -> bool {
        self.mvt.is_some()
    }

    fn t_matvec(&self, x: &[S], y: &mut [S]) {
        debug_assert_eq!(x.len(), self.m, "MatShell::t_matvec x.len mismatch");
        debug_assert_eq!(y.len(), self.n, "MatShell::t_matvec y.len mismatch");
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

impl LinOpF64 for MatShell<f64> {
    #[inline]
    fn dims(&self) -> (usize, usize) {
        LinOp::dims(self)
    }

    #[inline]
    fn matvec(&self, x: &[f64], y: &mut [f64]) {
        LinOp::matvec(self, x, y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mat_shell_try_matvec_reports_dim_mismatch() {
        let op = MatShell::<f64>::new(2, 3, |x, y| {
            for (yi, xi) in y.iter_mut().zip(x.iter()) {
                *yi = *xi;
            }
        });
        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![0.0; 1];
        let err = op.try_matvec(&x, &mut y).unwrap_err();
        assert!(matches!(err, KError::InvalidInput(_)));
    }
}
