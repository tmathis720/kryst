// Jacobi preconditioner implementation

use crate::preconditioner::Preconditioner;
use crate::core::traits::{MatVec, Indexing};
use crate::error::KError;
use num_traits::Float;

/// Jacobi preconditioner: M⁻¹ = D⁻¹
pub struct Jacobi<T> {
    pub(crate) inv_diag: Vec<T>,
}

impl<T: Float> Jacobi<T> {
    /// new with empty state; user must call `setup`.
    pub fn new() -> Self {
        Self { inv_diag: Vec::new() }
    }
}

impl<T: num_traits::Float> Default for Jacobi<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<M, V, T> Preconditioner<M, V> for Jacobi<T>
where
    M: MatVec<V> + Indexing,
    V: AsRef<[T]> + AsMut<[T]> + From<Vec<T>>,
    T: Float + Send + Sync,
{
    fn setup(&mut self, a: &M) -> Result<(), KError> {
        let n = a.nrows();
        let mut diag = vec![T::zero(); n];
        let mut e = vec![T::zero(); n];
        let mut col = vec![T::zero(); n];
        for i in 0..n {
            e.iter_mut().for_each(|x| *x = T::zero());
            e[i] = T::one();
            let e_v = V::from(e.clone());
            let mut col_v = V::from(col.clone());
            a.matvec(&e_v, &mut col_v);
            col = col_v.as_ref().to_vec();
            diag[i] = col[i];
        }
        self.inv_diag = diag.into_iter()
            .map(|d| if d != T::zero() { T::one() / d } else { T::zero() })
            .collect();
        Ok(())
    }

    fn apply(&self, x: &V, y: &mut V) -> Result<(), KError> {
        let x_ref = x.as_ref();
        let y_mut = y.as_mut();
        for i in 0..x_ref.len() {
            y_mut[i] = self.inv_diag[i] * x_ref[i];
        }
        Ok(())
    }
}
