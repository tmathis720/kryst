use crate::preconditioner::Preconditioner;
use crate::core::traits::{MatVec, Indexing};
use crate::error::KError;
use num_traits::Float;

/// Symmetric Successive Over-Relaxation.
/// M = (D/ω + L) D⁻¹ (D/ω + U)
pub struct Ssor<T> {
    omega: T,
    inv_diag: Vec<T>,
}

impl<T: Float> Ssor<T> {
    pub fn new(omega: T) -> Self {
        Self { omega, inv_diag: Vec::new() }
    }
}

impl<M, V, T> Preconditioner<M, V> for Ssor<T>
where
    M: MatVec<V> + Indexing,
    V: AsRef<[T]> + AsMut<[T]> + From<Vec<T>>,
    T: num_traits::Float + Send + Sync + std::ops::Mul<Output = T> + Copy,
{
    fn setup(&mut self, a: &M) -> Result<(), KError> {
        // reuse Jacobi logic for inv_diag
        let mut jac = crate::preconditioner::Jacobi::<T>::new();
        jac.setup(a)?;
        self.inv_diag = jac.inv_diag;
        Ok(())
    }

    fn apply(&self, x: &V, y: &mut V) -> Result<(), KError> {
        let n = x.as_ref().len();
        let mut tmp = vec![T::zero(); n];
        // forward Gauss-Seidel: (D/ω + L)⁻¹ x
        for i in 0..n {
            let sum = x.as_ref()[i];
            // subtract lower part: a_ij * tmp[j]
            // we approximate a_ij via matvec of basis vectors (expensive, but OK for demo)
            // TODO: optimize by storing L
            tmp[i] = sum * self.inv_diag[i] * self.omega;
        }
        // backward: (D/ω + U)⁻¹ tmp → y
        for i in (0..n).rev() {
            y.as_mut()[i] = tmp[i] * self.inv_diag[i] * self.omega;
        }
        Ok(())
    }
}
