//! Additive Schwarz preconditioner (ASM)
//!
//! Based on Saad, and inspired by PETSc's PCASM. Supports Rayon in shared memory and MPI for distributed vectors.

use crate::core::traits::MatVec;
use crate::solver::LinearSolver;
use crate::preconditioner::Preconditioner;
use crate::error::KError;
use std::sync::Mutex;
#[cfg(feature = "rayon")]
use rayon::prelude::*;
#[cfg(feature = "rayon")]
use rayon::iter::IntoParallelRefIterator;

/// Additive Schwarz (overlapping block Jacobi) preconditioner
pub struct AdditiveSchwarz<M, V, T> {
    /// Number of overlap layers
    pub overlap: usize,
    /// Local subdomain index sets (global indices) per block
    pub subdomains: Vec<Vec<usize>>,
    /// One inner solver and submatrix per subdomain
    pub local_blocks: Vec<(M, Mutex<Box<dyn LinearSolver<M, V, Scalar = T, Error = KError> + Send + Sync>>)>,
}

impl<M, V, T> AdditiveSchwarz<M, V, T>
where
    M: MatVec<V> + Clone + Send + Sync + crate::core::traits::SubmatrixExtract,
    V: From<Vec<T>> + AsRef<[T]> + AsMut<[T]> + Clone + Send + Sync,
    T: 'static + num_traits::Float + From<f64> + Send + Sync,
{
    /// Create a new ASM with given overlap and user-defined subdomain partitions.
    /// If `subdomains` is empty, will later partition rows uniformly.
    pub fn new(overlap: usize, subdomains: Vec<Vec<usize>>) -> Self {
        Self { overlap, subdomains, local_blocks: Vec::new() }
    }

    /// Setup: extract submatrices and configure each local solver (e.g. GMRES+ILU)
    pub fn setup<S>(&mut self, a: &M, mut solver_factory: impl FnMut() -> S)
    where
        S: LinearSolver<M, V, Scalar = T, Error = KError> + Send + Sync + 'static,
        M: crate::core::traits::MatShape + Clone + crate::core::traits::SubmatrixExtract,
    {
        // If no explicit subdomains, partition uniformly by row
        if self.subdomains.is_empty() {
            let n = a.nrows();
            let p = self.subdomains.capacity().max(1);
            let chunk = (n + p - 1) / p;
            self.subdomains = (0..p)
                .map(|i| {
                    let start = i * chunk;
                    let end = ((i + 1) * chunk).min(n);
                    (start..end).collect()
                })
                .collect();
        }
        // Build per-block submatrix and setup solvers
        self.local_blocks = self.subdomains.iter().map(|indices| {
            let a_sub: M = a.submatrix(indices);
            let mut ksp = solver_factory();
            let _ = ksp.solve(&a_sub, None, &V::from(vec![T::zero(); indices.len()]), &mut V::from(vec![T::zero(); indices.len()]));
            (a_sub, Mutex::new(Box::new(ksp) as _))
        }).collect();
    }
}

impl<M, V, T> Preconditioner<M, V> for AdditiveSchwarz<M, V, T>
where
    M: MatVec<V> + Clone + Send + Sync,
    V: From<Vec<T>> + AsRef<[T]> + AsMut<[T]> + Clone + Send + Sync,
    T: 'static + num_traits::Float + From<f64> + Send + Sync,
{
    /// Apply `z = P^{-1} r` via overlapping block solves.
    fn apply(&self, r: &V, z: &mut V) -> Result<(), KError> {
        for zi in z.as_mut().iter_mut() { *zi = T::zero(); }
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;
            // Each block's result is a (indices, x_blk) pair
            let block_results: Vec<(Vec<usize>, Vec<T>)> = self.subdomains
                .par_iter()
                .zip(self.local_blocks.par_iter())
                .map(|(indices, (a_sub, ksp_mutex))| {
                    let r_blk = V::from(indices.iter().map(|&i| r.as_ref()[i]).collect());
                    let mut x_blk = V::from(vec![T::zero(); indices.len()]);
                    let mut ksp = ksp_mutex.lock().unwrap();
                    let _ = ksp.solve(a_sub, None, &r_blk, &mut x_blk);
                    (indices.clone(), x_blk.as_ref().to_vec())
                })
                .collect();
            // Serial reduction: sum all block results into z
            for (indices, x_blk) in block_results {
                for (j, &gi) in indices.iter().enumerate() {
                    z.as_mut()[gi] = z.as_ref()[gi] + x_blk[j];
                }
            }
        }
        #[cfg(not(feature = "rayon"))]
        {
            self.subdomains
                .iter()
                .zip(self.local_blocks.iter())
                .for_each(|(indices, (a_sub, ksp_mutex))| {
                    let r_blk = V::from(indices.iter().map(|&i| r.as_ref()[i]).collect());
                    let mut x_blk = V::from(vec![T::zero(); indices.len()]);
                    let mut ksp = ksp_mutex.lock().unwrap();
                    let _ = ksp.solve(a_sub, None, &r_blk, &mut x_blk);
                    for (j, &gi) in indices.iter().enumerate() {
                        z.as_mut()[gi] = z.as_ref()[gi] + x_blk.as_ref()[j];
                    }
                });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::direct_lu::LuSolver;

    #[test]
    fn asm_dense_lu_blocks() {
        // 4x4 identity matrix
        let a = faer::Mat::<f64>::from_fn(4, 4, |i, j| if i == j { 1.0 } else { 0.0 });
        let subdomains = vec![vec![0, 1], vec![2, 3]];
        let mut asm = AdditiveSchwarz::<faer::Mat<f64>, Vec<f64>, f64>::new(0, subdomains);
        asm.setup(&a, || LuSolver::<f64>::new());
        let r = vec![1.0, 2.0, 3.0, 4.0];
        let mut z = vec![0.0; 4];
        asm.apply(&r, &mut z).unwrap();
        // For identity, ASM should return the input
        assert_eq!(z, r);
    }
}
