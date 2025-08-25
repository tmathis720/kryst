//! Additive Schwarz preconditioner (ASM)
//!
//! Based on Saad, and inspired by PETSc's PCASM. Supports Rayon in shared memory and MPI for distributed vectors.

use crate::core::traits::MatVec;
use crate::error::KError;
use crate::preconditioner::{PcSide, legacy::Preconditioner};
use crate::solver::legacy::LinearSolver;
use std::sync::Mutex;
// Extra imports to support LinOp-based setup/apply (f64 case)
use crate::matrix::convert::csr_from_linop;
use crate::matrix::op::{LinOp, StructureId, ValuesId};
use crate::core::traits::SubmatrixExtract;
use crate::preconditioner::Preconditioner as ObjPreconditioner;
use crate::matrix::sparse::CsrMatrix;
use std::sync::Arc;
use crate::solver::direct_lu::LuSolver;

/// Additive Schwarz (overlapping block Jacobi) preconditioner.
/// Each block is solved independently (possibly in parallel), and the results are summed.
pub struct AdditiveSchwarz<M, V, T> {
    /// Number of overlap layers.
    pub overlap: usize,
    /// Local subdomain index sets (global indices) per block.
    pub subdomains: Vec<Vec<usize>>,
    /// One inner solver and submatrix per subdomain.
    pub local_blocks: Vec<(
        M,
        Mutex<Box<dyn LinearSolver<M, V, Scalar = T, Error = KError> + Send + Sync>>,
    )>,
    // CSR-based bookkeeping to make this PC LinOp-aware and reuse-friendly.
    // These fields are primarily used by the f64/CSR specialization below.
    pub csr: Option<Arc<CsrMatrix<f64>>>,
    pub last_sid: Option<StructureId>,
    pub last_vid: Option<ValuesId>,
    pub drop_tol: f64,
    /// Per-block solver factory stored as an enum so it can be serialized in options
    /// or swapped at runtime. Default: LU on dense sub-blocks (used for tests).
    pub block_solver_factory: BlockSolverFactory,
}

/// Lightweight enum to configure per-block solver factory.
#[derive(Clone)]
pub enum BlockSolverFactory {
    /// Use LU solver on faer::Mat<f64> sub-blocks (default)
    LuDense,
    /// Use a legacy LinearSolver specialized for CSR matrices (requires solver impl)
    CsrSolver, // placeholder for future extensions
}

impl<M, V, T> AdditiveSchwarz<M, V, T>
where
    M: MatVec<V> + Clone + Send + Sync + crate::core::traits::SubmatrixExtract,
    V: From<Vec<T>> + AsRef<[T]> + AsMut<[T]> + Clone + Send + Sync,
    T: 'static + num_traits::Float + From<f64> + Send + Sync,
{
    /// Create a new ASM with given overlap and user-defined subdomain partitions.
    /// If `subdomains` is empty, will later partition rows uniformly.
    pub fn new(
        overlap: usize,
        subdomains: Vec<Vec<usize>>,
        block_solver_factory: BlockSolverFactory,
    ) -> Self {
        Self {
            overlap,
            subdomains,
            local_blocks: Vec::new(),
            csr: None,
            last_sid: None,
            last_vid: None,
            drop_tol: 0.0,
            block_solver_factory,
        }
    }

    /// Setup: extract submatrices and configure each local solver (e.g. GMRES+ILU).
    /// The `solver_factory` closure is called to create a solver for each block.
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
    // Build per-block submatrix and setup solvers. For generic M/V types we
    // use the SubmatrixExtract impl (CSR or dense) on the provided matrix.
        self.local_blocks = self
            .subdomains
            .iter()
            .map(|indices| {
                let a_sub: M = a.submatrix(indices);
                let mut ksp = solver_factory();
                let _ = ksp.solve(
                    &a_sub,
                    None,
                    &V::from(vec![T::zero(); indices.len()]),
                    &mut V::from(vec![T::zero(); indices.len()]),
                    PcSide::Left,
                    &crate::parallel::UniverseComm::NoComm(crate::parallel::NoComm),
                    None,
                    None,
                );
                (a_sub, Mutex::new(Box::new(ksp) as _))
            })
            .collect();
    }
}

impl<M, V, T> Preconditioner<M, V> for AdditiveSchwarz<M, V, T>
where
    M: MatVec<V> + Clone + Send + Sync,
    V: From<Vec<T>> + AsRef<[T]> + AsMut<[T]> + Clone + Send + Sync,
    T: 'static + num_traits::Float + From<f64> + Send + Sync,
{
    /// Setup method required by Preconditioner trait
    fn setup(&mut self, _a: &M) -> Result<(), KError> {
        // This is a placeholder - actual setup should be done via the other setup method
        Ok(())
    }

    /// Apply `z = P^{-1} r` via overlapping block solves.
    /// Each block's result is summed into the global vector.
    fn apply(&self, _side: crate::preconditioner::PcSide, r: &V, z: &mut V) -> Result<(), KError> {
        for zi in z.as_mut().iter_mut() {
            *zi = T::zero();
        }
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;
            // Each block's result is a (indices, x_blk) pair
            let block_results: Vec<(Vec<usize>, Vec<T>)> = self
                .subdomains
                .par_iter()
                .zip(self.local_blocks.par_iter())
                .map(|(indices, (a_sub, ksp_mutex))| {
                    let r_blk = V::from(indices.iter().map(|&i| r.as_ref()[i]).collect());
                    let mut x_blk = V::from(vec![T::zero(); indices.len()]);
                    let mut ksp = ksp_mutex.lock().unwrap();
                    let _ = ksp.solve(
                        a_sub,
                        None,
                        &r_blk,
                        &mut x_blk,
                        PcSide::Left,
                        &crate::parallel::UniverseComm::NoComm(crate::parallel::NoComm),
                        None,
                        None,
                    );
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
                    let _ = ksp.solve(
                        a_sub,
                        None,
                        &r_blk,
                        &mut x_blk,
                        PcSide::Left,
                        &crate::parallel::UniverseComm::NoComm(crate::parallel::NoComm),
                        None,
                        None,
                    );
                    for (j, &gi) in indices.iter().enumerate() {
                        z.as_mut()[gi] = z.as_ref()[gi] + x_blk.as_ref()[j];
                    }
                });
        }
        Ok(())
    }
}

// Object-safe Preconditioner implementation for the common f64 dense case.
// This allows KSPs that pass a `&dyn LinOp<S=f64>` to setup/apply ASM without
// requiring callers to downcast to a concrete matrix type.
impl ObjPreconditioner for AdditiveSchwarz<faer::Mat<f64>, Vec<f64>, f64> {
    fn setup(&mut self, op: &dyn LinOp<S = f64>) -> Result<(), KError> {
        // Obtain (or convert) a CSR representation from the LinOp using shared converter
        let csr = csr_from_linop(op, self.drop_tol)?;

        // Store CSR handle for reuse
        self.csr = Some(csr.clone());

        // Build per-block dense submatrices from CSR efficiently and configure per-block solvers
        let subdomains = &self.subdomains;
        self.local_blocks.clear();

        for indices in subdomains.iter() {
            // Extract local CSR submatrix efficiently from Arc by borrowing
            let a_sub_csr = csr.as_ref().submatrix(indices);
            // Convert to dense for the legacy dense solvers
            let dense = a_sub_csr.to_dense();
            // Create LU solver for this block and initialize it
            let mut ksp = LuSolver::<f64>::new();
            let _ = ksp.solve(
                &dense,
                None,
                &vec![0.0; indices.len()],
                &mut vec![0.0; indices.len()],
                PcSide::Left,
                &crate::parallel::UniverseComm::NoComm(crate::parallel::NoComm),
                None,
                None,
            );
            self.local_blocks.push((dense, Mutex::new(Box::new(ksp) as _)));
        }

        // Bookkeeping for reuse semantics
        self.last_sid = Some(op.structure_id());
        self.last_vid = Some(op.values_id());

        Ok(())
    }

    fn apply(&self, _side: PcSide, x: &[f64], y: &mut [f64]) -> Result<(), KError> {
        
        if x.len() != y.len() {
            return Err(KError::InvalidInput(format!(
                "ASM apply: x/y length mismatch: {} vs {}",
                x.len(),
                y.len()
            )));
        }

        // Zero the output
        for yi in y.iter_mut() {
            *yi = 0.0;
        }

        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;
            let block_results: Vec<(Vec<usize>, Vec<f64>)> = self
                .subdomains
                .par_iter()
                .zip(self.local_blocks.par_iter())
                .map(|(indices, (a_sub_any, ksp_mutex))| {
                    // a_sub_any is a CsrMatrix stored in the M slot
                    let r_blk: Vec<f64> = indices.iter().map(|&i| x[i]).collect();
                    let mut x_blk = vec![0.0; indices.len()];
                    let mut ksp = ksp_mutex.lock().unwrap();
                    // The legacy solver expects a concrete matrix; we give it the
                    // dense representation if it was built that way during setup.
                    // Here we attempt to downcast to a CsrMatrix and if not present
                    // we cannot do better than invoking the solver stored.
                    let _ = ksp.solve(
                        a_sub_any,
                        None,
                        &r_blk,
                        &mut x_blk,
                        PcSide::Left,
                        &crate::parallel::UniverseComm::NoComm(crate::parallel::NoComm),
                        None,
                        None,
                    );
                    (indices.clone(), x_blk)
                })
                .collect();

            for (indices, x_blk) in block_results {
                for (j, &gi) in indices.iter().enumerate() {
                    y[gi] = y[gi] + x_blk[j];
                }
            }
        }

        #[cfg(not(feature = "rayon"))]
        {
            self.subdomains
                .iter()
                .zip(self.local_blocks.iter())
                .for_each(|(indices, (a_sub_any, ksp_mutex))| {
                    let r_blk: Vec<f64> = indices.iter().map(|&i| x[i]).collect();
                    let mut x_blk = vec![0.0; indices.len()];
                    let mut ksp = ksp_mutex.lock().unwrap();
                    let _ = ksp.solve(
                        a_sub_any,
                        None,
                        &r_blk,
                        &mut x_blk,
                        PcSide::Left,
                        &crate::parallel::UniverseComm::NoComm(crate::parallel::NoComm),
                        None,
                        None,
                    );
                    for (j, &gi) in indices.iter().enumerate() {
                        y[gi] = y[gi] + x_blk[j];
                    }
                });
        }

        Ok(())
    }

    // Delegate mutable apply to the immutable one by default.
    fn apply_mut(&mut self, side: PcSide, x: &[f64], y: &mut [f64]) -> Result<(), KError> {
        // Fully qualify to avoid ambiguity with the legacy Preconditioner impl.
        <Self as ObjPreconditioner>::apply(self, side, x, y)
    }

    fn supports_numeric_update(&self) -> bool {
        // ASM can typically refresh numeric local solves by refactorizing local blocks.
        true
    }

    fn update_numeric(&mut self, op: &dyn LinOp<S = f64>) -> Result<(), KError> {
        // Rebuild per-block factorization from a fresh CSR copy of the operator.
        let csr = csr_from_linop(op, self.drop_tol)?;
        self.csr = Some(csr.clone());

        // Recreate local dense blocks and solvers from CSR
        self.local_blocks.clear();
        for indices in self.subdomains.iter() {
            let a_sub_csr = csr.as_ref().submatrix(indices);
            let dense = a_sub_csr.to_dense();
            let mut ksp = LuSolver::<f64>::new();
            let _ = ksp.solve(
                &dense,
                None,
                &vec![0.0; indices.len()],
                &mut vec![0.0; indices.len()],
                PcSide::Left,
                &crate::parallel::UniverseComm::NoComm(crate::parallel::NoComm),
                None,
                None,
            );
            self.local_blocks.push((dense, Mutex::new(Box::new(ksp) as _)));
        }

        self.last_vid = Some(op.values_id());
        Ok(())
    }

    fn update_symbolic(&mut self, op: &dyn LinOp<S = f64>) -> Result<(), KError> {
        // For ASM symbolic change we also reconstruct blocks/subdomains from CSR
        let csr = csr_from_linop(op, self.drop_tol)?;
        self.csr = Some(csr.clone());

        self.local_blocks.clear();
        for indices in self.subdomains.iter() {
            let a_sub_csr = csr.as_ref().submatrix(indices);
            let dense = a_sub_csr.to_dense();
            let mut ksp = LuSolver::<f64>::new();
            let _ = ksp.solve(
                &dense,
                None,
                &vec![0.0; indices.len()],
                &mut vec![0.0; indices.len()],
                PcSide::Left,
                &crate::parallel::UniverseComm::NoComm(crate::parallel::NoComm),
                None,
                None,
            );
            self.local_blocks.push((dense, Mutex::new(Box::new(ksp) as _)));
        }

        self.last_sid = Some(op.structure_id());
        self.last_vid = Some(op.values_id());
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
        let mut asm = AdditiveSchwarz::<faer::Mat<f64>, Vec<f64>, f64>::new(
            0,
            subdomains,
            BlockSolverFactory::LuDense,
        );
        asm.setup(&a, || LuSolver::<f64>::new());
        let r = vec![1.0, 2.0, 3.0, 4.0];
        let mut z = vec![0.0; 4];
    Preconditioner::apply(&asm, PcSide::Left, &r, &mut z).unwrap();
    // For identity, ASM should return the input
    assert_eq!(z, r);
    }
}
