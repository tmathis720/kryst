//! Additive Schwarz preconditioner (ASM)
//!
//! Based on Saad, and inspired by PETSc's PCASM. Supports Rayon in shared memory and MPI for distributed vectors.

#[allow(unused_imports)]
use crate::algebra::blas::{dot_conj, nrm2};
#[allow(unused_imports)]
use crate::algebra::prelude::*;
use crate::core::traits::MatVec;
use crate::error::KError;
use crate::preconditioner::{PcSide, legacy::Preconditioner as LegacyPreconditioner};
use crate::solver::legacy::LinearSolver;
use std::sync::Mutex;
// Extra imports to support LinOp-based setup/apply (f64 case)
use crate::core::traits::SubmatrixExtract;
use crate::matrix::convert::csr_from_linop;
use crate::matrix::op::CsrOp;
use crate::matrix::op::{LinOp, StructureId, ValuesId};
use crate::matrix::sparse::CsrMatrix;
use crate::preconditioner::PcCaps;
use crate::preconditioner::Preconditioner as DynPreconditioner;
use crate::preconditioner::Preconditioner as ObjPreconditioner;
use crate::preconditioner::ilu_csr::{
    IluCsr, IluCsrConfig, IluKind, PivotStrategy, ReorderingOptions,
};
#[cfg(feature = "dense-direct")]
use crate::solver::direct_lu::LuSolver;
use crate::utils::partition::{contiguous_partition, greedy_nnz_balanced_partition};
use std::sync::Arc;

#[cfg(feature = "complex")]
use crate::algebra::bridge::BridgeScratch;
#[cfg(feature = "complex")]
use crate::preconditioner::pc_bridge::{apply_pc_mut_s, apply_pc_s};

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
    /// ASM variant: ASM (sum) or RAS (interior injection)
    pub asm_mode: AsmMode,
    /// Weighting strategy for overlaps.
    pub weighting: Weighting,
    /// Optional hint: number of primary partitions to build when `subdomains` empty.
    pub nparts_hint: Option<usize>,
    /// Dense threshold heuristic for per-block solver selection (currently not switching paths).
    pub dense_threshold: usize,
    /// Computed: owner subdomain id for each global row.
    pub owner_of: Vec<usize>,
    /// Computed: coverage count for each global row (how many blocks include it).
    pub cover_count: Vec<usize>,
    /// Per-block metadata aligned with `subdomains`/`local_blocks`.
    pub blocks_meta: Vec<SubdomainMeta>,
    /// CSR + ILU(0) blocks for the f64 specialization (used when block solver = CSR path).
    pub local_blocks_csr: Vec<(Arc<CsrMatrix<f64>>, std::sync::Arc<IluCsr>)>,
}

/// Lightweight enum to configure per-block solver factory.
#[derive(Clone)]
pub enum BlockSolverFactory {
    /// Use LU solver on faer::Mat<f64> sub-blocks (default)
    LuDense,
    /// Use a legacy LinearSolver specialized for CSR matrices (requires solver impl)
    CsrSolver, // placeholder for future extensions
}

/// ASM mode selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AsmMode {
    /// Classical ASM: restrict to overlapped blocks and sum (optionally weighted) results.
    ASM,
    /// Restricted ASM: inject only on primary-owner (interior) DOFs.
    RAS,
}

/// Partition-of-unity weighting variants for overlaps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Weighting {
    /// No weighting; plain sum. Safe for 0-overlap or RAS with interior-only.
    None,
    /// Uniform PoU: each covered node gets weight = 1 / cover_count.
    Uniform,
    /// Smooth linear: phi = dist_to_boundary + 1.
    SmoothLinear,
    /// Smooth polynomial: phi = (dist_to_boundary + 1)^p, p >= 2
    SmoothPoly(u32),
}

/// Per-subdomain metadata used during apply.
#[derive(Clone, Default)]
pub struct SubdomainMeta {
    /// Overlapped index set (sorted).
    pub indices: Vec<usize>,
    /// Marks primary-owner DOFs inside the overlapped set.
    pub interior_mask: Vec<bool>,
    /// Weights aligned with `indices` (PoU weights for ASM; masked in RAS if desired).
    pub weights: Vec<R>,
}

impl<M, V, T> AdditiveSchwarz<M, V, T>
where
    M: MatVec<V> + Clone + Send + Sync + crate::core::traits::SubmatrixExtract<S = T>,
    V: From<Vec<T>> + AsRef<[T]> + AsMut<[T]> + Clone + Send + Sync,
    T: 'static + KrystScalar<Real = R> + From<f64> + PartialOrd + Send + Sync,
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
            asm_mode: AsmMode::ASM,
            weighting: if overlap == 0 {
                Weighting::None
            } else {
                Weighting::Uniform
            },
            nparts_hint: None,
            dense_threshold: 192,
            owner_of: Vec::new(),
            cover_count: Vec::new(),
            blocks_meta: Vec::new(),
            local_blocks_csr: Vec::new(),
        }
    }

    /// Setup: extract submatrices and configure each local solver (e.g. GMRES+ILU).
    /// The `solver_factory` closure is called to create a solver for each block.
    pub fn setup<S>(&mut self, a: &M, mut solver_factory: impl FnMut() -> S)
    where
        S: LinearSolver<M, V, Scalar = T, Error = KError> + Send + Sync + 'static,
        M: crate::core::traits::MatShape + Clone + crate::core::traits::SubmatrixExtract<S = T>,
    {
        // If no explicit subdomains, partition uniformly by row
        if self.subdomains.is_empty() {
            let n = a.nrows();
            let p = self.subdomains.capacity().max(1);
            let chunk = n.div_ceil(p);
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

impl<M, V, T> LegacyPreconditioner<M, V> for AdditiveSchwarz<M, V, T>
where
    M: MatVec<V> + Clone + Send + Sync,
    V: From<Vec<T>> + AsRef<[T]> + AsMut<[T]> + Clone + Send + Sync,
    T: 'static + KrystScalar<Real = R> + From<f64> + PartialOrd + Send + Sync,
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

        // 1) Build/validate primary partition (owner_of)
        let n = csr.nrows();
        if self.subdomains.is_empty() {
            // Auto partition by rows using a greedy nnz-balanced heuristic
            let p = self
                .nparts_hint
                .unwrap_or_else(|| crate::parallel::threads::current_rayon_threads().max(1));
            let rp = csr.row_ptr();
            let mut nnz_per_row = vec![0usize; n];
            for i in 0..n {
                nnz_per_row[i] = rp[i + 1] - rp[i];
            }
            self.owner_of = greedy_nnz_balanced_partition(n, p, Some(&nnz_per_row));
        } else {
            // Infer owners from provided subdomains (assume non-overlapping; if overlapping,
            // first-declared owner wins deterministically).
            let p = self.subdomains.len();
            self.owner_of.clear();
            self.owner_of.resize(n, p);
            for (s, set) in self.subdomains.iter().enumerate() {
                for &gi in set {
                    if gi < n && self.owner_of[gi] == p {
                        self.owner_of[gi] = s;
                    }
                }
            }
            if self.owner_of.contains(&p) {
                let fallback = contiguous_partition(n, p.max(1));
                for i in 0..n {
                    if self.owner_of[i] == p {
                        self.owner_of[i] = fallback[i];
                    }
                }
            }
        }

        // 2) Expand overlap (k layers) to construct overlapped blocks
        let adj = build_adjacency(csr.as_ref());
        let overlapped = expand_overlap(&adj, &self.owner_of, self.overlap);
        self.subdomains = overlapped;

        // 3) Build per-block metadata (interior mask + placeholder weights)
        self.blocks_meta = self
            .subdomains
            .iter()
            .enumerate()
            .map(|(s, idx)| SubdomainMeta {
                indices: idx.clone(),
                interior_mask: idx.iter().map(|&gi| self.owner_of[gi] == s).collect(),
                weights: vec![S::one().real(); idx.len()],
            })
            .collect();

        // 4) Compute weights per strategy
        self.cover_count = compute_weights(
            &mut self.blocks_meta,
            &self.owner_of,
            &adj,
            self.weighting,
            n,
        );

        // 5) Build per-block solvers according to factory
        self.local_blocks.clear();
        self.local_blocks_csr.clear();
        match self.block_solver_factory {
            BlockSolverFactory::LuDense => {
                #[cfg(not(feature = "dense-direct"))]
                {
                    // Fallback: treat LuDense as CSR solver when dense-direct is disabled
                    let cfg = IluCsrConfig {
                        kind: IluKind::Ilu0,
                        pivot: PivotStrategy::DiagonalPerturbation,
                        pivot_threshold: 1e-12,
                        diag_perturb_factor: 1e-10,
                        level_sched: cfg!(feature = "rayon"),
                        numeric_update_fixed: true,
                        logging: 0,
                        reordering: ReorderingOptions::default(),
                    };
                    for meta in self.blocks_meta.iter() {
                        let idx = &meta.indices;
                        let a_sub_csr = Arc::new(csr.as_ref().submatrix(idx));
                        let mut ilu = IluCsr::new_with_config(cfg.clone());
                        let op = CsrOp::new(a_sub_csr.clone());
                        ilu.setup(&op)?;
                        self.local_blocks_csr
                            .push((a_sub_csr, std::sync::Arc::new(ilu)));
                    }
                }
                #[cfg(feature = "dense-direct")]
                {
                    for meta in self.blocks_meta.iter() {
                        let idx = &meta.indices;
                        let a_sub_csr = csr.as_ref().submatrix(idx);
                        let dense = a_sub_csr.to_dense();
                        let mut ksp = LuSolver::<f64>::new();
                        let _ = ksp.solve(
                            &dense,
                            None,
                            &vec![R::zero(); idx.len()],
                            &mut vec![R::zero(); idx.len()],
                            PcSide::Left,
                            &crate::parallel::UniverseComm::NoComm(crate::parallel::NoComm),
                            None,
                            None,
                        );
                        self.local_blocks
                            .push((dense, Mutex::new(Box::new(ksp) as _)));
                    }
                }
            }
            BlockSolverFactory::CsrSolver => {
                let cfg = IluCsrConfig {
                    kind: IluKind::Ilu0,
                    pivot: PivotStrategy::DiagonalPerturbation,
                    pivot_threshold: 1e-12,
                    diag_perturb_factor: 1e-10,
                    level_sched: cfg!(feature = "rayon"),
                    numeric_update_fixed: true,
                    logging: 0,
                    reordering: ReorderingOptions::default(),
                };
                for meta in self.blocks_meta.iter() {
                    let idx = &meta.indices;
                    let a_sub_csr = Arc::new(csr.as_ref().submatrix(idx));
                    let mut ilu = IluCsr::new_with_config(cfg.clone());
                    let op = CsrOp::new(a_sub_csr.clone());
                    ilu.setup(&op)?;
                    self.local_blocks_csr
                        .push((a_sub_csr, std::sync::Arc::new(ilu)));
                }
            }
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
            *yi = R::zero();
        }

        if self.blocks_meta.is_empty() {
            // Legacy/fallback path hit when callers used the generic setup routine
            // instead of the LinOp-aware one. We still have `subdomains` and
            // `local_blocks` populated, so reuse that data directly.
            return self.apply_dense_blocks_legacy(x, y);
        }

        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;
            let block_results: Vec<(Vec<usize>, Vec<R>, Vec<bool>, Vec<R>)> =
                match self.block_solver_factory {
                    BlockSolverFactory::LuDense => self
                        .blocks_meta
                        .par_iter()
                        .zip(self.local_blocks.par_iter())
                        .map(|(meta, (a_sub_any, ksp_mutex))| {
                            let indices = &meta.indices;
                            let r_blk: Vec<R> = indices.iter().map(|&i| x[i]).collect();
                            let mut x_blk = vec![R::zero(); indices.len()];
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
                            (
                                indices.clone(),
                                x_blk,
                                meta.interior_mask.clone(),
                                meta.weights.clone(),
                            )
                        })
                        .collect(),
                    BlockSolverFactory::CsrSolver => self
                        .blocks_meta
                        .par_iter()
                        .zip(self.local_blocks_csr.par_iter())
                        .map(|(meta, (_a_sub, ilu))| {
                            let indices = &meta.indices;
                            let r_blk: Vec<R> = indices.iter().map(|&i| x[i]).collect();
                            let mut x_blk = vec![R::zero(); indices.len()];
                            let _ = ilu.apply(PcSide::Left, &r_blk, &mut x_blk);
                            (
                                indices.clone(),
                                x_blk,
                                meta.interior_mask.clone(),
                                meta.weights.clone(),
                            )
                        })
                        .collect(),
                };
            match self.asm_mode {
                AsmMode::ASM => {
                    for (indices, x_blk, _mask, w) in block_results {
                        if matches!(self.weighting, Weighting::None) {
                            for (j, &gi) in indices.iter().enumerate() {
                                y[gi] += x_blk[j];
                            }
                        } else {
                            for (j, &gi) in indices.iter().enumerate() {
                                y[gi] += w[j] * x_blk[j];
                            }
                        }
                    }
                }
                AsmMode::RAS => {
                    for (indices, x_blk, mask, w) in block_results {
                        for (j, &gi) in indices.iter().enumerate() {
                            if mask[j] {
                                let wij = match self.weighting {
                                    Weighting::None => 1.0,
                                    _ => w[j],
                                };
                                y[gi] += wij * x_blk[j];
                            }
                        }
                    }
                }
            }
        }

        #[cfg(not(feature = "rayon"))]
        {
            match self.block_solver_factory {
                BlockSolverFactory::LuDense => {
                    self.blocks_meta
                        .iter()
                        .zip(self.local_blocks.iter())
                        .for_each(|(meta, (a_sub_any, ksp_mutex))| {
                            let indices = &meta.indices;
                            let r_blk: Vec<R> = indices.iter().map(|&i| x[i]).collect();
                            let mut x_blk = vec![R::zero(); indices.len()];
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
                            match self.asm_mode {
                                AsmMode::ASM => {
                                    if matches!(self.weighting, Weighting::None) {
                                        for (j, &gi) in indices.iter().enumerate() {
                                            y[gi] += x_blk[j];
                                        }
                                    } else {
                                        for (j, &gi) in indices.iter().enumerate() {
                                            y[gi] += meta.weights[j] * x_blk[j];
                                        }
                                    }
                                }
                                AsmMode::RAS => {
                                    for (j, &gi) in indices.iter().enumerate() {
                                        if meta.interior_mask[j] {
                                            let wij = match self.weighting {
                                                Weighting::None => 1.0,
                                                _ => meta.weights[j],
                                            };
                                            y[gi] += wij * x_blk[j];
                                        }
                                    }
                                }
                            }
                        });
                }
                BlockSolverFactory::CsrSolver => {
                    self.blocks_meta
                        .iter()
                        .zip(self.local_blocks_csr.iter())
                        .for_each(|(meta, (_a_sub, ilu))| {
                            let indices = &meta.indices;
                            let r_blk: Vec<R> = indices.iter().map(|&i| x[i]).collect();
                            let mut x_blk = vec![R::zero(); indices.len()];
                            let _ = ilu.apply(PcSide::Left, &r_blk, &mut x_blk);
                            match self.asm_mode {
                                AsmMode::ASM => {
                                    if matches!(self.weighting, Weighting::None) {
                                        for (j, &gi) in indices.iter().enumerate() {
                                            y[gi] += x_blk[j];
                                        }
                                    } else {
                                        for (j, &gi) in indices.iter().enumerate() {
                                            y[gi] += meta.weights[j] * x_blk[j];
                                        }
                                    }
                                }
                                AsmMode::RAS => {
                                    for (j, &gi) in indices.iter().enumerate() {
                                        if meta.interior_mask[j] {
                                            let wij = match self.weighting {
                                                Weighting::None => 1.0,
                                                _ => meta.weights[j],
                                            };
                                            y[gi] += wij * x_blk[j];
                                        }
                                    }
                                }
                            }
                        });
                }
            }
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

        // Recreate local blocks and solvers according to factory
        self.local_blocks.clear();
        self.local_blocks_csr.clear();
        match self.block_solver_factory {
            BlockSolverFactory::LuDense => {
                #[cfg(feature = "dense-direct")]
                {
                    for meta in self.blocks_meta.iter() {
                        let indices = &meta.indices;
                        let a_sub_csr = csr.as_ref().submatrix(indices);
                        let dense = a_sub_csr.to_dense();
                        let mut ksp = LuSolver::<f64>::new();
                        let _ = ksp.solve(
                            &dense,
                            None,
                            &vec![R::zero(); indices.len()],
                            &mut vec![R::zero(); indices.len()],
                            PcSide::Left,
                            &crate::parallel::UniverseComm::NoComm(crate::parallel::NoComm),
                            None,
                            None,
                        );
                        self.local_blocks
                            .push((dense, Mutex::new(Box::new(ksp) as _)));
                    }
                }
                #[cfg(not(feature = "dense-direct"))]
                {
                    // Fallback to CSR ILU blocks when dense-direct is disabled
                    let cfg = IluCsrConfig {
                        kind: IluKind::Ilu0,
                        pivot: PivotStrategy::DiagonalPerturbation,
                        pivot_threshold: 1e-12,
                        diag_perturb_factor: 1e-10,
                        level_sched: cfg!(feature = "rayon"),
                        numeric_update_fixed: true,
                        logging: 0,
                        reordering: ReorderingOptions::default(),
                    };
                    for meta in self.blocks_meta.iter() {
                        let indices = &meta.indices;
                        let a_sub_csr = Arc::new(csr.as_ref().submatrix(indices));
                        let mut ilu = IluCsr::new_with_config(cfg.clone());
                        let op = CsrOp::new(a_sub_csr.clone());
                        ilu.setup(&op)?;
                        self.local_blocks_csr
                            .push((a_sub_csr, std::sync::Arc::new(ilu)));
                    }
                }
            }
            BlockSolverFactory::CsrSolver => {
                let cfg = IluCsrConfig {
                    kind: IluKind::Ilu0,
                    pivot: PivotStrategy::DiagonalPerturbation,
                    pivot_threshold: 1e-12,
                    diag_perturb_factor: 1e-10,
                    level_sched: cfg!(feature = "rayon"),
                    numeric_update_fixed: true,
                    logging: 0,
                    reordering: ReorderingOptions::default(),
                };
                for meta in self.blocks_meta.iter() {
                    let indices = &meta.indices;
                    let a_sub_csr = Arc::new(csr.as_ref().submatrix(indices));
                    let mut ilu = IluCsr::new_with_config(cfg.clone());
                    let op = CsrOp::new(a_sub_csr.clone());
                    ilu.setup(&op)?;
                    self.local_blocks_csr
                        .push((a_sub_csr, std::sync::Arc::new(ilu)));
                }
            }
        }

        self.last_vid = Some(op.values_id());
        Ok(())
    }

    fn update_symbolic(&mut self, op: &dyn LinOp<S = f64>) -> Result<(), KError> {
        // Re-run full setup steps on the new structure
        let csr = csr_from_linop(op, self.drop_tol)?;
        self.csr = Some(csr.clone());

        let n = csr.nrows();
        // Preserve existing owner_of if provided subdomains were explicitly set; otherwise rebuild
        if self.subdomains.is_empty() || self.owner_of.is_empty() {
            let p = self
                .nparts_hint
                .unwrap_or_else(|| crate::parallel::threads::current_rayon_threads().max(1));
            let rp = csr.row_ptr();
            let mut nnz_per_row = vec![0usize; n];
            for i in 0..n {
                nnz_per_row[i] = rp[i + 1] - rp[i];
            }
            self.owner_of = greedy_nnz_balanced_partition(n, p, Some(&nnz_per_row));
        }

        // Overlap and metadata
        let adj = build_adjacency(csr.as_ref());
        let overlapped = expand_overlap(&adj, &self.owner_of, self.overlap);
        self.subdomains = overlapped;
        self.blocks_meta = self
            .subdomains
            .iter()
            .enumerate()
            .map(|(s, idx)| SubdomainMeta {
                indices: idx.clone(),
                interior_mask: idx.iter().map(|&gi| self.owner_of[gi] == s).collect(),
                weights: vec![S::one().real(); idx.len()],
            })
            .collect();
        self.cover_count = compute_weights(
            &mut self.blocks_meta,
            &self.owner_of,
            &adj,
            self.weighting,
            n,
        );

        // Recreate local blocks and solvers according to factory
        self.local_blocks.clear();
        self.local_blocks_csr.clear();
        match self.block_solver_factory {
            BlockSolverFactory::LuDense => {
                #[cfg(feature = "dense-direct")]
                {
                    for meta in self.blocks_meta.iter() {
                        let indices = &meta.indices;
                        let a_sub_csr = csr.as_ref().submatrix(indices);
                        let dense = a_sub_csr.to_dense();
                        let mut ksp = LuSolver::<f64>::new();
                        let _ = ksp.solve(
                            &dense,
                            None,
                            &vec![R::zero(); indices.len()],
                            &mut vec![R::zero(); indices.len()],
                            PcSide::Left,
                            &crate::parallel::UniverseComm::NoComm(crate::parallel::NoComm),
                            None,
                            None,
                        );
                        self.local_blocks
                            .push((dense, Mutex::new(Box::new(ksp) as _)));
                    }
                }
                #[cfg(not(feature = "dense-direct"))]
                {
                    // Fallback to CSR ILU blocks when dense-direct is disabled
                    let cfg = IluCsrConfig {
                        kind: IluKind::Ilu0,
                        pivot: PivotStrategy::DiagonalPerturbation,
                        pivot_threshold: 1e-12,
                        diag_perturb_factor: 1e-10,
                        level_sched: cfg!(feature = "rayon"),
                        numeric_update_fixed: true,
                        logging: 0,
                        reordering: ReorderingOptions::default(),
                    };
                    for meta in self.blocks_meta.iter() {
                        let indices = &meta.indices;
                        let a_sub_csr = Arc::new(csr.as_ref().submatrix(indices));
                        let mut ilu = IluCsr::new_with_config(cfg.clone());
                        let op = CsrOp::new(a_sub_csr.clone());
                        ilu.setup(&op)?;
                        self.local_blocks_csr
                            .push((a_sub_csr, std::sync::Arc::new(ilu)));
                    }
                }
            }
            BlockSolverFactory::CsrSolver => {
                let cfg = IluCsrConfig {
                    kind: IluKind::Ilu0,
                    pivot: PivotStrategy::DiagonalPerturbation,
                    pivot_threshold: 1e-12,
                    diag_perturb_factor: 1e-10,
                    level_sched: cfg!(feature = "rayon"),
                    numeric_update_fixed: true,
                    logging: 0,
                    reordering: ReorderingOptions::default(),
                };
                for meta in self.blocks_meta.iter() {
                    let indices = &meta.indices;
                    let a_sub_csr = Arc::new(csr.as_ref().submatrix(indices));
                    let mut ilu = IluCsr::new_with_config(cfg.clone());
                    let op = CsrOp::new(a_sub_csr.clone());
                    ilu.setup(&op)?;
                    self.local_blocks_csr
                        .push((a_sub_csr, std::sync::Arc::new(ilu)));
                }
            }
        }

        self.last_sid = Some(op.structure_id());
        self.last_vid = Some(op.values_id());
        Ok(())
    }
}

impl AdditiveSchwarz<faer::Mat<f64>, Vec<f64>, f64> {
    fn apply_dense_blocks_legacy(&self, x: &[f64], y: &mut [f64]) -> Result<(), KError> {
        if self.subdomains.len() != self.local_blocks.len() {
            return Err(KError::InvalidInput(
                "ASM legacy apply: subdomains/local_blocks length mismatch".into(),
            ));
        }
        for (indices, (a_sub, ksp_mutex)) in self.subdomains.iter().zip(self.local_blocks.iter()) {
            let r_blk: Vec<f64> = indices.iter().map(|&i| x[i]).collect();
            let mut x_blk = vec![R::zero(); indices.len()];
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
                y[gi] += x_blk[j];
            }
        }
        Ok(())
    }
}

#[cfg(all(test, feature = "dense-direct"))]
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
        crate::preconditioner::Preconditioner::apply(&asm, PcSide::Left, &r, &mut z).unwrap();
        // For identity, ASM should return the input
        assert_eq!(z, r);
    }
}

// ---------------- f64-only helpers (private) ----------------

fn build_adjacency(csr: &CsrMatrix<f64>) -> Vec<Vec<usize>> {
    let n = csr.nrows();
    let rp = csr.row_ptr();
    let cj = csr.col_idx();
    let mut adj = vec![Vec::<usize>::new(); n];
    for i in 0..n {
        for p in rp[i]..rp[i + 1] {
            let j = cj[p];
            if i != j {
                adj[i].push(j);
            }
        }
    }
    // Make symmetric
    for i in 0..n {
        for &j in adj[i].clone().iter() {
            if !adj[j].contains(&i) {
                adj[j].push(i);
            }
        }
    }
    for i in 0..n {
        adj[i].sort_unstable();
        adj[i].dedup();
    }
    adj
}

fn expand_overlap(adj: &[Vec<usize>], owner_of: &[usize], k: usize) -> Vec<Vec<usize>> {
    let n = owner_of.len();
    let p = owner_of.iter().copied().max().unwrap_or(0) + 1;
    use std::collections::BTreeSet;
    let mut blocks: Vec<BTreeSet<usize>> = vec![Default::default(); p];
    for i in 0..n {
        blocks[owner_of[i]].insert(i);
    }
    let mut frontier = blocks.clone();
    for _ in 0..k {
        let mut next = blocks.clone();
        for s in 0..p {
            for &u in &frontier[s] {
                for &v in &adj[u] {
                    next[s].insert(v);
                }
            }
        }
        frontier = next.clone();
        blocks = next;
    }
    blocks
        .into_iter()
        .map(|b| b.into_iter().collect())
        .collect()
}

fn compute_weights(
    blocks: &mut [SubdomainMeta],
    _owner_of: &[usize],
    adj: &[Vec<usize>],
    weighting: Weighting,
    n: usize,
) -> Vec<usize> {
    let mut cover_count = vec![0usize; n];
    for b in blocks.iter() {
        for &gi in &b.indices {
            cover_count[gi] += 1;
        }
    }

    if matches!(weighting, Weighting::None) {
        let one = S::one().real();
        for b in blocks.iter_mut() {
            b.weights.resize(b.indices.len(), one);
        }
        return cover_count;
    }

    let mut phi: Vec<Vec<R>> = vec![Vec::new(); blocks.len()];
    for (s, b) in blocks.iter().enumerate() {
        let mut dist = vec![0usize; b.indices.len()];
        if !matches!(weighting, Weighting::Uniform) {
            // membership map
            let mut in_s = vec![false; n];
            for &gi in &b.indices {
                in_s[gi] = true;
            }
            use std::collections::{HashMap, VecDeque};
            let mut q = VecDeque::new();
            let mut dmap: HashMap<usize, usize> = Default::default();
            for &gi in &b.indices {
                if adj[gi].iter().any(|&v| !in_s[v]) {
                    q.push_back(gi);
                    dmap.insert(gi, 0);
                }
            }
            while let Some(u) = q.pop_front() {
                let du = dmap[&u];
                for &v in &adj[u] {
                    if in_s[v] && !dmap.contains_key(&v) {
                        dmap.insert(v, du + 1);
                        q.push_back(v);
                    }
                }
            }
            for (j, &gi) in b.indices.iter().enumerate() {
                dist[j] = *dmap.get(&gi).unwrap_or(&0);
            }
        }
        let one = S::one().real();
        let phi_s: Vec<R> = match weighting {
            Weighting::Uniform => vec![one; b.indices.len()],
            Weighting::SmoothLinear => dist.iter().map(|&d| (d as R) + one).collect(),
            Weighting::SmoothPoly(p) => dist
                .iter()
                .map(|&d| ((d as R) + one).powi(p as i32))
                .collect(),
            Weighting::None => unreachable!(),
        };
        phi[s] = phi_s;
    }
    // Build coverer index: for each global index, which (block, local_pos) cover it
    let mut coverers: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n];
    for (t, bt) in blocks.iter().enumerate() {
        for (pos, &gi) in bt.indices.iter().enumerate() {
            coverers[gi].push((t, pos));
        }
    }

    let one = S::one().real();
    for (s, b) in blocks.iter_mut().enumerate() {
        b.weights.resize(b.indices.len(), one);
        for (j, &gi) in b.indices.iter().enumerate() {
            let denom = match weighting {
                Weighting::Uniform => cover_count[gi] as R,
                _ => {
                    let mut sum = R::default();
                    for &(t, pos) in &coverers[gi] {
                        sum += phi[t][pos];
                    }
                    if sum <= R::default() { one } else { sum }
                }
            };
            let num = match weighting {
                Weighting::Uniform => one,
                _ => {
                    let pos = b.indices.binary_search(&gi).unwrap();
                    phi[s][pos]
                }
            };
            b.weights[j] = num / denom;
        }
    }
    cover_count
}

impl AdditiveSchwarz<faer::Mat<f64>, Vec<f64>, f64> {
    /// Select ASM mode (ASM or RAS)
    pub fn set_mode(&mut self, mode: AsmMode) -> &mut Self {
        self.asm_mode = mode;
        self
    }
    /// Select PoU weighting strategy.
    pub fn set_weighting(&mut self, w: Weighting) -> &mut Self {
        self.weighting = w;
        self
    }
    /// Set overlap layers; adjusts default weighting if currently None.
    pub fn set_overlap(&mut self, k: usize) -> &mut Self {
        self.overlap = k;
        if matches!(self.weighting, Weighting::None) && k > 0 {
            self.weighting = Weighting::Uniform;
        }
        self
    }
    /// Set number of primary partitions if subdomains are not provided.
    pub fn set_num_parts(&mut self, p: usize) -> &mut Self {
        self.nparts_hint = Some(p.max(1));
        self
    }
    /// Set dense threshold for per-block solver selection (placeholder for future CSR path).
    pub fn set_dense_threshold(&mut self, n: usize) -> &mut Self {
        self.dense_threshold = n;
        self
    }
}

// ===== Modern ASM implementation used by AMG-DD =====================================

/// Combination rule for overlapping subdomain solves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AsmCombine {
    /// Classical additive Schwarz (overlapped contributions summed, optionally weighted).
    Additive,
    /// Restricted additive Schwarz (inject only interior/core unknowns per subdomain).
    Restricted,
    /// Optimized RAS/ORAS-style weighting (smooth partition of unity on overlap).
    Optimized,
}

/// Local solver choice for each subdomain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AsmLocalSolver {
    /// ILU(0) on the subdomain matrix.
    ILU,
}

/// Configuration for the standalone ASM preconditioner.
#[derive(Clone, Debug)]
pub struct AsmConfig {
    pub overlap: usize,
    pub combine: AsmCombine,
    pub local_solver: AsmLocalSolver,
    pub local_sweeps: usize,
    pub weight_partition_of_unity: bool,
    pub deterministic: bool,
    pub nparts: Option<usize>,
}

impl Default for AsmConfig {
    fn default() -> Self {
        Self {
            overlap: 0,
            combine: AsmCombine::Additive,
            local_solver: AsmLocalSolver::ILU,
            local_sweeps: 1,
            weight_partition_of_unity: true,
            deterministic: true,
            nparts: None,
        }
    }
}

/// Builder for [`Asm`].
#[derive(Clone, Debug)]
pub struct AsmBuilder {
    cfg: AsmConfig,
}

impl AsmBuilder {
    pub fn new() -> Self {
        Self {
            cfg: AsmConfig::default(),
        }
    }

    pub fn overlap(mut self, k: usize) -> Self {
        self.cfg.overlap = k;
        self
    }

    pub fn combine(mut self, combine: AsmCombine) -> Self {
        self.cfg.combine = combine;
        self
    }

    pub fn local_solver(mut self, solver: AsmLocalSolver) -> Self {
        self.cfg.local_solver = solver;
        self
    }

    pub fn local_sweeps(mut self, sweeps: usize) -> Self {
        self.cfg.local_sweeps = sweeps.max(1);
        self
    }

    pub fn weight_partition_of_unity(mut self, enabled: bool) -> Self {
        self.cfg.weight_partition_of_unity = enabled;
        self
    }

    pub fn deterministic(mut self, deterministic: bool) -> Self {
        self.cfg.deterministic = deterministic;
        self
    }

    pub fn parts(mut self, nparts: usize) -> Self {
        self.cfg.nparts = Some(nparts.max(1));
        self
    }

    pub fn build(self) -> Asm {
        Asm::with_config(self.cfg)
    }
}

impl Default for AsmBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// In-memory representation of a single overlapping subdomain.
struct AsmSubdomain {
    pro2glob: Vec<usize>,
    restrict: Vec<usize>,
    restrict_local: Vec<usize>,
    matrix: Arc<CsrMatrix<f64>>,
    solver: LocalSolver,
    rhs: Mutex<Vec<R>>,
    sol: Mutex<Vec<R>>,
    weights: Vec<R>,
}

impl AsmSubdomain {
    fn new(
        matrix: Arc<CsrMatrix<f64>>,
        pro2glob: Vec<usize>,
        restrict: Vec<usize>,
        weights: Vec<R>,
        solver: LocalSolver,
    ) -> Self {
        let n = pro2glob.len();
        let restrict_local = restrict
            .iter()
            .map(|g| {
                pro2glob
                    .binary_search(g)
                    .expect("restrict idx not in subdomain")
            })
            .collect();
        Self {
            rhs: Mutex::new(vec![R::zero(); n]),
            sol: Mutex::new(vec![R::zero(); n]),
            pro2glob,
            restrict,
            restrict_local,
            matrix,
            solver,
            weights,
        }
    }
}

enum LocalSolver {
    Ilu(IluCsr),
}

impl LocalSolver {
    fn from_config(kind: AsmLocalSolver, mat: &Arc<CsrMatrix<f64>>) -> Result<Self, KError> {
        match kind {
            AsmLocalSolver::ILU => {
                let cfg = IluCsrConfig {
                    kind: IluKind::Ilu0,
                    pivot: PivotStrategy::DiagonalPerturbation,
                    pivot_threshold: 1e-12,
                    diag_perturb_factor: 1e-10,
                    level_sched: cfg!(feature = "rayon"),
                    numeric_update_fixed: true,
                    logging: 0,
                    reordering: ReorderingOptions::default(),
                };
                let mut ilu = IluCsr::new_with_config(cfg);
                let op = CsrOp::new(mat.clone());
                ilu.setup(&op)?;
                Ok(LocalSolver::Ilu(ilu))
            }
        }
    }

    fn apply(&self, rhs: &[f64], sol: &mut [f64]) -> Result<(), KError> {
        match self {
            LocalSolver::Ilu(ilu) => ilu.apply(PcSide::Left, rhs, sol),
        }
    }

    fn update_numeric(&mut self, mat: &Arc<CsrMatrix<f64>>) -> Result<(), KError> {
        match self {
            LocalSolver::Ilu(ilu) => {
                let op = CsrOp::new(mat.clone());
                ilu.update_numeric(&op)
            }
        }
    }

    fn update_symbolic(&mut self, mat: &Arc<CsrMatrix<f64>>) -> Result<(), KError> {
        match self {
            LocalSolver::Ilu(ilu) => {
                let op = CsrOp::new(mat.clone());
                ilu.update_symbolic(&op)
            }
        }
    }
}

struct AsmState {
    a_fine: Arc<CsrMatrix<f64>>,
    subdomains: Vec<AsmSubdomain>,
}

/// Lightweight additive/restricted Schwarz preconditioner used by AMG-DD.
pub struct Asm {
    cfg: AsmConfig,
    state: Option<AsmState>,
    last_sid: Option<StructureId>,
    last_vid: Option<ValuesId>,
}

impl Asm {
    pub fn builder() -> AsmBuilder {
        AsmBuilder::new()
    }

    pub fn with_config(cfg: AsmConfig) -> Self {
        Self {
            cfg,
            state: None,
            last_sid: None,
            last_vid: None,
        }
    }

    pub fn dimension(&self) -> Option<usize> {
        self.state.as_ref().map(|s| s.a_fine.nrows())
    }

    pub fn matrix(&self) -> Option<&Arc<CsrMatrix<f64>>> {
        self.state.as_ref().map(|s| &s.a_fine)
    }

    fn build_subdomains(&self, csr: &Arc<CsrMatrix<f64>>) -> Result<Vec<AsmSubdomain>, KError> {
        let n = csr.nrows();
        let rp = csr.row_ptr();
        let mut nnz_per_row = vec![0usize; n];
        for i in 0..n {
            nnz_per_row[i] = rp[i + 1] - rp[i];
        }
        let nparts = self
            .cfg
            .nparts
            .unwrap_or_else(|| crate::parallel::threads::current_rayon_threads().max(1));
        let owner_of = greedy_nnz_balanced_partition(n, nparts, Some(&nnz_per_row));
        let adj = build_adjacency(csr.as_ref());
        let overlapped = expand_overlap(&adj, &owner_of, self.cfg.overlap);
        let weighting = match (self.cfg.combine, self.cfg.weight_partition_of_unity) {
            (AsmCombine::Additive, true) => Weighting::Uniform,
            (AsmCombine::Restricted, true) => Weighting::None,
            (AsmCombine::Optimized, true) => Weighting::SmoothLinear,
            _ => Weighting::None,
        };
        let mut metas: Vec<SubdomainMeta> = overlapped
            .iter()
            .enumerate()
            .map(|(s, idx)| SubdomainMeta {
                indices: idx.clone(),
                interior_mask: idx.iter().map(|&gi| owner_of[gi] == s).collect(),
                weights: vec![S::one().real(); idx.len()],
            })
            .collect();
        if !matches!(weighting, Weighting::None) {
            compute_weights(&mut metas, &owner_of, &adj, weighting, n);
        }

        let mut subdomains = Vec::with_capacity(overlapped.len());
        for (s, meta) in metas.into_iter().enumerate() {
            let pro2glob = meta.indices;
            let restrict: Vec<usize> = match self.cfg.combine {
                AsmCombine::Additive => pro2glob.clone(),
                AsmCombine::Restricted | AsmCombine::Optimized => pro2glob
                    .iter()
                    .copied()
                    .filter(|&gi| owner_of[gi] == s)
                    .collect(),
            };
            let mat = Arc::new(csr.as_ref().submatrix(&pro2glob));
            let solver = LocalSolver::from_config(self.cfg.local_solver, &mat)?;
            let weights = if matches!(weighting, Weighting::None) {
                vec![S::one().real(); pro2glob.len()]
            } else {
                meta.weights
            };
            subdomains.push(AsmSubdomain::new(mat, pro2glob, restrict, weights, solver));
        }
        Ok(subdomains)
    }

    fn apply_impl(&self, rhs: &[f64], out: &mut [f64]) -> Result<(), KError> {
        let state = self
            .state
            .as_ref()
            .ok_or_else(|| KError::InvalidInput("ASM not set up".into()))?;
        if rhs.len() != out.len() {
            return Err(KError::InvalidInput("ASM apply dimension mismatch".into()));
        }
        for yi in out.iter_mut() {
            *yi = R::zero();
        }
        for sub in &state.subdomains {
            let mut rhs_loc = sub.rhs.lock().unwrap();
            let mut sol_loc = sub.sol.lock().unwrap();
            for (li, &gi) in sub.pro2glob.iter().enumerate() {
                rhs_loc[li] = rhs[gi];
                sol_loc[li] = R::zero();
            }
            sub.solver.apply(&rhs_loc, &mut sol_loc)?;
            match self.cfg.combine {
                AsmCombine::Additive | AsmCombine::Optimized => {
                    for (li, &gi) in sub.pro2glob.iter().enumerate() {
                        out[gi] += sub.weights[li] * sol_loc[li];
                    }
                }
                AsmCombine::Restricted => {
                    for (&gi, &li) in sub.restrict.iter().zip(sub.restrict_local.iter()) {
                        out[gi] += sub.weights[li] * sol_loc[li];
                    }
                }
            }
        }
        Ok(())
    }
}

impl DynPreconditioner for Asm {
    fn setup(&mut self, op: &dyn LinOp<S = f64>) -> Result<(), KError> {
        let csr = csr_from_linop(op, 0.0)?;
        let subdomains = self.build_subdomains(&csr)?;
        self.state = Some(AsmState {
            a_fine: csr.clone(),
            subdomains,
        });
        self.last_sid = Some(op.structure_id());
        self.last_vid = Some(op.values_id());
        Ok(())
    }

    fn apply(&self, _side: PcSide, rhs: &[f64], out: &mut [f64]) -> Result<(), KError> {
        self.apply_impl(rhs, out)
    }

    fn supports_numeric_update(&self) -> bool {
        true
    }

    fn update_numeric(&mut self, op: &dyn LinOp<S = f64>) -> Result<(), KError> {
        if Some(op.structure_id()) != self.last_sid {
            return Err(KError::Unsupported(
                "ASM numeric update requires identical sparsity pattern",
            ));
        }
        let csr = csr_from_linop(op, 0.0)?;
        let state = self
            .state
            .as_mut()
            .ok_or_else(|| KError::InvalidInput("ASM not set up".into()))?;
        for sub in state.subdomains.iter_mut() {
            let fresh = csr.as_ref().submatrix(&sub.pro2glob);
            let mat = Arc::make_mut(&mut sub.matrix);
            mat.values_mut().copy_from_slice(fresh.values());
            sub.solver.update_numeric(&sub.matrix)?;
        }
        state.a_fine = csr.clone();
        self.last_vid = Some(op.values_id());
        Ok(())
    }

    fn update_symbolic(&mut self, op: &dyn LinOp<S = f64>) -> Result<(), KError> {
        let csr = csr_from_linop(op, 0.0)?;
        let mut subdomains = self.build_subdomains(&csr)?;
        for sub in subdomains.iter_mut() {
            sub.solver.update_symbolic(&sub.matrix)?;
        }
        self.state = Some(AsmState {
            a_fine: csr.clone(),
            subdomains,
        });
        self.last_sid = Some(op.structure_id());
        self.last_vid = Some(op.values_id());
        Ok(())
    }

    fn capabilities(&self) -> PcCaps {
        PcCaps::default()
    }
}

#[cfg(feature = "complex")]
impl crate::ops::kpc::KPreconditioner for Asm {
    type Scalar = S;

    #[inline]
    fn dims(&self) -> (usize, usize) {
        self.dimension().map(|n| (n, n)).unwrap_or((0, 0))
    }

    fn apply_s(
        &self,
        side: PcSide,
        x: &[S],
        y: &mut [S],
        scratch: &mut BridgeScratch,
    ) -> Result<(), KError> {
        apply_pc_s(self, side, x, y, scratch)
    }

    fn apply_mut_s(
        &mut self,
        side: PcSide,
        x: &[S],
        y: &mut [S],
        scratch: &mut BridgeScratch,
    ) -> Result<(), KError> {
        apply_pc_mut_s(self, side, x, y, scratch)
    }
}
