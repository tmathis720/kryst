#[cfg(feature = "mpi")]
use super::{AsmBlockSolver, AsmInnerPc, AsmMode, Weighting};
#[cfg(feature = "mpi")]
use crate::algebra::prelude::*;
#[cfg(feature = "mpi")]
use crate::error::KError;
#[cfg(feature = "mpi")]
use crate::matrix::DistCsrOp;
#[cfg(all(feature = "mpi", not(feature = "complex")))]
use crate::matrix::convert::materialize_linop_with_hint;
#[cfg(all(feature = "mpi", not(feature = "complex")))]
use crate::matrix::format::FormatHint;
#[cfg(feature = "mpi")]
use crate::matrix::format::OpFormat;
#[cfg(feature = "mpi")]
use crate::matrix::op::CsrOp;
#[cfg(all(
    feature = "mpi",
    feature = "backend-faer",
    feature = "legacy-pc-bridge"
))]
use crate::matrix::op::DenseOp;
use crate::matrix::op::{DistLayout, LinOp, StructureId, ValuesId};
#[cfg(feature = "mpi")]
use crate::matrix::sparse::CsrMatrix;
#[cfg(feature = "mpi")]
use crate::parallel::{Comm, UniverseComm, contiguous_partition};
#[cfg(feature = "mpi")]
use crate::preconditioner::builders::build_jacobi;
#[cfg(all(feature = "mpi", not(feature = "complex")))]
use crate::preconditioner::builders::{
    build_ilu0_with_conditioning, build_ilut_with_conditioning, build_ilutp_with_conditioning,
};
use crate::preconditioner::dist::DistCoarseStrategy;
#[cfg(feature = "mpi")]
use crate::preconditioner::{PcDistributedSupport, PcSide, Preconditioner};
use crate::utils::conditioning::ConditioningOptions;
#[cfg(feature = "mpi")]
use std::collections::{HashMap, HashSet};
#[cfg(feature = "mpi")]
use std::sync::Arc;

#[cfg(feature = "mpi")]
use super::comm_plan::{CommPlan, alltoallv_scalar, alltoallv_u64};
#[cfg(feature = "mpi")]
use super::subdomain::{RemoteRow, build_subdomain_csr, request_remote_rows};

#[cfg(feature = "mpi")]
#[derive(Debug)]
pub struct DistributedAsm {
    overlap: usize,
    subdomain_hint: Option<usize>,
    block_solver: AsmBlockSolver,
    inner_pc: AsmInnerPc,
    mode: AsmMode,
    weighting: Weighting,
    coarse_strategy: DistCoarseStrategy,
    state: Option<DistributedAsmState>,
    last_sid: Option<StructureId>,
    last_vid: Option<ValuesId>,
}

#[cfg(feature = "mpi")]
#[derive(Debug)]
struct DistributedAsmState {
    comm: UniverseComm,
    layout: DistLayout,
    local_csr: Arc<CsrMatrix<S>>,
    subdofs: Vec<usize>,
    sub_map: HashMap<usize, usize>,
    comm_plan: CommPlan,
    sub_csr: Arc<CsrMatrix<S>>,
    solver: SubdomainSolver,
    weights: Option<Vec<R>>,
    coarse: Option<DistributedAsmCoarse>,
}

#[cfg(feature = "mpi")]
#[derive(Debug)]
struct DistributedAsmCoarse {
    strategy: DistCoarseStrategy,
    ownership: Vec<(usize, usize)>,
    root_matrix: Option<Vec<Vec<S>>>,
}

#[cfg(feature = "mpi")]
impl DistributedAsm {
    pub fn new(
        overlap: usize,
        subdomain_hint: Option<usize>,
        block_solver: AsmBlockSolver,
        inner_pc: AsmInnerPc,
        mode: AsmMode,
        weighting: Weighting,
        coarse_strategy: DistCoarseStrategy,
    ) -> Self {
        Self {
            overlap,
            subdomain_hint,
            block_solver,
            inner_pc,
            mode,
            weighting,
            coarse_strategy,
            state: None,
            last_sid: None,
            last_vid: None,
        }
    }

    pub fn new_ras(
        overlap: usize,
        subdomain_hint: Option<usize>,
        block_solver: AsmBlockSolver,
        inner_pc: AsmInnerPc,
        weighting: Weighting,
    ) -> Self {
        Self::new(
            overlap,
            subdomain_hint,
            block_solver,
            inner_pc,
            AsmMode::RAS,
            weighting,
            DistCoarseStrategy::None,
        )
    }
}

#[cfg(feature = "mpi")]
impl Preconditioner for DistributedAsm {
    fn dims(&self) -> (usize, usize) {
        self.state
            .as_ref()
            .map(|s| {
                (
                    s.layout.row_end - s.layout.row_start,
                    s.layout.row_end - s.layout.row_start,
                )
            })
            .unwrap_or((0, 0))
    }

    fn setup(&mut self, op: &dyn LinOp<S = S>) -> Result<(), KError> {
        let comm = op.comm();
        if comm.size() <= 1 {
            return Err(KError::Unsupported(
                "Distributed ASM requires an MPI communicator".into(),
            ));
        }

        let layout = dist_layout_from_op(op).ok_or_else(|| {
            KError::Unsupported("Distributed ASM requires a distributed layout".into())
        })?;

        let local_csr = materialize_local_csr(op)?;

        let ownership = build_ownership(&layout, comm.size());

        let (subdofs, remote_rows) =
            build_overlap_set(&local_csr, &layout, &ownership, self.overlap, &comm)?;
        let mut remote_rows = remote_rows;

        let missing: Vec<usize> = subdofs
            .iter()
            .copied()
            .filter(|g| *g < layout.row_start || *g >= layout.row_end)
            .filter(|g| !remote_rows.contains_key(g))
            .collect();
        if !missing.is_empty() {
            let fetched = request_remote_rows(
                &comm,
                &ownership,
                layout.row_start,
                layout.row_end,
                &local_csr,
                &missing,
            )?;
            remote_rows.extend(fetched);
        }

        let sub_csr = Arc::new(build_subdomain_csr(
            &subdofs,
            layout.row_start,
            layout.row_end,
            &local_csr,
            &remote_rows,
        )?);

        let comm_plan = build_comm_plan(&comm, &ownership, &subdofs)?;
        let imported_rows: usize = comm_plan.imports.iter().map(|rows| rows.len()).sum();
        log::info!(
            "distributed {:?} ASM setup: rank={} overlap={} imported_rows={} local_subdomain_nnz={}",
            self.mode,
            comm.rank(),
            self.overlap,
            imported_rows,
            sub_csr.nnz()
        );
        let sub_map = subdofs.iter().enumerate().map(|(i, &g)| (g, i)).collect();

        let mut solver = SubdomainSolver::new(self.block_solver, self.inner_pc)?;
        solver.setup(&sub_csr)?;

        let weights = build_ras_weights(&layout, &comm_plan, self.weighting);
        let coarse =
            build_coarse_space(self.coarse_strategy, &comm, &layout, &local_csr, &ownership)?;

        self.last_sid = Some(op.structure_id());
        self.last_vid = Some(op.values_id());
        self.state = Some(DistributedAsmState {
            comm,
            layout,
            local_csr,
            subdofs,
            sub_map,
            comm_plan,
            sub_csr,
            solver,
            weights,
            coarse,
        });
        Ok(())
    }

    fn apply(&self, _side: PcSide, x: &[S], y: &mut [S]) -> Result<(), KError> {
        let state = self.state.as_ref().ok_or_else(|| {
            KError::InvalidInput("Distributed ASM preconditioner not setup".into())
        })?;
        let n_local = state.layout.row_end - state.layout.row_start;
        if x.len() != n_local || y.len() != n_local {
            return Err(KError::InvalidInput(
                "Distributed ASM apply length mismatch".into(),
            ));
        }

        for yi in y.iter_mut() {
            *yi = S::zero();
        }

        let recv = state
            .comm_plan
            .exchange_values(&state.comm, state.layout.row_start, x)?;

        let mut rhs = vec![S::zero(); state.subdofs.len()];
        for &g in state.subdofs.iter() {
            if g >= state.layout.row_start && g < state.layout.row_end {
                let local_idx = g - state.layout.row_start;
                let sub_idx = *state
                    .sub_map
                    .get(&g)
                    .expect("subdomain map missing owned entry");
                rhs[sub_idx] = x[local_idx];
            }
        }
        for (peer, imports) in state.comm_plan.imports.iter().enumerate() {
            for (slot, &g) in imports.iter().enumerate() {
                let sub_idx = *state
                    .sub_map
                    .get(&g)
                    .expect("subdomain map missing import entry");
                rhs[sub_idx] = recv[peer][slot];
            }
        }

        let mut sol = vec![S::zero(); state.subdofs.len()];
        state.solver.solve(&rhs, &mut sol)?;

        for &g in state.subdofs.iter() {
            if g >= state.layout.row_start && g < state.layout.row_end {
                let local_idx = g - state.layout.row_start;
                let sub_idx = *state
                    .sub_map
                    .get(&g)
                    .expect("subdomain map missing owned entry");
                let weight = state
                    .weights
                    .as_ref()
                    .and_then(|w| w.get(local_idx))
                    .copied()
                    .unwrap_or(1.0);
                y[local_idx] = weight * sol[sub_idx];
            }
        }

        if self.mode == AsmMode::ASM {
            let mut send = vec![Vec::<S>::new(); state.comm.size()];
            for (peer, imports) in state.comm_plan.imports.iter().enumerate() {
                if peer == state.comm.rank() {
                    continue;
                }
                let mut payload = Vec::with_capacity(imports.len());
                for &g in imports {
                    let sub_idx = *state
                        .sub_map
                        .get(&g)
                        .expect("subdomain map missing ASM import contribution");
                    payload.push(sol[sub_idx]);
                }
                send[peer] = payload;
            }
            let recv = alltoallv_scalar(&state.comm, &send)?;
            for (peer, exported) in state.comm_plan.exports.iter().enumerate() {
                for (slot, &g) in exported.iter().enumerate() {
                    if g >= state.layout.row_start && g < state.layout.row_end {
                        let local_idx = g - state.layout.row_start;
                        let weight = state
                            .weights
                            .as_ref()
                            .and_then(|w| w.get(local_idx))
                            .copied()
                            .unwrap_or(1.0);
                        y[local_idx] += weight * recv[peer][slot];
                    }
                }
            }
        }
        if let Some(coarse) = state.coarse.as_ref() {
            coarse_additive_correction(coarse, &state.comm, x, y)?;
        }

        Ok(())
    }

    fn supports_numeric_update(&self) -> bool {
        self.state
            .as_ref()
            .map(|s| s.solver.supports_numeric_update())
            .unwrap_or(false)
    }

    fn update_numeric(&mut self, op: &dyn LinOp<S = S>) -> Result<(), KError> {
        let state = self.state.as_mut().ok_or_else(|| {
            KError::InvalidInput("Distributed ASM preconditioner not setup".into())
        })?;
        if Some(op.structure_id()) != self.last_sid {
            return self.update_symbolic(op);
        }

        let local_csr = materialize_local_csr(op)?;
        let ownership = build_ownership(&state.layout, state.comm.size());
        let missing: Vec<usize> = state
            .subdofs
            .iter()
            .copied()
            .filter(|g| *g < state.layout.row_start || *g >= state.layout.row_end)
            .collect();
        let remote_rows = request_remote_rows(
            &state.comm,
            &ownership,
            state.layout.row_start,
            state.layout.row_end,
            &local_csr,
            &missing,
        )?;
        let sub_csr = Arc::new(build_subdomain_csr(
            &state.subdofs,
            state.layout.row_start,
            state.layout.row_end,
            &local_csr,
            &remote_rows,
        )?);

        if state.solver.supports_numeric_update() {
            state.solver.update_numeric(&sub_csr)?;
        } else {
            state.solver.setup(&sub_csr)?;
        }

        state.local_csr = local_csr;
        state.sub_csr = sub_csr;
        if let Some(coarse) = state.coarse.as_mut() {
            coarse.root_matrix = build_root_gather_coarse_matrix(
                coarse.strategy,
                &state.comm,
                &state.local_csr,
                &coarse.ownership,
            )?;
        }
        self.last_vid = Some(op.values_id());
        Ok(())
    }

    fn update_symbolic(&mut self, op: &dyn LinOp<S = S>) -> Result<(), KError> {
        self.setup(op)
    }

    fn required_format(&self) -> OpFormat {
        OpFormat::Csr
    }

    fn distributed_support(&self) -> PcDistributedSupport {
        PcDistributedSupport::Distributed
    }
}

#[cfg(feature = "mpi")]
fn materialize_local_csr(op: &dyn LinOp<S = S>) -> Result<Arc<CsrMatrix<S>>, KError> {
    if let Some(dist) = op.as_any().downcast_ref::<DistCsrOp>() {
        return Ok(Arc::new(dist.local_matrix()));
    }
    if let Some(csr) = op.as_any().downcast_ref::<CsrMatrix<S>>() {
        return Ok(Arc::new(csr.clone()));
    }
    #[cfg(not(feature = "complex"))]
    {
        let mat = materialize_linop_with_hint(op, FormatHint::Csr, 0.0)?;
        if let Some(csr) = mat.as_any().downcast_ref::<CsrMatrix<f64>>() {
            return Ok(Arc::new(csr.clone()));
        }
    }
    Err(KError::Unsupported(
        "Distributed ASM requires a CSR materializable operator".into(),
    ))
}

#[cfg(feature = "mpi")]
fn dist_layout_from_op(op: &dyn LinOp<S = S>) -> Option<DistLayout> {
    if let Some(layout) = op.dist_layout() {
        return Some(layout.clone());
    }
    if let Some(dist) = op.as_any().downcast_ref::<DistCsrOp>() {
        return Some(DistLayout {
            global_rows: dist.n_global,
            global_cols: dist.n_global,
            row_start: dist.row_start,
            row_end: dist.row_end,
            col_start: dist.row_start,
            col_end: dist.row_end,
        });
    }
    None
}

#[cfg(feature = "mpi")]
fn build_ownership(layout: &DistLayout, size: usize) -> Vec<(usize, usize)> {
    let mut ownership = Vec::with_capacity(size);
    for rank in 0..size {
        ownership.push(contiguous_partition(layout.global_rows, rank, size));
    }
    ownership
}

#[cfg(feature = "mpi")]
fn build_overlap_set(
    local: &CsrMatrix<S>,
    layout: &DistLayout,
    ownership: &[(usize, usize)],
    overlap: usize,
    comm: &UniverseComm,
) -> Result<(Vec<usize>, HashMap<usize, RemoteRow>), KError> {
    let mut subdofs: HashSet<usize> = (layout.row_start..layout.row_end).collect();
    let mut frontier: Vec<usize> = (layout.row_start..layout.row_end).collect();
    let mut remote_rows: HashMap<usize, RemoteRow> = HashMap::new();

    for _ in 0..overlap {
        if frontier.is_empty() {
            break;
        }
        let mut to_request = Vec::new();
        for &g in &frontier {
            if g < layout.row_start || g >= layout.row_end {
                if !remote_rows.contains_key(&g) {
                    to_request.push(g);
                }
            }
        }
        if !to_request.is_empty() {
            let fetched = request_remote_rows(
                comm,
                ownership,
                layout.row_start,
                layout.row_end,
                local,
                &to_request,
            )?;
            remote_rows.extend(fetched);
        }

        let mut next = Vec::new();
        for &g in &frontier {
            let cols: Vec<usize> = if g >= layout.row_start && g < layout.row_end {
                let local_row = g - layout.row_start;
                let start = local.row_ptr()[local_row];
                let end = local.row_ptr()[local_row + 1];
                local.col_idx()[start..end].to_vec()
            } else if let Some(row) = remote_rows.get(&g) {
                row.cols.clone()
            } else {
                Vec::new()
            };

            for col in cols {
                if subdofs.insert(col) {
                    next.push(col);
                }
            }
        }
        frontier = next;
    }

    let mut subdofs: Vec<usize> = subdofs.into_iter().collect();
    subdofs.sort_unstable();

    Ok((subdofs, remote_rows))
}

#[cfg(feature = "mpi")]
fn build_comm_plan(
    comm: &UniverseComm,
    ownership: &[(usize, usize)],
    subdofs: &[usize],
) -> Result<CommPlan, KError> {
    let size = comm.size();
    let rank = comm.rank();

    let mut imports = vec![Vec::<usize>::new(); size];
    for &g in subdofs {
        let owner = owner_of(g, ownership);
        if owner != rank {
            imports[owner].push(g);
        }
    }

    let mut send = vec![Vec::<u64>::new(); size];
    for (peer, list) in imports.iter().enumerate() {
        if peer == rank {
            continue;
        }
        send[peer] = list.iter().map(|&g| g as u64).collect();
    }

    let recv = alltoallv_u64(comm, &send)?;
    let mut exports = vec![Vec::<usize>::new(); size];
    for (peer, data) in recv.iter().enumerate() {
        if peer == rank {
            continue;
        }
        exports[peer] = data.iter().map(|&g| g as usize).collect();
    }

    let import_locs = imports
        .iter()
        .map(|list| vec![0usize; list.len()])
        .collect();

    Ok(CommPlan {
        imports,
        exports,
        import_locs,
    })
}

#[cfg(feature = "mpi")]
fn build_ras_weights(
    layout: &DistLayout,
    comm_plan: &CommPlan,
    weighting: Weighting,
) -> Option<Vec<R>> {
    if !matches!(weighting, Weighting::Uniform) {
        return None;
    }
    let n_local = layout.row_end - layout.row_start;
    if n_local == 0 {
        return Some(Vec::new());
    }
    let mut cover_count = vec![1usize; n_local];
    for export in &comm_plan.exports {
        for &g in export {
            if g >= layout.row_start && g < layout.row_end {
                cover_count[g - layout.row_start] += 1;
            }
        }
    }
    Some(
        cover_count
            .into_iter()
            .map(|count| 1.0 / (count as R))
            .collect(),
    )
}

#[cfg(feature = "mpi")]
fn owner_of(g: usize, ownership: &[(usize, usize)]) -> usize {
    let mut lo = 0usize;
    let mut hi = ownership.len().saturating_sub(1);
    while lo <= hi {
        let mid = (lo + hi) / 2;
        let (start, end) = ownership[mid];
        if g < start {
            if mid == 0 {
                break;
            }
            hi = mid - 1;
        } else if g >= end {
            lo = mid + 1;
        } else {
            return mid;
        }
    }
    lo.min(ownership.len().saturating_sub(1))
}

#[cfg(feature = "mpi")]
fn build_coarse_space(
    requested: DistCoarseStrategy,
    comm: &UniverseComm,
    layout: &DistLayout,
    local_csr: &CsrMatrix<S>,
    ownership: &[(usize, usize)],
) -> Result<Option<DistributedAsmCoarse>, KError> {
    let strategy = match requested {
        DistCoarseStrategy::None => return Ok(None),
        DistCoarseStrategy::RootGather => DistCoarseStrategy::RootGather,
        DistCoarseStrategy::LocalPrototype | DistCoarseStrategy::SuperLuDist => {
            log::warn!(
                "Distributed ASM coarse strategy {} currently routes to root_gather.",
                requested
            );
            DistCoarseStrategy::RootGather
        }
    };
    let _ = layout;
    let root_matrix = build_root_gather_coarse_matrix(strategy, comm, local_csr, ownership)?;
    Ok(Some(DistributedAsmCoarse {
        strategy,
        ownership: ownership.to_vec(),
        root_matrix,
    }))
}

#[cfg(feature = "mpi")]
fn build_root_gather_coarse_matrix(
    strategy: DistCoarseStrategy,
    comm: &UniverseComm,
    local_csr: &CsrMatrix<S>,
    ownership: &[(usize, usize)],
) -> Result<Option<Vec<Vec<S>>>, KError> {
    if !matches!(strategy, DistCoarseStrategy::RootGather) {
        return Ok(None);
    }
    let n_coarse = comm.size();
    let mut local_row = vec![S::zero(); n_coarse];
    for local_row_idx in 0..local_csr.nrows() {
        let start = local_csr.row_ptr()[local_row_idx];
        let end = local_csr.row_ptr()[local_row_idx + 1];
        for slot in start..end {
            let col = local_csr.col_idx()[slot];
            let owner = owner_of(col, ownership);
            local_row[owner] += local_csr.values()[slot];
        }
    }

    let mut send = vec![Vec::<S>::new(); comm.size()];
    send[0] = local_row;
    let gathered = alltoallv_scalar(comm, &send)?;
    if comm.rank() != 0 {
        return Ok(None);
    }
    let mut coarse = vec![vec![S::zero(); n_coarse]; n_coarse];
    for rank in 0..comm.size() {
        if gathered[rank].len() != n_coarse {
            return Err(KError::InvalidInput(
                "coarse assembly gather produced an unexpected payload size".into(),
            ));
        }
        coarse[rank].copy_from_slice(&gathered[rank]);
    }
    Ok(Some(coarse))
}

#[cfg(feature = "mpi")]
fn coarse_additive_correction(
    coarse: &DistributedAsmCoarse,
    comm: &UniverseComm,
    x: &[S],
    y: &mut [S],
) -> Result<(), KError> {
    if !matches!(coarse.strategy, DistCoarseStrategy::RootGather) {
        return Ok(());
    }

    let local_sum = x.iter().copied().fold(S::zero(), |acc, v| acc + v);
    let mut send = vec![Vec::<S>::new(); comm.size()];
    send[0] = vec![local_sum];
    let gathered_rhs = alltoallv_scalar(comm, &send)?;

    let z_local = if comm.rank() == 0 {
        let mat = coarse.root_matrix.as_ref().ok_or_else(|| {
            KError::InvalidInput("root-gather coarse matrix missing on rank 0".into())
        })?;
        let mut rhs = vec![S::zero(); comm.size()];
        for r in 0..comm.size() {
            if gathered_rhs[r].len() != 1 {
                return Err(KError::InvalidInput(
                    "coarse rhs gather produced an unexpected payload size".into(),
                ));
            }
            rhs[r] = gathered_rhs[r][0];
        }
        let sol = dense_solve_with_diagonal_shift(mat, &rhs, 1e-12)?;

        let mut scatter = vec![Vec::<S>::new(); comm.size()];
        for r in 0..comm.size() {
            scatter[r] = vec![sol[r]];
        }
        let recv = alltoallv_scalar(comm, &scatter)?;
        if recv[0].len() != 1 {
            return Err(KError::InvalidInput(
                "coarse scatter to root produced an unexpected payload size".into(),
            ));
        }
        recv[0][0]
    } else {
        let scatter = vec![Vec::<S>::new(); comm.size()];
        let recv = alltoallv_scalar(comm, &scatter)?;
        if recv[0].len() != 1 {
            return Err(KError::InvalidInput(
                "coarse scatter receive produced an unexpected payload size".into(),
            ));
        }
        recv[0][0]
    };

    for yi in y.iter_mut() {
        *yi += z_local;
    }
    Ok(())
}

#[cfg(feature = "mpi")]
fn dense_solve_with_diagonal_shift(
    a: &[Vec<S>],
    b: &[S],
    diag_shift: f64,
) -> Result<Vec<S>, KError> {
    let n = a.len();
    if b.len() != n || a.iter().any(|row| row.len() != n) {
        return Err(KError::InvalidInput(
            "coarse dense solve requires a square matrix and matching rhs".into(),
        ));
    }
    let mut m = a.to_vec();
    let mut rhs = b.to_vec();
    let shift = S::from_real(diag_shift);
    for i in 0..n {
        m[i][i] += shift;
    }

    for k in 0..n {
        let mut piv = k;
        let mut piv_abs = m[k][k].abs();
        for row in (k + 1)..n {
            let cand = m[row][k].abs();
            if cand > piv_abs {
                piv = row;
                piv_abs = cand;
            }
        }
        if piv_abs <= 1e-20 {
            return Err(KError::InvalidInput(
                "coarse dense solve singular pivot".into(),
            ));
        }
        if piv != k {
            m.swap(k, piv);
            rhs.swap(k, piv);
        }
        for row in (k + 1)..n {
            let factor = m[row][k] / m[k][k];
            if factor.abs() == 0.0 {
                continue;
            }
            for col in k..n {
                m[row][col] -= factor * m[k][col];
            }
            rhs[row] -= factor * rhs[k];
        }
    }

    let mut x = vec![S::zero(); n];
    for i in (0..n).rev() {
        let mut sum = rhs[i];
        for j in (i + 1)..n {
            sum -= m[i][j] * x[j];
        }
        x[i] = sum / m[i][i];
    }
    Ok(x)
}

#[cfg(feature = "mpi")]
enum SubdomainSolver {
    Csr(Box<dyn Preconditioner>),
    #[cfg(all(feature = "backend-faer", feature = "legacy-pc-bridge"))]
    Dense(Box<dyn Preconditioner>),
}

#[cfg(feature = "mpi")]
impl std::fmt::Debug for SubdomainSolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self {
            SubdomainSolver::Csr(_) => f
                .debug_struct("SubdomainSolver")
                .field("backend", &"csr")
                .finish(),
            #[cfg(all(feature = "backend-faer", feature = "legacy-pc-bridge"))]
            SubdomainSolver::Dense(_) => f
                .debug_struct("SubdomainSolver")
                .field("backend", &"dense")
                .finish(),
        }
    }
}

#[cfg(feature = "mpi")]
impl SubdomainSolver {
    fn new(block_solver: AsmBlockSolver, inner_pc: AsmInnerPc) -> Result<Self, KError> {
        let conditioning = ConditioningOptions::default();
        match inner_pc {
            AsmInnerPc::Jacobi => {
                let pc = build_jacobi()?;
                Ok(match block_solver {
                    AsmBlockSolver::LuDense => {
                        #[cfg(all(feature = "backend-faer", feature = "legacy-pc-bridge"))]
                        {
                            SubdomainSolver::Dense(pc)
                        }
                        #[cfg(not(all(feature = "backend-faer", feature = "legacy-pc-bridge")))]
                        {
                            SubdomainSolver::Csr(pc)
                        }
                    }
                    AsmBlockSolver::Csr => SubdomainSolver::Csr(pc),
                })
            }
            AsmInnerPc::Ilu0 => {
                #[cfg(feature = "complex")]
                {
                    return Err(KError::Unsupported(
                        "ILU0 is not available for complex ASM blocks".into(),
                    ));
                }
                #[cfg(not(feature = "complex"))]
                {
                    if matches!(block_solver, AsmBlockSolver::LuDense) {
                        return Err(KError::Unsupported(
                            "dense ASM block solver does not support ilu0; use pc_asm_block_solver=csr"
                                .into(),
                        ));
                    }
                    let pc = build_ilu0_with_conditioning(conditioning)?;
                    Ok(SubdomainSolver::Csr(pc))
                }
            }
            AsmInnerPc::Ilut { drop_tol, max_fill } => {
                #[cfg(feature = "complex")]
                {
                    return Err(KError::Unsupported(
                        "ILUT is not available for complex ASM blocks".into(),
                    ));
                }
                #[cfg(not(feature = "complex"))]
                {
                    if matches!(block_solver, AsmBlockSolver::LuDense) {
                        return Err(KError::Unsupported(
                            "dense ASM block solver does not support ilut; use pc_asm_block_solver=csr"
                                .into(),
                        ));
                    }
                    let pc = build_ilut_with_conditioning(drop_tol, max_fill, None, conditioning)?;
                    Ok(SubdomainSolver::Csr(pc))
                }
            }
            AsmInnerPc::Ilutp {
                drop_tol,
                max_fill,
                perm_tol,
            } => {
                #[cfg(feature = "complex")]
                {
                    return Err(KError::Unsupported(
                        "ILUTP is not available for complex ASM blocks".into(),
                    ));
                }
                #[cfg(not(feature = "complex"))]
                {
                    if matches!(block_solver, AsmBlockSolver::Csr) {
                        return Err(KError::Unsupported(
                            "ILUTP requires dense ASM blocks; use pc_asm_block_solver=ludense"
                                .into(),
                        ));
                    }
                    let pc = build_ilutp_with_conditioning(
                        max_fill,
                        drop_tol,
                        perm_tol,
                        None,
                        conditioning,
                    )?;
                    #[cfg(all(feature = "backend-faer", feature = "legacy-pc-bridge"))]
                    {
                        Ok(SubdomainSolver::Dense(pc))
                    }
                    #[cfg(not(all(feature = "backend-faer", feature = "legacy-pc-bridge")))]
                    {
                        let _ = pc;
                        Err(KError::Unsupported(
                            "ILUTP requires features \"backend-faer\" and \"legacy-pc-bridge\""
                                .into(),
                        ))
                    }
                }
            }
        }
    }

    fn setup(&mut self, mat: &Arc<CsrMatrix<S>>) -> Result<(), KError> {
        match self {
            SubdomainSolver::Csr(pc) => {
                let op = CsrOp::new(mat.clone());
                pc.setup(&op)
            }
            #[cfg(all(feature = "backend-faer", feature = "legacy-pc-bridge"))]
            SubdomainSolver::Dense(pc) => {
                let dense = mat.to_dense()?;
                let op = DenseOp::new(Arc::new(dense));
                pc.setup(&op)
            }
        }
    }

    fn solve(&self, rhs: &[S], x: &mut [S]) -> Result<(), KError> {
        match self {
            SubdomainSolver::Csr(pc) => pc.apply(PcSide::Left, rhs, x),
            #[cfg(all(feature = "backend-faer", feature = "legacy-pc-bridge"))]
            SubdomainSolver::Dense(pc) => pc.apply(PcSide::Left, rhs, x),
        }
    }

    fn update_numeric(&mut self, mat: &Arc<CsrMatrix<S>>) -> Result<(), KError> {
        match self {
            SubdomainSolver::Csr(pc) => {
                let op = CsrOp::new(mat.clone());
                pc.update_numeric(&op)
            }
            #[cfg(all(feature = "backend-faer", feature = "legacy-pc-bridge"))]
            SubdomainSolver::Dense(pc) => {
                let dense = mat.to_dense()?;
                let op = DenseOp::new(Arc::new(dense));
                pc.update_numeric(&op)
            }
        }
    }

    fn supports_numeric_update(&self) -> bool {
        match self {
            SubdomainSolver::Csr(pc) => pc.supports_numeric_update(),
            #[cfg(all(feature = "backend-faer", feature = "legacy-pc-bridge"))]
            SubdomainSolver::Dense(pc) => pc.supports_numeric_update(),
        }
    }
}
