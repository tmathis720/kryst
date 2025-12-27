#[cfg(all(feature = "mpi", not(feature = "complex")))]
use crate::algebra::prelude::*;
#[cfg(all(feature = "mpi", not(feature = "complex")))]
use crate::error::KError;
#[cfg(all(feature = "mpi", not(feature = "complex")))]
use crate::matrix::convert::materialize_linop_with_hint;
use crate::matrix::format::FormatHint;
#[cfg(all(feature = "mpi", not(feature = "complex")))]
use crate::matrix::format::OpFormat;
use crate::matrix::op::{DistLayout, LinOp, StructureId, ValuesId};
#[cfg(all(feature = "mpi", not(feature = "complex")))]
use crate::matrix::sparse::CsrMatrix;
#[cfg(all(feature = "mpi", not(feature = "complex")))]
use crate::matrix::DistCsrOp;
#[cfg(all(feature = "mpi", not(feature = "complex")))]
use crate::parallel::{Comm, UniverseComm, contiguous_partition};
#[cfg(all(feature = "mpi", not(feature = "complex")))]
use super::{AsmBlockSolver, AsmMode, Weighting};
#[cfg(all(feature = "mpi", not(feature = "complex")))]
use crate::preconditioner::ilu_csr::{
    IluCsr, IluCsrConfig, IluKind, PivotStrategy, ReorderingOptions,
};
#[cfg(all(feature = "mpi", not(feature = "complex")))]
use crate::preconditioner::{PcSide, Preconditioner};
#[cfg(all(feature = "mpi", not(feature = "complex")))]
use crate::matrix::op::CsrOp;
#[cfg(all(feature = "mpi", not(feature = "complex")))]
use std::collections::{HashMap, HashSet};
#[cfg(all(feature = "mpi", not(feature = "complex")))]
use std::sync::Arc;

#[cfg(all(feature = "mpi", not(feature = "complex")))]
use super::comm_plan::{CommPlan, alltoallv_u64};
#[cfg(all(feature = "mpi", not(feature = "complex")))]
use super::subdomain::{RemoteRow, build_subdomain_csr, request_remote_rows};

#[cfg(all(feature = "mpi", not(feature = "complex")))]
#[derive(Debug)]
pub struct DistributedAsm {
    overlap: usize,
    subdomain_hint: Option<usize>,
    block_solver: AsmBlockSolver,
    mode: AsmMode,
    weighting: Weighting,
    state: Option<DistributedAsmState>,
    last_sid: Option<StructureId>,
    last_vid: Option<ValuesId>,
}

#[cfg(all(feature = "mpi", not(feature = "complex")))]
#[derive(Debug)]
struct DistributedAsmState {
    comm: UniverseComm,
    layout: DistLayout,
    local_csr: Arc<CsrMatrix<f64>>,
    subdofs: Vec<usize>,
    sub_map: HashMap<usize, usize>,
    comm_plan: CommPlan,
    sub_csr: Arc<CsrMatrix<f64>>,
    solver: SubdomainSolver,
}

#[cfg(all(feature = "mpi", not(feature = "complex")))]
impl DistributedAsm {
    pub fn new(
        overlap: usize,
        subdomain_hint: Option<usize>,
        block_solver: AsmBlockSolver,
        mode: AsmMode,
        weighting: Weighting,
    ) -> Self {
        Self {
            overlap,
            subdomain_hint,
            block_solver,
            mode,
            weighting,
            state: None,
            last_sid: None,
            last_vid: None,
        }
    }
}

#[cfg(all(feature = "mpi", not(feature = "complex")))]
impl Preconditioner for DistributedAsm {
    fn dims(&self) -> (usize, usize) {
        self.state
            .as_ref()
            .map(|s| (s.layout.row_end - s.layout.row_start, s.layout.row_end - s.layout.row_start))
            .unwrap_or((0, 0))
    }

    fn setup(&mut self, op: &dyn LinOp<S = S>) -> Result<(), KError> {
        if self.mode == AsmMode::ASM {
            return Err(KError::Unsupported(
                "Distributed ASM currently supports only RAS mode".into(),
            ));
        }
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

        let (subdofs, remote_rows) = build_overlap_set(
            &local_csr,
            &layout,
            &ownership,
            self.overlap,
            &comm,
        )?;
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
        let sub_map = subdofs
            .iter()
            .enumerate()
            .map(|(i, &g)| (g, i))
            .collect();

        let mut solver = SubdomainSolver::new(self.block_solver)?;
        solver.setup(&sub_csr)?;

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
                y[local_idx] = sol[sub_idx];
            }
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
        self.last_vid = Some(op.values_id());
        Ok(())
    }

    fn update_symbolic(&mut self, op: &dyn LinOp<S = S>) -> Result<(), KError> {
        self.setup(op)
    }

    fn required_format(&self) -> OpFormat {
        OpFormat::Csr
    }
}

#[cfg(all(feature = "mpi", not(feature = "complex")))]
fn materialize_local_csr(op: &dyn LinOp<S = S>) -> Result<Arc<CsrMatrix<f64>>, KError> {
    let mat = materialize_linop_with_hint(op, FormatHint::Csr, 0.0)?;
    if let Some(csr) = mat.as_any().downcast_ref::<CsrMatrix<f64>>() {
        return Ok(Arc::new(csr.clone()));
    }
    Err(KError::Unsupported(
        "Distributed ASM requires a CSR materializable operator".into(),
    ))
}

#[cfg(all(feature = "mpi", not(feature = "complex")))]
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

#[cfg(all(feature = "mpi", not(feature = "complex")))]
fn build_ownership(layout: &DistLayout, size: usize) -> Vec<(usize, usize)> {
    let mut ownership = Vec::with_capacity(size);
    for rank in 0..size {
        ownership.push(contiguous_partition(layout.global_rows, rank, size));
    }
    ownership
}

#[cfg(all(feature = "mpi", not(feature = "complex")))]
fn build_overlap_set(
    local: &CsrMatrix<f64>,
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

#[cfg(all(feature = "mpi", not(feature = "complex")))]
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

#[cfg(all(feature = "mpi", not(feature = "complex")))]
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

#[cfg(all(feature = "mpi", not(feature = "complex")))]
struct SubdomainSolver {
    ilu: IluCsr,
}

#[cfg(all(feature = "mpi", not(feature = "complex")))]
impl SubdomainSolver {
    fn new(kind: AsmBlockSolver) -> Result<Self, KError> {
        let _ = kind;
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
        Ok(Self {
            ilu: IluCsr::new_with_config(cfg),
        })
    }

    fn setup(&mut self, mat: &Arc<CsrMatrix<f64>>) -> Result<(), KError> {
        let op = CsrOp::new(mat.clone());
        self.ilu.setup(&op)
    }

    fn solve(&self, rhs: &[S], x: &mut [S]) -> Result<(), KError> {
        self.ilu.apply(PcSide::Left, rhs, x)
    }

    fn update_numeric(&mut self, mat: &Arc<CsrMatrix<f64>>) -> Result<(), KError> {
        let op = CsrOp::new(mat.clone());
        self.ilu.update_numeric(&op)
    }

    fn supports_numeric_update(&self) -> bool {
        self.ilu.supports_numeric_update()
    }
}
