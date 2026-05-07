#![cfg(all(feature = "complex", feature = "backend-faer"))]

use std::collections::{BTreeMap, BTreeSet};

use crate::algebra::bridge::BridgeScratch;
use crate::algebra::scalar::{KrystScalar, S};
use crate::error::KError;
use crate::matrix::sparse::CsrMatrix;
use crate::ops::kpc::KPreconditioner;
use crate::parallel::UniverseComm;
use crate::preconditioner::ilu_csr::{IluCsr, IluCsrConfig, IluKind};
use crate::preconditioner::{PcSide, Preconditioner};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OverlapRestriction {
    Asm,
    Ras,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OverlapIluDiagnostics {
    pub owned_start: usize,
    pub owned_end: usize,
    pub overlap: usize,
    pub subdomain_rows: Vec<usize>,
    pub ghost_rows: Vec<usize>,
    pub ghost_columns: Vec<usize>,
}

/// One-subdomain-per-rank overlapping ILU preconditioner for complex CSR examples.
///
/// The builder expands the owned row interval through graph-neighbor layers in the
/// global CSR graph, extracts the induced subdomain matrix, factors it with
/// [`IluCsr`], gathers the current distributed vector into the ghost entries needed
/// by the subdomain solve, and returns the owned portion of the correction.
pub struct OverlapIluPc {
    comm: UniverseComm,
    owned_start: usize,
    owned_end: usize,
    global_n: usize,
    subdomain_rows: Vec<usize>,
    owned_local_positions: Vec<usize>,
    ilu: IluCsr,
    restriction: OverlapRestriction,
    diagnostics: OverlapIluDiagnostics,
}

impl OverlapIluPc {
    pub fn setup_from_global_csr(
        global: &CsrMatrix<S>,
        comm: UniverseComm,
        owned_start: usize,
        owned_end: usize,
        overlap: usize,
        kind: IluKind,
        mut cfg: IluCsrConfig,
        restriction: OverlapRestriction,
    ) -> Result<Self, KError> {
        if global.nrows() != global.ncols() {
            return Err(KError::InvalidInput(format!(
                "overlapping ILU requires a square global matrix, got {}x{}",
                global.nrows(),
                global.ncols()
            )));
        }
        if owned_start > owned_end || owned_end > global.nrows() {
            return Err(KError::InvalidInput(format!(
                "invalid owned range [{owned_start}, {owned_end}) for global size {}",
                global.nrows()
            )));
        }

        cfg.kind = kind;
        let subdomain_rows =
            expand_owned_rows_by_graph_overlap(global, owned_start, owned_end, overlap);
        let (local, global_to_local) = extract_induced_subdomain(global, &subdomain_rows)?;
        let owned_local_positions = (owned_start..owned_end)
            .map(|g| {
                global_to_local.get(&g).copied().ok_or_else(|| {
                    KError::InvalidInput(format!(
                        "owned row {g} missing from overlapping ILU subdomain"
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        let mut ilu = IluCsr::new_with_config(cfg);
        ilu.setup(&local)?;

        let owned: BTreeSet<_> = (owned_start..owned_end).collect();
        let ghosts: Vec<_> = subdomain_rows
            .iter()
            .copied()
            .filter(|g| !owned.contains(g))
            .collect();
        let ghost_columns = ghosts.clone();
        let diagnostics = OverlapIluDiagnostics {
            owned_start,
            owned_end,
            overlap,
            subdomain_rows,
            ghost_rows: ghosts,
            ghost_columns,
        };

        Ok(Self {
            comm,
            owned_start,
            owned_end,
            global_n: global.nrows(),
            subdomain_rows: diagnostics.subdomain_rows.clone(),
            owned_local_positions,
            ilu,
            restriction,
            diagnostics,
        })
    }

    pub fn diagnostics(&self) -> &OverlapIluDiagnostics {
        &self.diagnostics
    }

    fn gather_owned_input(&self, x_owned: &[S]) -> Result<Vec<S>, KError> {
        let owned_n = self.owned_end - self.owned_start;
        if x_owned.len() != owned_n {
            return Err(KError::InvalidInput(format!(
                "overlapping ILU apply expected owned input length {owned_n}, got {}",
                x_owned.len()
            )));
        }
        let mut global_x = vec![S::zero(); self.global_n];
        global_x[self.owned_start..self.owned_end].copy_from_slice(x_owned);
        self.comm.allreduce_sum_scalars(&mut global_x);
        Ok(global_x)
    }
}

impl KPreconditioner for OverlapIluPc {
    type Scalar = S;

    fn dims(&self) -> (usize, usize) {
        (
            self.owned_end - self.owned_start,
            self.owned_end - self.owned_start,
        )
    }

    fn apply_s(
        &self,
        side: PcSide,
        x: &[S],
        y: &mut [S],
        scratch: &mut BridgeScratch,
    ) -> Result<(), KError> {
        let owned_n = self.owned_end - self.owned_start;
        if y.len() != owned_n {
            return Err(KError::InvalidInput(format!(
                "overlapping ILU apply expected owned output length {owned_n}, got {}",
                y.len()
            )));
        }
        let global_x = self.gather_owned_input(x)?;
        let local_x: Vec<_> = self.subdomain_rows.iter().map(|&g| global_x[g]).collect();
        let mut local_y = vec![S::zero(); self.subdomain_rows.len()];
        self.ilu.apply_s(side, &local_x, &mut local_y, scratch)?;
        match self.restriction {
            OverlapRestriction::Asm | OverlapRestriction::Ras => {
                for (dst, &local_pos) in y.iter_mut().zip(&self.owned_local_positions) {
                    *dst = local_y[local_pos];
                }
            }
        }
        Ok(())
    }
}

pub fn expand_owned_rows_by_graph_overlap(
    global: &CsrMatrix<S>,
    owned_start: usize,
    owned_end: usize,
    overlap: usize,
) -> Vec<usize> {
    let mut subdomain: BTreeSet<usize> = (owned_start..owned_end).collect();
    let mut frontier = subdomain.clone();
    for _ in 0..overlap {
        let mut next = BTreeSet::new();
        for row in frontier.iter().copied() {
            for nz in global.row_ptr()[row]..global.row_ptr()[row + 1] {
                let col = global.col_idx()[nz];
                if col < global.nrows() && subdomain.insert(col) {
                    next.insert(col);
                }
            }
        }
        frontier = next;
        if frontier.is_empty() {
            break;
        }
    }
    subdomain.into_iter().collect()
}

fn extract_induced_subdomain(
    global: &CsrMatrix<S>,
    subdomain_rows: &[usize],
) -> Result<(CsrMatrix<S>, BTreeMap<usize, usize>), KError> {
    let global_to_local: BTreeMap<usize, usize> = subdomain_rows
        .iter()
        .copied()
        .enumerate()
        .map(|(l, g)| (g, l))
        .collect();
    let mut row_ptr = Vec::with_capacity(subdomain_rows.len() + 1);
    let mut col_idx = Vec::new();
    let mut values = Vec::new();
    row_ptr.push(0);
    for &global_row in subdomain_rows {
        for nz in global.row_ptr()[global_row]..global.row_ptr()[global_row + 1] {
            let global_col = global.col_idx()[nz];
            if let Some(&local_col) = global_to_local.get(&global_col) {
                col_idx.push(local_col);
                values.push(global.values()[nz]);
            }
        }
        row_ptr.push(col_idx.len());
    }
    Ok((
        CsrMatrix::from_csr(
            subdomain_rows.len(),
            subdomain_rows.len(),
            row_ptr,
            col_idx,
            values,
        ),
        global_to_local,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parallel::NoComm;

    fn four_by_four_chain() -> CsrMatrix<S> {
        CsrMatrix::from_csr(
            4,
            4,
            vec![0, 2, 5, 8, 10],
            vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3],
            vec![
                S::from_real(4.0),
                S::from_real(-1.0),
                S::from_real(-1.0),
                S::from_real(4.0),
                S::from_real(-1.0),
                S::from_real(-1.0),
                S::from_real(4.0),
                S::from_real(-1.0),
                S::from_real(-1.0),
                S::from_real(4.0),
            ],
        )
    }

    fn owned_block_0_2(global: &CsrMatrix<S>) -> CsrMatrix<S> {
        CsrMatrix::from_csr(
            2,
            2,
            vec![0, 2, 4],
            vec![0, 1, 0, 1],
            vec![
                global.values()[0],
                global.values()[1],
                global.values()[2],
                global.values()[3],
            ],
        )
    }

    #[test]
    fn overlap_zero_matches_owned_block_ilu0_apply() {
        let global = four_by_four_chain();
        let mut cfg = IluCsrConfig::default();
        cfg.kind = IluKind::Ilu0;
        let mut local = IluCsr::new_with_config(cfg.clone());
        local
            .setup(&owned_block_0_2(&global))
            .expect("setup local ILU0");

        let pc = OverlapIluPc::setup_from_global_csr(
            &global,
            UniverseComm::NoComm(NoComm),
            0,
            2,
            0,
            IluKind::Ilu0,
            cfg,
            OverlapRestriction::Ras,
        )
        .expect("setup overlap ILU0");

        let x = vec![S::from_real(1.0), S::from_real(2.0)];
        let mut y_local = vec![S::zero(); 2];
        let mut y_overlap = vec![S::zero(); 2];
        let mut scratch = BridgeScratch::default();
        local
            .apply_s(PcSide::Right, &x, &mut y_local, &mut scratch)
            .expect("apply local ILU0");
        pc.apply_s(PcSide::Right, &x, &mut y_overlap, &mut scratch)
            .expect("apply overlap ILU0");
        for (a, b) in y_local.iter().zip(&y_overlap) {
            assert!((*a - *b).abs() < 1e-12);
        }
    }

    #[test]
    fn overlap_one_adds_ghost_rows_and_columns() {
        let global = four_by_four_chain();
        let cfg = IluCsrConfig::default();
        let pc = OverlapIluPc::setup_from_global_csr(
            &global,
            UniverseComm::NoComm(NoComm),
            0,
            2,
            1,
            IluKind::Ilu0,
            cfg,
            OverlapRestriction::Ras,
        )
        .expect("setup overlap ILU0");
        let diag = pc.diagnostics();
        assert_eq!(diag.subdomain_rows, vec![0, 1, 2]);
        assert_eq!(diag.ghost_rows, vec![2]);
        assert_eq!(diag.ghost_columns, vec![2]);
    }
}
