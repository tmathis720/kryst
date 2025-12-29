//! Block GMRES family drivers.

#[allow(unused_imports)]
use crate::algebra::blas::{dot_conj, nrm2};
#[allow(unused_imports)]
use crate::algebra::prelude::*;
use crate::algebra::prelude::S;
use crate::context::ksp_context::Workspace;
use crate::error::KError;
use crate::parallel::UniverseComm;
use crate::preconditioner::{PcSide, Preconditioner};
use crate::solver::LinearSolver;
use crate::utils::convergence::{ConvergedReason, Convergence, SolveStats, SolverCounters};
use std::any::Any;

use super::BlockKrylovOptions;
use super::block_vec::BlockVec;

#[cfg(feature = "backend-faer")]
use crate::parallel::{global_dot_conj, global_nrm2, global_nrm2_many};
#[cfg(feature = "backend-faer")]
use faer::linalg::solvers::{FullPivLu, SolveCore};
#[cfg(feature = "backend-faer")]
use faer::{Conj, MatMut, MatRef};
#[cfg(feature = "backend-faer")]
use super::arnoldi::block_arnoldi_step;

/// Block GMRES solver for multiple right-hand sides.
pub struct BlockGmresSolver {
    pub options: BlockKrylovOptions,
}

// Conjugation helper for building normal equations:
#[cfg(feature = "complex")]
#[inline]
fn conj_s(x: S) -> S { x.conj() }

#[cfg(not(feature = "complex"))]
#[inline]
fn conj_s(x: S) -> S { x }

impl BlockGmresSolver {
    pub fn new(options: BlockKrylovOptions) -> Self {
        Self { options }
    }
}

impl LinearSolver for BlockGmresSolver {
    type Error = KError;

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn solve(
        &mut self,
        a: &dyn crate::matrix::op::LinOp<S = S>,
        pc: Option<&mut dyn Preconditioner>,
        b: &[S],
        x: &mut [S],
        pc_side: PcSide,
        comm: &UniverseComm,
        monitors: Option<&[Box<dyn Fn(usize, f64) + Send + Sync>]>,
        work: Option<&mut crate::context::ksp_context::Workspace>,
    ) -> Result<SolveStats<f64>, Self::Error> {
        #[cfg(not(feature = "backend-faer"))]
        {
            let _ = (a, pc, b, x, pc_side, comm, monitors, work);
            return Err(KError::Unsupported(
                "block GMRES requires the backend-faer feature",
            ));
        }
        #[cfg(feature = "backend-faer")]
        {
            if pc.is_some() {
                return Err(KError::Unsupported(
                    "block GMRES preconditioning is not implemented",
                ));
            }
            if !matches!(pc_side, PcSide::Left) {
                return Err(KError::Unsupported(
                    "block GMRES currently supports only left preconditioning",
                ));
            }
            let (nrows, ncols) = a.dims();
            if nrows != ncols {
                return Err(KError::InvalidInput(
                    "block GMRES requires a square operator".into(),
                ));
            }
            let p = self.options.block_size;
            if p == 0 {
                return Err(KError::InvalidInput(
                    "block GMRES requires a positive block size".into(),
                ));
            }
            let expected_len = ncols.saturating_mul(p);
            if !(b.len() == expected_len || (p == 1 && b.len() == ncols)) {
                return Err(KError::InvalidInput(
                    "block GMRES expects b to be column-major with block_size columns".into(),
                ));
            }
            if !(x.len() == expected_len || (p == 1 && x.len() == ncols)) {
                return Err(KError::InvalidInput(
                    "block GMRES expects x to be column-major with block_size columns".into(),
                ));
            }
            let mons = monitors.unwrap_or(&[]);
            let mut local_ws = Workspace::default();
            let work = work.unwrap_or(&mut local_ws);

            let mut b_block = BlockVec::new(ncols, p);
            fill_block_from_slice(&mut b_block, b)?;
            let mut x_block = BlockVec::new(ncols, p);
            fill_block_from_slice(&mut x_block, x)?;

            let mut r_block = BlockVec::new(ncols, p);
            let mut ax_block = BlockVec::new(ncols, p);
            compute_residual(a, &b_block, &x_block, &mut r_block, &mut ax_block);

            let bnorm = block_norm_max(&b_block, comm);
            let mut rnorm = block_norm_max(&r_block, comm);
            let mut iterations = 0usize;
            let mut reason = ConvergedReason::Continued;
            let mut counters = SolverCounters::default();
            let conv = Convergence::new(
                self.options.rtol,
                self.options.atol,
                self.options.dtol,
                self.options.max_iters,
            );
            let (stop_reason, mut stats) = conv.check(rnorm, bnorm, iterations);
            reason = stop_reason;
            if reason != ConvergedReason::Continued {
                write_block_to_slice(&x_block, x)?;
                stats.counters = counters;
                return Ok(stats);
            }

            let restart = self.options.restart_blocks.max(1);
            let mut basis: Vec<BlockVec> = Vec::with_capacity(restart + 1);
            let total_rows = (restart + 1) * p;
            let total_cols = restart * p;
            let mut h_full: Vec<S> = vec![0.0.into(); total_rows * total_cols];

            while iterations < self.options.max_iters {
                let mut v0 = r_block.clone();
                let beta = block_qr(&mut v0, comm, work)?;
                basis.clear();
                basis.push(v0);
                for val in &mut h_full {
                    *val = 0.0.into();
                }

                let mut x_cycle = x_block.clone();
                let mut w_block = BlockVec::new(ncols, p);

                for j in 0..restart {
                    let vj = &basis[j];
                    for col in 0..p {
                        let vj_col = vj.col(col);
                        let w_col = w_block.col_mut(col);
                        a.matvec(vj_col, w_col);
                    }
                    let arnoldi = block_arnoldi_step(&basis, &mut w_block, comm, work, self.options.max_cond)?;

                    let cols_h = (j + 1) * p;
                    let rows_h = (j + 2) * p;
                    for (block_idx, block_coeffs) in arnoldi.coeffs.chunks(p * p).enumerate() {
                        let row_offset = block_idx * p;
                        for row in 0..p {
                            for col in 0..p {
                                let value = block_coeffs[row * p + col];
                                let row_idx = row_offset + row;
                                let col_idx = j * p + col;
                                h_full[row_idx + col_idx * total_rows] = value;
                            }
                        }
                    }
                    let row_offset = (j + 1) * p;
                    for row in 0..p {
                        for col in 0..p {
                            let value = arnoldi.r_block[row * p + col];
                            let row_idx = row_offset + row;
                            let col_idx = j * p + col;
                            h_full[row_idx + col_idx * total_rows] = value;
                        }
                    }
                    basis.push(w_block.clone());

                    let h_slice: Vec<S> = extract_h(&h_full, total_rows, rows_h, cols_h);
                    let g_slice: Vec<S> = build_g(rows_h, p, &beta);
                    let y_slice: Vec<S> = solve_normal_eq(&h_slice, rows_h, cols_h, &g_slice, p)?;

                    update_solution(
                        &mut x_cycle,
                        &x_block,
                        &basis,
                        &y_slice,
                        cols_h,
                    );

                    compute_residual(a, &b_block, &x_cycle, &mut r_block, &mut ax_block);
                    rnorm = block_norm_max(&r_block, comm);
                    iterations += 1;
                    for m in mons {
                        m(iterations, rnorm);
                    }
                    let (iter_reason, iter_stats) = conv.check(rnorm, bnorm, iterations);
                    reason = iter_reason;
                    if reason != ConvergedReason::Continued {
                        x_block = x_cycle;
                        stats = iter_stats;
                        stats.counters = counters;
                        write_block_to_slice(&x_block, x)?;
                        return Ok(stats);
                    }
                    if iterations >= self.options.max_iters {
                        break;
                    }
                }

                x_block = x_cycle;
                compute_residual(a, &b_block, &x_block, &mut r_block, &mut ax_block);
                rnorm = block_norm_max(&r_block, comm);
                let (iter_reason, iter_stats) = conv.check(rnorm, bnorm, iterations);
                reason = iter_reason;
                if reason != ConvergedReason::Continued {
                    stats = iter_stats;
                    stats.counters = counters;
                    write_block_to_slice(&x_block, x)?;
                    return Ok(stats);
                }
            }

            let mut stats = SolveStats::new(iterations, rnorm, ConvergedReason::DivergedMaxIts);
            stats.counters = counters;
            write_block_to_slice(&x_block, x)?;
            Ok(stats)
        }
    }
}

#[cfg(feature = "backend-faer")]
fn fill_block_from_slice(block: &mut BlockVec, data: &[S]) -> Result<(), KError> {
    let n = block.nrows();
    let p = block.ncols();
    if data.len() == n {
        if p != 1 {
            return Err(KError::InvalidInput(
                "block GMRES expects a full block for block_size > 1".into(),
            ));
        }
        block.as_mut_slice().copy_from_slice(data);
        return Ok(());
    }
    if data.len() != n * p {
        return Err(KError::InvalidInput(
            "block GMRES expects column-major block storage".into(),
        ));
    }
    block.as_mut_slice().copy_from_slice(data);
    Ok(())
}

#[cfg(feature = "backend-faer")]
fn write_block_to_slice(block: &BlockVec, data: &mut [S]) -> Result<(), KError> {
    let n = block.nrows();
    let p = block.ncols();
    if data.len() == n && p == 1 {
        data.copy_from_slice(block.as_slice());
        return Ok(());
    }
    if data.len() != n * p {
        return Err(KError::InvalidInput(
            "block GMRES expects column-major block storage".into(),
        ));
    }
    data.copy_from_slice(block.as_slice());
    Ok(())
}

#[cfg(feature = "backend-faer")]
fn compute_residual(
    a: &dyn crate::matrix::op::LinOp<S = S>,
    b: &BlockVec,
    x: &BlockVec,
    r: &mut BlockVec,
    ax: &mut BlockVec,
) {
    let p = b.ncols();
    let n = b.nrows();
    for col in 0..p {
        let xcol = x.col(col);
        let axcol = ax.col_mut(col);
        a.matvec(xcol, axcol);
    }
    for col in 0..p {
        let bcol = b.col(col);
        let axcol = ax.col(col);
        let rcol = r.col_mut(col);
        for i in 0..n {
            rcol[i] = bcol[i] - axcol[i];
        }
    }
}

#[cfg(feature = "backend-faer")]
fn block_norm_max(block: &BlockVec, comm: &UniverseComm) -> f64 {
    let p = block.ncols();
    let mut cols: Vec<&[S]> = Vec::with_capacity(p);
    for col in 0..p {
        cols.push(block.col(col));
    }
    let norms = global_nrm2_many(comm, &cols);
    norms
        .into_iter()
        .fold(0.0_f64, |acc, val| acc.max(val))
}

#[cfg(feature = "backend-faer")]
fn block_qr(
    block: &mut BlockVec,
    comm: &UniverseComm,
    work: &mut Workspace,
) -> Result<Vec<S>, KError> {
    let p = block.ncols();
    let n = block.nrows();
    let mut r: Vec<S> = vec![0.0.into(); p * p];
    work.blk_scratch.resize(n, 0.0.into());
    let col_buf = &mut work.blk_scratch[..n];
    for j in 0..p {
        col_buf.copy_from_slice(block.col(j));
        for i in 0..j {
            let qi = block.col(i);
            let dot = global_dot_conj(comm, qi, &col_buf[..]);
            r[i * p + j] = dot;
            for (buf, &qi_val) in col_buf.iter_mut().zip(qi.iter()) {
                *buf -= dot * qi_val;
            }
        }
        let norm = global_nrm2(comm, &col_buf[..]);
        if norm <= 0.0 {
            return Err(KError::FactorError(
                "block GMRES: dependent block encountered".into(),
            ));
        }
        r[j * p + j] = norm.into();
        let inv = 1.0 / norm;
        let col_mut = block.col_mut(j);
        for (dst, &src) in col_mut.iter_mut().zip(col_buf.iter()) {
            *dst = src * inv;
        }
    }
    Ok(r)
}

#[cfg(feature = "backend-faer")]
fn extract_h(h_full: &[S], ld: usize, rows: usize, cols: usize) -> Vec<S> {
    let mut h: Vec<S> = vec![0.0.into(); rows * cols];
    for col in 0..cols {
        for row in 0..rows {
            h[row + col * rows] = h_full[row + col * ld];
        }
    }
    h
}

#[cfg(feature = "backend-faer")]
fn build_g(rows: usize, p: usize, beta: &[S]) -> Vec<S> {
    let mut g: Vec<S> = vec![0.0.into(); rows * p];
    for row in 0..p {
        for col in 0..p {
            g[row + col * rows] = beta[row * p + col];
        }
    }
    g
}

#[cfg(feature = "backend-faer")]
fn solve_normal_eq(
    h: &[S],
    rows: usize,
    cols: usize,
    g: &[S],
    p: usize,
) -> Result<Vec<S>, KError> {
    let mut ht_h = vec![0.0.into(); cols * cols];
    let mut ht_g = vec![0.0.into(); cols * p];
    for col_i in 0..cols {
        for col_j in 0..cols {
            let mut sum: S = 0.0.into();
            for row in 0..rows {
                let h_i = h[row + col_i * rows];
                let h_j = h[row + col_j * rows];
                sum += conj_s(h_i) * h_j; // H^H H
            }
            ht_h[col_i + col_j * cols] = sum;
        }
        for rhs in 0..p {
            let mut sum: S = 0.0.into();
            for row in 0..rows {
                let h_i = h[row + col_i * rows];
                let g_val = g[row + rhs * rows];
                sum += conj_s(h_i) * g_val; // H^H g
            }
            ht_g[col_i + rhs * cols] = sum;
        }
    }
    let a = MatRef::from_column_major_slice(&ht_h, cols, cols);
    let lu = FullPivLu::new(a);
    let mut y = ht_g;
    let y_mat = MatMut::from_column_major_slice_mut(&mut y, cols, p);
    lu.solve_in_place_with_conj(Conj::No, y_mat);
    Ok(y)
}

#[cfg(feature = "backend-faer")]
fn update_solution(
    x_out: &mut BlockVec,
    x_base: &BlockVec,
    basis: &[BlockVec],
    y: &[S],
    rows_y: usize,
) {
    let n = x_out.nrows();
    let p = x_out.ncols();
    x_out.as_mut_slice().copy_from_slice(x_base.as_slice());
    let num_blocks = rows_y / p;
    for block_idx in 0..num_blocks {
        let v = &basis[block_idx];
        for rhs in 0..p {
            let x_col = x_out.col_mut(rhs);
            for r in 0..p {
                let coeff = y[block_idx * p + r + rhs * rows_y];
                if coeff == 0.0.into() {
                    continue;
                }
                let v_col = v.col(r);
                for i in 0..n {
                    x_col[i] += coeff * v_col[i];
                }
            }
        }
    }
}
