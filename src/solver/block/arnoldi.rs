//! Block Arnoldi helpers.

#[allow(unused_imports)]
use crate::algebra::blas::{dot_conj, nrm2};
#[allow(unused_imports)]
use crate::algebra::prelude::*;
use crate::context::ksp_context::Workspace;
use crate::error::KError;
use crate::parallel::UniverseComm;
use crate::utils::reduction::AllreduceOps;

use super::block_vec::BlockVec;
use super::kernels;

/// Result produced by a single block Arnoldi step.
pub struct ArnoldiOutput {
    /// Projection coefficients for each previously constructed block.
    /// Flattened as `block_index * p * p + row * p + col` with `p = block_size`.
    pub coeffs: Vec<S>,
    /// Upper-triangular block returned by the Cholesky-QR step.
    pub r_block: Vec<S>,
}

/// Perform a block Arnoldi step.
///
/// * `basis` - orthonormal basis blocks accumulated so far. The current block
///   should be included as the last entry.
/// * `w` - scratch block containing the result of applying the operator to the
///   search block. On return it is overwritten with the orthonormalised block.
/// * `max_cond` - conditioning guard for the Cholesky factor.
///
/// The function packs projection coefficients and the symmetric Gram matrix
/// into a single reduction to minimise synchronisation.
pub fn block_arnoldi_step(
    basis: &[BlockVec],
    w: &mut BlockVec,
    comm: &UniverseComm,
    work: &mut Workspace,
    _max_cond: R,
) -> Result<ArnoldiOutput, KError> {
    if basis.is_empty() {
        return Err(KError::InvalidInput(
            "block Arnoldi requires at least one basis block".into(),
        ));
    }
    let p = w.ncols();
    let n = w.nrows();
    let num_blocks = basis.len();

    let mut columns: Vec<&[S]> = Vec::with_capacity(num_blocks * p);
    for block in basis {
        if block.ncols() != p || block.nrows() != n {
            return Err(KError::InvalidInput(
                "basis blocks must share the same dimensions".into(),
            ));
        }
        for col in 0..p {
            columns.push(block.col(col));
        }
    }

    let mut c_local = vec![S::zero(); columns.len() * p];
    kernels::tall_t_times_block(&columns, w, &mut c_local);

    let mut payload = std::mem::take(&mut work.blk_payload);
    payload.clear();
    payload.reserve(pack_scalars_len(c_local.len()));
    pack_scalars(&mut payload, &c_local);

    let (handle, send) = comm.allreduce_n_async(payload, work.reduction_options())?;
    let reduced = UniverseComm::wait_vec(handle);
    work.blk_payload = send;
    let c_global = unpack_scalars(&reduced);

    kernels::block_project(&columns, &c_global, columns.len(), p, w);

    let r_block = classical_qr(w, work)?;

    let mut coeffs = vec![S::zero(); num_blocks * p * p];
    for (block_idx, block_coeffs) in coeffs.chunks_mut(p * p).enumerate() {
        for row in 0..p {
            for col in 0..p {
                block_coeffs[row * p + col] = c_global[(block_idx * p + row) * p + col];
            }
        }
    }

    Ok(ArnoldiOutput { coeffs, r_block })
}

fn classical_qr(block: &mut BlockVec, work: &mut Workspace) -> Result<Vec<S>, KError> {
    let p = block.ncols();
    let n = block.nrows();
    let mut r = vec![S::zero(); p * p];
    work.blk_scratch.resize(n, S::zero());
    let col_buf = &mut work.blk_scratch[..n];
    for j in 0..p {
        {
            let col = block.col(j);
            col_buf.copy_from_slice(col);
        }
        for i in 0..j {
            let qi = block.col(i);
            let dot = dot_conj(qi, &col_buf[..]);
            r[i * p + j] = dot;
            for (buf, &qi_val) in col_buf.iter_mut().zip(qi.iter()) {
                *buf -= dot * qi_val;
            }
        }
        let norm = nrm2(&col_buf[..]);
        if norm <= R::default() {
            return Err(KError::FactorError(
                "block Arnoldi: dependent block encountered".into(),
            ));
        }
        r[j * p + j] = S::from_real(norm);
        let inv = S::from_real(1.0 / norm);
        let col_mut = block.col_mut(j);
        for (dst, &src) in col_mut.iter_mut().zip(col_buf.iter()) {
            *dst = src * inv;
        }
    }
    Ok(r)
}

#[cfg(feature = "complex")]
fn pack_scalars(payload: &mut Vec<R>, values: &[S]) {
    for &val in values {
        payload.push(val.real());
        payload.push(val.imag());
    }
}

#[cfg(not(feature = "complex"))]
fn pack_scalars(payload: &mut Vec<R>, values: &[S]) {
    payload.extend(values.iter().map(|&val| val.real()));
}

#[cfg(feature = "complex")]
fn pack_scalars_len(len: usize) -> usize {
    len * 2
}

#[cfg(not(feature = "complex"))]
fn pack_scalars_len(len: usize) -> usize {
    len
}

#[cfg(feature = "complex")]
fn unpack_scalars(buffer: &[R]) -> Vec<S> {
    buffer
        .chunks_exact(2)
        .map(|chunk| S::from_parts(chunk[0], chunk[1]))
        .collect()
}

#[cfg(not(feature = "complex"))]
fn unpack_scalars(buffer: &[R]) -> Vec<S> {
    buffer.iter().map(|&re| S::from_real(re)).collect()
}
