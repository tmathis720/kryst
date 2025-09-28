//! Block Arnoldi helpers.

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
    pub coeffs: Vec<f64>,
    /// Upper-triangular block returned by the Cholesky-QR step.
    pub r_block: Vec<f64>,
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
    max_cond: f64,
) -> Result<ArnoldiOutput, KError> {
    if basis.is_empty() {
        return Err(KError::InvalidInput(
            "block Arnoldi requires at least one basis block".into(),
        ));
    }
    let p = w.ncols();
    let n = w.nrows();
    let num_blocks = basis.len();

    let mut columns: Vec<&[f64]> = Vec::with_capacity(num_blocks * p);
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

    let mut c_local = vec![0.0; columns.len() * p];
    kernels::tall_t_times_block(&columns, w, &mut c_local);
    let mut g_local = vec![0.0; p * p];
    kernels::gram_pxp(w, w, &mut g_local);

    let mut payload = vec![0.0; c_local.len() + g_local.len()];
    payload[..c_local.len()].copy_from_slice(&c_local);
    payload[c_local.len()..].copy_from_slice(&g_local);

    let (handle, _send) = comm.allreduce_n_async(payload, work.reduction_options())?;
    let reduced = UniverseComm::wait_vec(handle);
    let (c_global, g_global) = reduced.split_at(columns.len() * p);

    kernels::block_project(&columns, c_global, columns.len(), p, w);

    let mut s = g_global.to_vec();
    for i in 0..p {
        for j in 0..p {
            let mut sum = 0.0;
            for row in 0..columns.len() {
                let lhs = c_global[row * p + i];
                let rhs = c_global[row * p + j];
                sum += lhs * rhs;
            }
            s[i * p + j] -= sum;
        }
    }

    let mut r_block = s.clone();
    let use_fallback = match chol_upper(&mut r_block, p) {
        Ok(()) => {
            let mut max_diag: f64 = 0.0;
            let mut min_diag: f64 = f64::INFINITY;
            for idx in 0..p {
                let diag = r_block[idx * p + idx].abs();
                max_diag = max_diag.max(diag);
                min_diag = min_diag.min(diag);
            }
            if min_diag <= 0.0 {
                true
            } else {
                let cond = max_diag / min_diag;
                !cond.is_finite() || cond > max_cond
            }
        }
        Err(_) => true,
    };

    if use_fallback {
        r_block = classical_qr(w)?;
    } else {
        triangular_solve_right_upper(&r_block, p, w);
    }

    let mut coeffs = vec![0.0; num_blocks * p * p];
    for (block_idx, block_coeffs) in coeffs.chunks_mut(p * p).enumerate() {
        for row in 0..p {
            for col in 0..p {
                block_coeffs[row * p + col] = c_global[(block_idx * p + row) * p + col];
            }
        }
    }

    Ok(ArnoldiOutput { coeffs, r_block })
}

fn chol_upper(mat: &mut [f64], n: usize) -> Result<(), KError> {
    for j in 0..n {
        for i in 0..=j {
            let mut sum = mat[i * n + j];
            for k in 0..i {
                sum -= mat[k * n + i] * mat[k * n + j];
            }
            if i == j {
                if sum <= 0.0 || !sum.is_finite() {
                    return Err(KError::FactorError(
                        "block Arnoldi: Cholesky factorisation failed".into(),
                    ));
                }
                mat[i * n + j] = sum.sqrt();
            } else {
                let diag = mat[i * n + i];
                if diag.abs() <= f64::EPSILON {
                    return Err(KError::FactorError(
                        "block Arnoldi: zero diagonal during Cholesky".into(),
                    ));
                }
                mat[i * n + j] = sum / diag;
            }
        }
        for i in (j + 1)..n {
            mat[i * n + j] = 0.0;
        }
    }
    Ok(())
}

fn triangular_solve_right_upper(r: &[f64], p: usize, block: &mut BlockVec) {
    let n = block.nrows();
    let mut row_buf = vec![0.0; p];
    for row in 0..n {
        for col in 0..p {
            row_buf[col] = block.col(col)[row];
        }
        for j in (0..p).rev() {
            let mut sum = row_buf[j];
            for k in (j + 1)..p {
                sum -= row_buf[k] * r[j * p + k];
            }
            row_buf[j] = sum / r[j * p + j];
        }
        for col in 0..p {
            block.col_mut(col)[row] = row_buf[col];
        }
    }
}

fn classical_qr(block: &mut BlockVec) -> Result<Vec<f64>, KError> {
    let p = block.ncols();
    let n = block.nrows();
    let mut r = vec![0.0; p * p];
    let mut col_buf = vec![0.0; n];
    for j in 0..p {
        {
            let col = block.col(j);
            col_buf.copy_from_slice(col);
        }
        for i in 0..j {
            let qi = block.col(i);
            let mut dot = 0.0;
            for k in 0..n {
                dot += qi[k] * col_buf[k];
            }
            r[i * p + j] = dot;
            for k in 0..n {
                col_buf[k] -= dot * qi[k];
            }
        }
        let mut norm_sq = 0.0;
        for k in 0..n {
            norm_sq += col_buf[k] * col_buf[k];
        }
        let norm = norm_sq.sqrt();
        if norm <= f64::EPSILON {
            return Err(KError::FactorError(
                "block Arnoldi: dependent block encountered".into(),
            ));
        }
        r[j * p + j] = norm;
        for k in 0..n {
            block.col_mut(j)[k] = col_buf[k] / norm;
        }
    }
    Ok(r)
}
