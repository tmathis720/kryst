use crate::algebra::bridge::BridgeScratch;
use crate::algebra::prelude::*;
use crate::ops::klinop::KLinOp;
use crate::solver::common::ReductCtx;
use crate::utils::convergence::ConvergedReason;

#[inline]
pub fn true_residual_norm<A: KLinOp<Scalar = S> + ?Sized>(
    a: &A,
    b: &[S],
    x: &[S],
    red: &ReductCtx,
    tmp: &mut [S],
    scratch: &mut BridgeScratch,
) -> R {
    a.matvec_s(x, tmp, scratch);
    for i in 0..tmp.len() {
        tmp[i] = b[i] - tmp[i];
    }
    red.norm2(tmp)
}

#[inline]
pub fn true_residual_converged_reason(
    true_residual: R,
    bnorm: R,
    atol: R,
    rtol: R,
) -> Option<ConvergedReason> {
    let thr = atol.max(rtol * bnorm);
    if true_residual <= thr {
        return Some(if true_residual <= atol {
            ConvergedReason::ConvergedAtol
        } else {
            ConvergedReason::ConvergedRtol
        });
    }
    None
}
