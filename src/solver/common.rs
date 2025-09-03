use crate::matrix::op::LinOp;
use crate::parallel::{Comm, UniverseComm};

/// Recompute the true residual norm ||r||_2 where r = b - A x.
///
/// This uses the provided `comm` for the dot-product reduction so it works in
/// both serial and distributed settings.
#[inline]
pub fn recompute_true_residual_norm<C: Comm>(
    a: &dyn LinOp<S = f64>,
    b: &[f64],
    x: &[f64],
    comm: &C,
    tmp: &mut [f64], // length = ncols
) -> f64 {
    a.matvec(x, tmp);
    let mut local = 0.0;
    for i in 0..tmp.len() {
        tmp[i] = b[i] - tmp[i];
        local += tmp[i] * tmp[i];
    }
    if comm.size() == 1 {
        local.sqrt()
    } else {
        comm.allreduce_sum(local).sqrt()
    }
}

/// Compute the residual norm used for iteration monitors (the "reported" norm):
/// - Left preconditioning:  ||M^{-1} r||_2
/// - Right/Symmetric:      ||r||_2
///
/// The `r_true` slice is modified in-place to hold `r = b - A x` on entry, and
/// when `side` is Left and a preconditioner is provided, `scratch` is used to
/// hold `z = M^{-1} r`.
#[inline]
pub fn reported_residual_norm(
    side: crate::preconditioner::PcSide,
    pc: Option<&dyn crate::preconditioner::Preconditioner>,
    r_true: &mut [f64],  // input: r = b - Ax, length = n
    scratch: &mut [f64], // length = n (used for M^{-1} r)
    comm: &UniverseComm,
) -> f64 {
    match side {
        crate::preconditioner::PcSide::Left => {
            if let Some(m) = pc {
                let _ = m.apply(crate::preconditioner::PcSide::Left, r_true, scratch);
                comm.dot(scratch, scratch).sqrt()
            } else {
                // no PC: Left semantics degrade to ||r||
                comm.dot(r_true, r_true).sqrt()
            }
        }
        crate::preconditioner::PcSide::Right | crate::preconditioner::PcSide::Symmetric => {
            comm.dot(r_true, r_true).sqrt()
        }
    }
}
