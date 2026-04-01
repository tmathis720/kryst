mod aug_recycling;
mod block_arnoldi;
mod block_solvers;
mod breakdown_true_residual;
mod cg_pipelined;
mod cg_side;
mod gmres_left_right;
mod gmres_right_z_basis;
mod gmres_variants;
mod gmres_workspace_reuse;
mod idrs;
mod reason_mapping;
mod stability;
mod sync_counts;

pub mod util {
    use crate::algebra::prelude::*;
    use crate::matrix::op::LinOp;
    use crate::matrix::sparse::CsrMatrix;
    use crate::matrix::utils::{self, poisson};

    pub fn spd_poisson2d(n: usize) -> CsrMatrix<f64> {
        poisson::poisson_5pt_2d(n)
    }

    #[allow(dead_code)]
    pub fn spd_poisson3d(n: usize) -> CsrMatrix<f64> {
        poisson::poisson_7pt_3d(n)
    }

    pub fn rotated_anisotropy_2d(n: usize, theta: f64, eps: f64) -> CsrMatrix<f64> {
        utils::anisotropic_poisson_2d(n, theta, eps)
    }

    pub fn nonsym_convdiff_2d(n: usize, peclet: f64) -> CsrMatrix<f64> {
        utils::convection_diffusion_2d(n, peclet)
    }

    pub fn rhs_random(n: usize, seed: u64) -> Vec<R> {
        utils::random_rhs(n, seed)
            .into_iter()
            .map(R::from)
            .collect()
    }

    pub fn vec_norm(v: &[R]) -> R {
        v.iter().map(|&x| x * x).sum::<R>().sqrt()
    }

    pub fn true_residual_norm(op: &dyn LinOp<S = f64>, x: &[R], b: &[R]) -> R {
        let mut ax: Vec<R> = vec![R::default(); b.len()];
        op.matvec(x, &mut ax);
        for (ax_i, &b_i) in ax.iter_mut().zip(b) {
            *ax_i -= b_i;
        }
        vec_norm(&ax)
    }
}
