#[cfg(test)]
mod tests_gmres_lr {
    use crate::algebra::prelude::*;
    use crate::context::ksp_context::{KspContext, SolverType};
    use crate::context::pc_context::PcType;
    use crate::error::KError;
    use crate::matrix::op::LinOp;
    use crate::preconditioner::PcSide;
    use crate::testkit::{ATOL, is_zero};
    use faer::Mat;
    use std::sync::Arc;

    fn true_residual_norm(a: &dyn LinOp<S = f64>, x: &[R], b: &[R]) -> R {
        let n = b.len();
        let mut r: Vec<R> = b.to_vec();
        let mut ax: Vec<R> = vec![R::default(); n];
        a.matvec(x, &mut ax);
        for i in 0..n {
            r[i] -= ax[i];
        }
        let sum = r.iter().map(|&v| v * v).sum::<R>();
        sum.sqrt()
    }

    #[test]
    fn gmres_left_right_same_solution_jacobi() -> Result<(), KError> {
        // Nonsymmetric, strictly diagonally dominant (easy to solve)
        let a = Mat::from_fn(3, 3, |i, j| match (i, j) {
            (0, 0) => R::from(4.0),
            (0, 1) => R::from(1.0),
            (0, 2) => R::default(),
            (1, 0) => R::from(-2.0),
            (1, 1) => R::from(3.0),
            (1, 2) => R::from(1.0),
            (2, 0) => R::default(),
            (2, 1) => R::from(-1.0),
            (2, 2) => R::from(2.0),
            _ => unreachable!(),
        });
        let b = [R::from(1.0), R::from(2.0), R::from(3.0)];

        let amat: Arc<dyn LinOp<S = f64>> = Arc::new(a.clone());

        // --- LEFT preconditioning
        let mut ksp_left = KspContext::new();
        ksp_left.set_type(SolverType::Gmres)?;
        ksp_left.set_pc_type(PcType::Jacobi, None)?;
        ksp_left.set_operators(amat.clone(), None);
        ksp_left.pc_side = PcSide::Left;
        ksp_left.rtol = 1e-10;
        let mut x_left = [R::default(); 3];
        let stats_left = ksp_left.solve(&b, &mut x_left)?;
        assert!(stats_left.iterations > 0);
        assert!(
            stats_left.final_residual < 1e-4,
            "left solver failed to converge"
        );

        // --- RIGHT preconditioning
        let mut ksp_right = KspContext::new();
        ksp_right.set_type(SolverType::Gmres)?;
        ksp_right.set_pc_type(PcType::Jacobi, None)?;
        ksp_right.set_operators(amat.clone(), None);
        ksp_right.pc_side = PcSide::Right;
        ksp_right.rtol = 1e-10;
        let mut x_right = [R::default(); 3];
        let stats_right = ksp_right.solve(&b, &mut x_right)?;
        assert!(stats_right.iterations > 0);
        assert!(
            stats_right.final_residual < 1e-4,
            "right solver failed to converge"
        );

        // True residuals are small and comparable
        let res_l = true_residual_norm(amat.as_ref(), &x_left, &b);
        let res_r = true_residual_norm(amat.as_ref(), &x_right, &b);
        assert!(
            res_l < R::from(1e-4),
            "left true residual too large: {res_l:e}"
        );
        assert!(
            res_r < R::from(1e-4),
            "right true residual too large: {res_r:e}"
        );
        assert!(
            (res_l - res_r).abs() < R::from(1e-4),
            "true residuals differ: {res_l:e} vs {res_r:e}"
        );

        // Solutions match to tolerance
        for i in 0..3 {
            assert!(
                (x_left[i] - x_right[i]).abs() < R::from(1e-4),
                "x_left[{i}] != x_right[{i}]"
            );
        }

        // Internal sanity: Left keeps no Z basis; Right populates it (via z_mem)
        if let Some(wl) = ksp_left.debug_workspace() {
            // Left-preconditioned GMRES does not allocate z_mem
            assert!(
                wl.z_mem.is_empty(),
                "Left GMRES should not populate Z basis"
            );
        }
        if let Some(wr) = ksp_right.debug_workspace() {
            // Right-preconditioned GMRES uses column-major slabs: v_mem (m+1 cols), z_mem (m cols)
            // Count how many Z columns were actually populated (non-zero norm)
            let n = wr.tmp1.len();
            let m = if n > 0 { wr.z_mem.len() / n } else { 0 };
            let mut zcols_used = 0usize;
            for j in 0..m {
                let col = &wr.z_mem[j * n..(j + 1) * n];
                if col.iter().any(|&v| !is_zero(v, ATOL)) {
                    zcols_used += 1;
                }
            }
            assert!(zcols_used > 0, "Right GMRES should populate Z basis");
        }

        // And both reported a result
        Ok(())
    }
}
