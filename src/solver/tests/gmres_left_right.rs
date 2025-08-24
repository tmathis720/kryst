#[cfg(test)]
mod tests_gmres_lr {
    use crate::context::ksp_context::{KspContext, SolverType};
    use crate::context::pc_context::PcType;
    use crate::error::KError;
    use crate::matrix::op::LinOp;
    use crate::preconditioner::PcSide;
    use faer::Mat;
    use std::sync::Arc;

    fn true_residual_norm(a: &dyn LinOp<S = f64>, x: &[f64], b: &[f64]) -> f64 {
        let n = b.len();
        let mut r = b.to_vec();
        let mut ax = vec![0.0; n];
        a.matvec(x, &mut ax);
        for i in 0..n {
            r[i] -= ax[i];
        }
        (r.iter().map(|v| v * v).sum::<f64>()).sqrt()
    }

    #[test]
    fn gmres_left_right_same_solution_jacobi() -> Result<(), KError> {
        // Nonsymmetric, strictly diagonally dominant (easy to solve)
        let a = Mat::from_fn(3, 3, |i, j| match (i, j) {
            (0, 0) => 4.0,
            (0, 1) => 1.0,
            (0, 2) => 0.0,
            (1, 0) => -2.0,
            (1, 1) => 3.0,
            (1, 2) => 1.0,
            (2, 0) => 0.0,
            (2, 1) => -1.0,
            (2, 2) => 2.0,
            _ => unreachable!(),
        });
        let b = [1.0, 2.0, 3.0];

        let amat: Arc<dyn LinOp<S = f64>> = Arc::new(a.clone());

        // --- LEFT preconditioning
        let mut ksp_left = KspContext::new();
        ksp_left.set_type(SolverType::Gmres)?;
        ksp_left.set_pc_type(PcType::Jacobi, None)?;
        ksp_left.set_operators(amat.clone(), None);
        ksp_left.pc_side = PcSide::Left;
        ksp_left.rtol = 1e-10;
        let mut x_left = [0.0; 3];
        let stats_left = ksp_left.solve(&b, &mut x_left)?;
        assert!(stats_left.iterations > 0);
        assert!(stats_left.final_residual < 1e-4, "left solver failed to converge");

        // --- RIGHT preconditioning
        let mut ksp_right = KspContext::new();
        ksp_right.set_type(SolverType::Gmres)?;
        ksp_right.set_pc_type(PcType::Jacobi, None)?;
        ksp_right.set_operators(amat.clone(), None);
        ksp_right.pc_side = PcSide::Right;
        ksp_right.rtol = 1e-10;
        let mut x_right = [0.0; 3];
        let stats_right = ksp_right.solve(&b, &mut x_right)?;
        assert!(stats_right.iterations > 0);
        assert!(stats_right.final_residual < 1e-4, "right solver failed to converge");

        // True residuals are small and comparable
        let res_l = true_residual_norm(amat.as_ref(), &x_left, &b);
        let res_r = true_residual_norm(amat.as_ref(), &x_right, &b);
        assert!(res_l < 1e-4, "left true residual too large: {res_l:e}");
        assert!(res_r < 1e-4, "right true residual too large: {res_r:e}");
        assert!(
            (res_l - res_r).abs() < 1e-4,
            "true residuals differ: {res_l:e} vs {res_r:e}"
        );

        // Solutions match to tolerance
        for i in 0..3 {
            assert!(
                (x_left[i] - x_right[i]).abs() < 1e-4,
                "x_left[{i}] != x_right[{i}]"
            );
        }

        // Internal sanity: Left keeps no Z basis; Right populates it
        if let Some(wl) = ksp_left.debug_workspace() {
            assert!(wl.z.is_empty(), "Left GMRES should not populate Z basis");
        }
        if let Some(wr) = ksp_right.debug_workspace() {
            assert!(!wr.z.is_empty(), "Right GMRES should populate Z basis");
            assert_eq!(
                wr.z.len(),
                wr.q.len().saturating_sub(1),
                "Z basis length should match Krylov dim"
            );
        }

        // And both reported a result
        Ok(())
    }
}
