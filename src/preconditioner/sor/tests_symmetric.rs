#[cfg(all(test, feature = "legacy-pc-bridge"))]
mod tests_sor_symmetric {
    use crate::error::KError;
    use crate::preconditioner::sor::{MatSorType, Sor};
    use crate::preconditioner::{LegacyOpPreconditioner, PcSide, Preconditioner};
    use faer::Mat;

    #[test]
    fn sor_symmetric_is_forward_then_backward() -> Result<(), KError> {
        // SPD-ish tridiagonal
        let a = Mat::from_fn(3, 3, |i, j| match (i, j) {
            (0, 0) => 4.0,
            (0, 1) => -1.0,
            (0, 2) => 0.0,
            (1, 0) => -1.0,
            (1, 1) => 4.0,
            (1, 2) => -1.0,
            (2, 0) => 0.0,
            (2, 1) => -1.0,
            (2, 2) => 3.0,
            _ => unreachable!(),
        });
        let x = [1.0, -2.0, 3.0];

        // Lower-only
        let sor_lower = Sor::<Mat<f64>, Vec<f64>>::new(1.0, 1, 0, MatSorType::APPLY_LOWER, 0.0);
        let mut pc_lower = LegacyOpPreconditioner::new(Box::new(sor_lower));
        pc_lower.setup(&a)?;
        let mut y_fwd = [0.0; 3];
        pc_lower.apply(PcSide::Left, &x, &mut y_fwd)?;

        // Upper-only
        let sor_upper = Sor::<Mat<f64>, Vec<f64>>::new(1.0, 1, 0, MatSorType::APPLY_UPPER, 0.0);
        let mut pc_upper = LegacyOpPreconditioner::new(Box::new(sor_upper));
        pc_upper.setup(&a)?;
        let mut y_bwd = [0.0; 3];
        pc_upper.apply(PcSide::Left, &y_fwd, &mut y_bwd)?;

        // Symmetric
        let sor_sym = Sor::<Mat<f64>, Vec<f64>>::new(1.0, 1, 0, MatSorType::SYMMETRIC_SWEEP, 0.0);
        let mut pc_sym = LegacyOpPreconditioner::new(Box::new(sor_sym));
        pc_sym.setup(&a)?;
        let mut y_sym = [0.0; 3];
        // should not matter which side we pass for an M^{-1} application
        pc_sym.apply(PcSide::Right, &x, &mut y_sym)?;

        for i in 0..3 {
            assert!(
                (y_bwd[i] - y_sym[i]).abs() < 1e-12,
                "symmetric mismatch at {i}"
            );
        }
        Ok(())
    }
}
