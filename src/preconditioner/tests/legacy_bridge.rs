#[cfg(feature = "legacy-pc-bridge")]
#[test]
fn legacy_bridge_reuses_scratch() {
    use crate::preconditioner::{PcSide, Preconditioner, legacy};
    use faer::Mat;

    // Minimal legacy PC that scales input by 2 into output.
    struct Twice;
    impl legacy::Preconditioner<Mat<f64>, Vec<f64>> for Twice {
        fn setup(&mut self, _a: &Mat<f64>) -> Result<(), crate::error::KError> {
            Ok(())
        }
        fn apply(
            &self,
            _side: PcSide,
            r: &Vec<f64>,
            z: &mut Vec<f64>,
        ) -> Result<(), crate::error::KError> {
            for (zi, ri) in z.iter_mut().zip(r.iter()) {
                *zi = 2.0 * *ri;
            }
            Ok(())
        }
    }

    let mut adapter = crate::preconditioner::LegacyOpPreconditioner::new(Box::new(Twice));
    let a = Mat::<f64>::zeros(3, 3);
    adapter.setup(&a).unwrap();

    let x = [1.0, 3.0, -2.0];
    let mut y = [0.0; 3];
    adapter.apply(PcSide::Left, &x, &mut y).unwrap();
    assert_eq!(y, [2.0, 6.0, -4.0]);

    let mut y2 = [0.0; 3];
    adapter.apply(PcSide::Left, &x, &mut y2).unwrap();
    assert_eq!(y2, [2.0, 6.0, -4.0]);
}

#[cfg(all(feature = "legacy-pc-bridge", feature = "complex"))]
#[test]
fn legacy_bridge_apply_s_matches_real_path() {
    use crate::algebra::bridge::BridgeScratch;
    use crate::algebra::prelude::*;
    use crate::ops::kpc::KPreconditioner;
    use crate::preconditioner::{PcSide, Preconditioner, legacy};
    use faer::Mat;

    struct Twice;
    impl legacy::Preconditioner<Mat<f64>, Vec<f64>> for Twice {
        fn setup(&mut self, _a: &Mat<f64>) -> Result<(), crate::error::KError> {
            Ok(())
        }

        fn apply(
            &self,
            _side: PcSide,
            r: &Vec<f64>,
            z: &mut Vec<f64>,
        ) -> Result<(), crate::error::KError> {
            for (zi, ri) in z.iter_mut().zip(r.iter()) {
                *zi = 2.0 * *ri;
            }
            Ok(())
        }
    }

    let mut adapter = crate::preconditioner::LegacyOpPreconditioner::new(Box::new(Twice));
    let a = Mat::<f64>::zeros(3, 3);
    adapter.setup(&a).unwrap();

    let x_real = [1.0, 3.0, -2.0];
    let mut y_real = [0.0; 3];
    adapter
        .apply(PcSide::Left, &x_real, &mut y_real)
        .expect("legacy apply");

    let expected: Vec<S> = y_real.iter().map(|&v| S::from_real(v)).collect();
    let x_s: Vec<S> = x_real.iter().map(|&v| S::from_real(v)).collect();

    let mut scratch = BridgeScratch::default();
    let dims = KPreconditioner::dims(&adapter);
    assert_eq!(dims, (3, 3));

    let mut y_s = vec![S::zero(); 3];
    KPreconditioner::apply_s(&adapter, PcSide::Left, &x_s, &mut y_s, &mut scratch)
        .expect("apply_s bridge");
    assert_eq!(y_s, expected);

    let mut y_mut = vec![S::zero(); 3];
    KPreconditioner::apply_mut_s(&mut adapter, PcSide::Left, &x_s, &mut y_mut, &mut scratch)
        .expect("apply_mut_s bridge");
    assert_eq!(y_mut, expected);
}
