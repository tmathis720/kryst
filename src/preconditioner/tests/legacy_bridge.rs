#![cfg(not(feature = "complex"))]

#[cfg(feature = "legacy-pc-bridge")]
#[test]
fn legacy_bridge_reuses_scratch() {
    use crate::algebra::prelude::*;
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
            let scale = S::from_real(2.0).real();
            for (zi, ri) in z.iter_mut().zip(r.iter()) {
                *zi = scale * *ri;
            }
            Ok(())
        }
    }

    let mut adapter = crate::preconditioner::LegacyOpPreconditioner::new(Box::new(Twice));
    let a = Mat::<f64>::zeros(3, 3);
    adapter.setup(&a).unwrap();

    let one = S::from_real(1.0).real();
    let three = S::from_real(3.0).real();
    let neg_two = S::from_real(-2.0).real();
    let zero = R::default();
    let x = [one, three, neg_two];
    let mut y = [zero; 3];
    adapter.apply(PcSide::Left, &x, &mut y).unwrap();
    let expected = [
        S::from_real(2.0).real(),
        S::from_real(6.0).real(),
        S::from_real(-4.0).real(),
    ];
    let expected_s = expected.map(S::from_real);
    let y_s = y.map(S::from_real);
    crate::assert_vec_close!("legacy bridge apply", &y_s, &expected_s);

    let mut y2 = [zero; 3];
    adapter.apply(PcSide::Left, &x, &mut y2).unwrap();
    let y2_s = y2.map(S::from_real);
    crate::assert_vec_close!("legacy bridge reuse", &y2_s, &expected_s);
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
            let scale = S::from_real(2.0).real();
            for (zi, ri) in z.iter_mut().zip(r.iter()) {
                *zi = scale * *ri;
            }
            Ok(())
        }
    }

    let mut adapter = crate::preconditioner::LegacyOpPreconditioner::new(Box::new(Twice));
    let a = Mat::<f64>::zeros(3, 3);
    adapter.setup(&a).unwrap();

    let one = S::from_real(1.0).real();
    let three = S::from_real(3.0).real();
    let neg_two = S::from_real(-2.0).real();
    let zero = R::default();
    let x_real = [one, three, neg_two];
    let mut y_real = [zero; 3];
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
    crate::assert_vec_close!("legacy bridge apply_s", &y_s, &expected);

    let mut y_mut = vec![S::zero(); 3];
    KPreconditioner::apply_mut_s(&mut adapter, PcSide::Left, &x_s, &mut y_mut, &mut scratch)
        .expect("apply_mut_s bridge");
    crate::assert_vec_close!("legacy bridge apply_mut_s", &y_mut, &expected);
}
