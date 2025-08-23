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
