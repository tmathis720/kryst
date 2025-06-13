use approx::assert_abs_diff_eq;
use faer::Mat;
use kryst::core::traits::{InnerProduct, MatVec};
use rand::Rng;

#[test]
fn matvec_random_small() {
    let n = 5;
    let mut rng = rand::thread_rng();
    let vals: Vec<f64> = (0..n * n).map(|_| rng.gen()).collect();
    // Use from_fn to build a column-major matrix
    let a = Mat::from_fn(n, n, |i, j| vals[j * n + i]);
    let x: Vec<f64> = (0..n).map(|_| rng.gen()).collect();
    let mut y = vec![0.0; n];
    a.matvec(&x, &mut y);

    // check y[i] == sum_j A[i,j]*x[j]
    for i in 0..n {
        let expected = (0..n).map(|j| vals[j * n + i] * x[j]).sum::<f64>();
        assert_abs_diff_eq!(y[i], expected, epsilon = 1e-12);
    }
}

#[test]
fn dot_and_norm() {
    let x = vec![1.0, 2.0, 3.0];
    let y = vec![4.0, -5.0, 6.0];
    let ip = ();
    let dot = ip.dot(&x, &y);
    assert_abs_diff_eq!(dot, 1.0 * 4.0 + 2.0 * (-5.0) + 3.0 * 6.0, epsilon = 1e-12);
    let norm_x = ip.norm(&x);
    let expected_norm = ((1.0f64).powi(2) + 2.0f64.powi(2) + 3.0f64.powi(2)).sqrt();
    assert_abs_diff_eq!(norm_x, expected_norm, epsilon = 1e-12);
}
