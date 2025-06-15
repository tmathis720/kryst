//! SOR/SSOR preconditioner tests for kryst
use approx::assert_relative_eq;
use kryst::preconditioner::{Sor, MatSorType, Preconditioner};
use faer::Mat;

fn make_tridiag(n: usize, a: f64, b: f64, c: f64) -> Mat<f64> {
    let mut mat = Mat::<f64>::zeros(n, n);
    for i in 0..n {
        if i > 0 { mat[(i, i-1)] = a; }
        mat[(i, i)] = b;
        if i+1 < n { mat[(i, i+1)] = c; }
    }
    mat
}

fn make_eye(n: usize) -> Mat<f64> {
    let mut mat = Mat::<f64>::zeros(n, n);
    for i in 0..n {
        mat[(i, i)] = 1.0;
    }
    mat
}

#[test]
fn test_sor_identity() {
    let n = 5;
    let a = make_eye(n);
    let mut sor = Sor::<Mat<f64>, Vec<f64>, f64>::new(1.0, 1, 1, MatSorType::APPLY_LOWER, 0.0);
    sor.setup(&a).unwrap();
    let x = vec![1.0; n];
    let mut y = vec![0.0; n];
    sor.apply(&x, &mut y).unwrap();
    assert_relative_eq!(x.as_slice(), y.as_slice(), epsilon=1e-12);
}

#[test]
fn test_sor_tridiag_forward() {
    let n = 5;
    let a = make_tridiag(n, -1.0, 4.0, -1.0);
    let mut sor = Sor::<Mat<f64>, Vec<f64>, f64>::new(1.0, 1, 1, MatSorType::APPLY_LOWER, 0.0);
    sor.setup(&a).unwrap();
    let x = vec![1.0; n];
    let mut y = vec![0.0; n];
    sor.apply(&x, &mut y).unwrap();
    // Compute expected SOR sweep (forward, in-place)
    let mut expected = vec![0.0; n];
    for i in 0..n {
        let left = if i > 0 { expected[i-1] } else { 0.0 };
        let right = if i+1 < n { x[i+1] } else { 0.0 };
        expected[i] = (x[i] + left + right) / 4.0;
    }
    for i in 0..n {
        assert!((y[i] - expected[i]).abs() < 1e-12_f64, "SOR mismatch at i={}: got {}, expected {}", i, y[i], expected[i]);
    }
}

#[test]
fn test_ssor_tridiag() {
    let n = 5;
    let a = make_tridiag(n, -1.0, 4.0, -1.0);
    let mut sor = Sor::<Mat<f64>, Vec<f64>, f64>::new(1.0, 1, 1, MatSorType::SYMMETRIC_SWEEP, 0.0);
    sor.setup(&a).unwrap();
    let x = vec![1.0; n];
    let mut y = vec![0.0; n];
    sor.apply(&x, &mut y).unwrap();
    assert!(y.iter().all(|&v| (v as f64).is_finite()));
}

#[test]
fn test_sor_display() {
    let sor = Sor::<Mat<f64>, Vec<f64>, f64>::new(1.5, 2, 1, MatSorType::APPLY_LOWER, 0.1);
    let s = format!("{}", sor);
    assert!(s.contains("SOR(omega=1.5"));
}
