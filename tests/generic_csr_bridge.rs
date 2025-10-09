use std::sync::Arc;

use kryst::algebra::prelude::*;
use kryst::assert_vec_close;
use kryst::matrix::convert::{dense_from_linop, to_csc_cached, to_csr_cached};
use kryst::matrix::csc::CscMatrix as FaerCscMatrix;
use kryst::matrix::csr::CsrMatrix as ScalarCsrMatrix;
use kryst::matrix::format::AsFormat;
use kryst::matrix::op::{GenericCsrOp, LinOp};
use kryst::matrix::sparse::CsrMatrix as FaerCsrMatrix;
use kryst::matrix::spmv::plan::SpmvTuning;

fn make_generic_op() -> (Arc<GenericCsrOp<f64>>, Vec<usize>, Vec<usize>, Vec<f64>) {
    let values: Vec<f64> = vec![1.0, -2.0, 3.5, 4.0, -1.5];
    let matrix = ScalarCsrMatrix::<f64>::new(
        3,
        3,
        vec![0, 2, 4, 5],
        vec![0, 2, 1, 2, 0],
        values.clone(),
    );
    let rowptr = matrix.rowptr.clone();
    let colind = matrix.colind.clone();
    let tuning = SpmvTuning {
        allow_simd: false,
        ..Default::default()
    };
    let op = Arc::new(GenericCsrOp::new(Arc::new(matrix), &tuning));
    (op, rowptr, colind, values)
}

#[test]
fn generic_csr_to_csr_cached_preserves_storage() {
    let (op, rowptr, colind, values) = make_generic_op();
    let csr = to_csr_cached(op.as_ref() as &dyn LinOp<S = f64>, R::default())
        .expect("conversion should succeed");
    assert_eq!(csr.row_ptr(), rowptr);
    assert_eq!(csr.col_idx(), colind);
    assert_eq!(csr.values(), values);
}

#[test]
fn generic_csr_to_csc_cached_roundtrips() {
    let (op, rowptr, colind, values) = make_generic_op();
    let csc = to_csc_cached(op.as_ref() as &dyn LinOp<S = f64>, R::default())
        .expect("conversion should succeed");
    let csr = AsFormat::to_csr_cached(csc.as_ref(), R::default());
    assert_eq!(csr.row_ptr(), rowptr);
    assert_eq!(csr.col_idx(), colind);
    assert_eq!(csr.values(), values);
}

#[test]
fn generic_csr_dense_conversion_matches_reference() {
    let (op, rowptr, colind, values) = make_generic_op();
    let dense =
        dense_from_linop(op.as_ref() as &dyn LinOp<S = f64>).expect("conversion should succeed");
    let mut expected: Vec<R> = vec![R::default(); 9];
    for i in 0..3 {
        for idx in rowptr[i]..rowptr[i + 1] {
            let j = colind[idx];
            expected[i * 3 + j] = values[idx];
        }
    }
    let mut actual: Vec<R> = vec![R::default(); 9];
    for i in 0..3 {
        for j in 0..3 {
            actual[i * 3 + j] = dense[(i, j)];
        }
    }
    assert_eq!(actual, expected);
}

#[test]
fn scalar_csr_spmv_matches_manual() {
    let rowptr = vec![0, 2, 4, 5];
    let colind = vec![0, 2, 1, 2, 0];
    let values_real: Vec<R> = vec![R::from(1.0), R::from(-2.0), R::from(3.5), R::from(4.0), R::from(-1.5)];
    let values = values_real
        .iter()
        .copied()
        .map(S::from_real)
        .collect::<Vec<_>>();
    let matrix = ScalarCsrMatrix::new(3, 3, rowptr.clone(), colind.clone(), values.clone());

    let x = vec![S::from_real(2.0), S::from_real(-1.0), S::from_real(0.5)];
    let mut actual = vec![S::zero(); 3];
    matrix.spmv(&x, &mut actual);

    let mut expected = vec![S::zero(); 3];
    for i in 0..3 {
        let start = rowptr[i];
        let end = rowptr[i + 1];
        let mut acc = S::zero();
        for idx in start..end {
            let j = colind[idx];
            acc = acc + values[idx] * x[j];
        }
        expected[i] = acc;
    }

    assert_vec_close!("scalar csr spmv", &expected, &actual);
}

#[test]
fn faer_csr_to_scalar_lifts_values() {
    let rowptr = vec![0, 2, 3];
    let colind = vec![0, 1, 1];
    let values = vec![1.0, -2.5, 3.0];
    let faer = FaerCsrMatrix::from_csr(2, 2, rowptr.clone(), colind.clone(), values.clone());

    let scalar = faer.to_scalar_csr();
    assert_eq!(scalar.rowptr(), rowptr);
    assert_eq!(scalar.colind(), colind);
    for (expected, actual) in values.iter().zip(scalar.values()) {
        assert_eq!(actual.real(), *expected);
        assert_eq!(actual.imag(), R::default());
    }
}

#[test]
fn faer_csc_t_matvec_matches_manual() {
    let colptr = vec![0, 2, 3];
    let rowidx = vec![0, 1, 1];
    let values = vec![S::from_real(1.0), S::from_real(-2.5), S::from_real(3.0)];
    let csc = FaerCscMatrix::from_csc(2, 2, colptr.clone(), rowidx.clone(), values.clone());

    let x = vec![S::from_real(0.5), S::from_real(-1.0)];
    let mut actual = vec![S::zero(); 2];
    csc.t_matvec(&x, &mut actual);

    let mut expected = vec![S::zero(); 2];
    for (j, yj) in expected.iter_mut().enumerate() {
        let mut acc = S::zero();
        for p in colptr[j]..colptr[j + 1] {
            let row = rowidx[p];
            acc = acc + values[p] * x[row];
        }
        *yj = acc;
    }

    assert_vec_close!("faer csc t_matvec", &expected, &actual);
}
