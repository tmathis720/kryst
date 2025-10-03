use std::sync::Arc;

use kryst::matrix::convert::{dense_from_linop, to_csc_cached, to_csr_cached};
use kryst::matrix::csr::CsrMatrix as ScalarCsrMatrix;
use kryst::matrix::format::AsFormat;
use kryst::matrix::op::{GenericCsrOp, LinOp};
use kryst::matrix::spmv::plan::SpmvTuning;

fn make_generic_op() -> (Arc<GenericCsrOp<f64>>, Vec<usize>, Vec<usize>, Vec<f64>) {
    let matrix = ScalarCsrMatrix::new(
        3,
        3,
        vec![0, 2, 4, 5],
        vec![0, 2, 1, 2, 0],
        vec![1.0, -2.0, 3.5, 4.0, -1.5],
    );
    let rowptr = matrix.rowptr.clone();
    let colind = matrix.colind.clone();
    let values = matrix.values.clone();
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
    let csr =
        to_csr_cached(op.as_ref() as &dyn LinOp<S = f64>, 0.0).expect("conversion should succeed");
    assert_eq!(csr.row_ptr(), rowptr);
    assert_eq!(csr.col_idx(), colind);
    assert_eq!(csr.values(), values);
}

#[test]
fn generic_csr_to_csc_cached_roundtrips() {
    let (op, rowptr, colind, values) = make_generic_op();
    let csc =
        to_csc_cached(op.as_ref() as &dyn LinOp<S = f64>, 0.0).expect("conversion should succeed");
    let csr = AsFormat::to_csr_cached(csc.as_ref(), 0.0);
    assert_eq!(csr.row_ptr(), rowptr);
    assert_eq!(csr.col_idx(), colind);
    assert_eq!(csr.values(), values);
}

#[test]
fn generic_csr_dense_conversion_matches_reference() {
    let (op, rowptr, colind, values) = make_generic_op();
    let dense =
        dense_from_linop(op.as_ref() as &dyn LinOp<S = f64>).expect("conversion should succeed");
    let mut expected = vec![0.0; 9];
    for i in 0..3 {
        for idx in rowptr[i]..rowptr[i + 1] {
            let j = colind[idx];
            expected[i * 3 + j] = values[idx];
        }
    }
    let mut actual = vec![0.0; 9];
    for i in 0..3 {
        for j in 0..3 {
            actual[i * 3 + j] = dense[(i, j)];
        }
    }
    assert_eq!(actual, expected);
}
