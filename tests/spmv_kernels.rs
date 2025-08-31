use kryst::matrix::{CscMatrix, CsrMatrix};
use kryst::matrix::sparse::SparseMatrix;

#[cfg(feature = "rayon")]
#[test]
fn csr_spmv_parallel_matches_serial() {
    // 2x3 matrix [[1,2,0],[0,3,4]]
    let csr = CsrMatrix::from_csr(
        2,
        3,
        vec![0, 2, 4],
        vec![0, 1, 1, 2],
        vec![1.0, 2.0, 3.0, 4.0],
    );
    let x = vec![1.0, 2.0, 3.0];
    let mut y_serial = vec![0.0; 2];
    csr.spmv(&x, &mut y_serial);
    let mut y_parallel = vec![0.0; 2];
    csr.spmv_parallel(&x, &mut y_parallel);
    assert_eq!(y_serial, y_parallel);
}

#[cfg(feature = "rayon")]
#[test]
fn csc_spmv_parallel_matches_serial() {
    // Matrix [[1,0],[2,3],[0,4]] in CSC form
    let csc = CscMatrix::from_csc(
        3,
        2,
        vec![0, 2, 4],
        vec![0, 1, 1, 2],
        vec![1.0, 2.0, 3.0, 4.0],
    );
    let x = vec![1.0, 2.0];
    let mut y_serial = vec![0.0; 3];
    csc.spmv(&x, &mut y_serial);
    let mut y_parallel = vec![0.0; 3];
    csc.spmv_parallel(&x, &mut y_parallel);
    assert_eq!(y_serial, y_parallel);
}

