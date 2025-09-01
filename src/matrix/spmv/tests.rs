use crate::matrix::{
    csc::CscMatrix,
    sparse::{CsrMatrix, SparseMatrix},
    spmv::{spmm_csr_block, spmv_csr_parallel, t_spmv_csr_parallel, TBackend},
};

#[test]
fn spmv_matches_reference_small() {
    // A = [[1,2,0],[0,3,4]]
    let a = CsrMatrix::from_csr(
        2,
        3,
        vec![0, 2, 4],
        vec![0, 1, 1, 2],
        vec![1.0, 2.0, 3.0, 4.0],
    );
    let x = vec![1.0, 1.0, 1.0];
    let mut y_ref = vec![0.0; 2];
    a.spmv(&x, &mut y_ref); // reference
    let mut y = vec![0.0; 2];
    spmv_csr_parallel(&a, &x, &mut y).unwrap();
    assert_eq!(y, y_ref); // bitwise parity on this case
}

#[test]
fn tspmv_matches_csc_path() {
    // Rectangular: 3x2 then test A^T * x (size 2)
    let a = CsrMatrix::from_csr(3, 2, vec![0, 1, 3, 3], vec![0, 0, 1], vec![5.0, 7.0, 9.0]);
    let x = vec![1.0, 2.0, 3.0];
    let mut y_csr = vec![0.0; 2];
    t_spmv_csr_parallel(&a, TBackend::CsrGather, &x, &mut y_csr).unwrap();

    let csc = CscMatrix::from_csc(3, 2, vec![0, 2, 3], vec![0, 1, 1], vec![5.0, 7.0, 9.0]);
    let mut y_csc = vec![0.0; 2];
    t_spmv_csr_parallel(&a, TBackend::Csc(&csc), &x, &mut y_csc).unwrap();

    assert_eq!(y_csr, y_csc);
}

#[test]
fn spmm_block_s_two_rhs() {
    // A: 2x3 as above
    let a = CsrMatrix::from_csr(
        2,
        3,
        vec![0, 2, 4],
        vec![0, 1, 1, 2],
        vec![1.0, 2.0, 3.0, 4.0],
    );
    let x0 = vec![1.0, 1.0, 1.0];
    let x1 = vec![2.0, 0.5, 0.0];
    let mut y0 = vec![0.0; 2];
    let mut y1 = vec![0.0; 2];

    spmm_csr_block(&a, 2, &[&x0, &x1], &mut [&mut y0, &mut y1]).unwrap();

    let mut r0 = vec![0.0; 2];
    let mut r1 = vec![0.0; 2];
    a.spmv(&x0, &mut r0);
    a.spmv(&x1, &mut r1);
    assert_eq!(y0, r0);
    assert_eq!(y1, r1);
}
