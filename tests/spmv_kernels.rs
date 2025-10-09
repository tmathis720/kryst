use kryst::LinOp;
use kryst::algebra::prelude::*;
use kryst::assert_vec_close;
#[cfg(feature = "rayon")]
use kryst::matrix::{csc::CscMatrix, csr::CsrMatrix as ScalarCsrMatrix, spmv};

#[cfg(feature = "rayon")]
#[test]
fn csr_spmv_parallel_matches_serial() {
    // 2x3 matrix [[1,2,0],[0,3,4]]
    let csr = ScalarCsrMatrix::new(
        2,
        3,
        vec![0, 2, 4],
        vec![0, 1, 1, 2],
        vec![
            S::from_real(1.0),
            S::from_real(2.0),
            S::from_real(3.0),
            S::from_real(4.0),
        ],
    );
    let x = vec![S::from_real(1.0), S::from_real(2.0), S::from_real(3.0)];
    let mut y_serial = vec![S::zero(); 2];
    csr.spmv(&x, &mut y_serial);
    let mut y_parallel = vec![S::zero(); 2];
    spmv::spmv_csr_parallel(&csr, &x, &mut y_parallel).unwrap();
    assert_vec_close!("csr parallel matches serial", &y_serial, &y_parallel);
}

#[cfg(feature = "rayon")]
#[test]
fn t_spmv_csr_parallel_matches_serial_csc_backend() {
    // 2x3 matrix [[1,2,0],[0,3,4]]
    let csr = ScalarCsrMatrix::new(
        2,
        3,
        vec![0, 2, 4],
        vec![0, 1, 1, 2],
        vec![
            S::from_real(1.0),
            S::from_real(2.0),
            S::from_real(3.0),
            S::from_real(4.0),
        ],
    );
    let csc = CscMatrix::from_csc(
        2,
        3,
        vec![0, 1, 3, 4],
        vec![0, 0, 1, 1],
        vec![
            S::from_real(1.0),
            S::from_real(2.0),
            S::from_real(3.0),
            S::from_real(4.0),
        ],
    );
    let x = vec![S::from_real(5.0), S::from_real(6.0)];
    let mut y_serial = vec![S::zero(); 3];
    csc.t_matvec(&x, &mut y_serial);
    let mut y_parallel = vec![S::zero(); 3];
    spmv::t_spmv_csr_parallel(&csr, spmv::TBackend::Csc(&csc), &x, &mut y_parallel).unwrap();
    assert_vec_close!(
        "transpose csr parallel matches serial",
        &y_serial,
        &y_parallel
    );
}

#[cfg(feature = "rayon")]
#[test]
fn t_spmv_csr_parallel_matches_serial_gather() {
    // 2x3 matrix [[1,2,0],[0,3,4]]
    let csr = ScalarCsrMatrix::new(
        2,
        3,
        vec![0, 2, 4],
        vec![0, 1, 1, 2],
        vec![
            S::from_real(1.0),
            S::from_real(2.0),
            S::from_real(3.0),
            S::from_real(4.0),
        ],
    );
    let x = vec![S::from_real(5.0), S::from_real(6.0)];
    let csc = CscMatrix::from_csc(
        2,
        3,
        vec![0, 1, 3, 4],
        vec![0, 0, 1, 1],
        vec![
            S::from_real(1.0),
            S::from_real(2.0),
            S::from_real(3.0),
            S::from_real(4.0),
        ],
    );
    let mut y_serial = vec![S::zero(); 3];
    csc.t_matvec(&x, &mut y_serial);
    let mut y_parallel = vec![S::zero(); 3];
    spmv::t_spmv_csr_parallel(&csr, spmv::TBackend::CsrGather, &x, &mut y_parallel).unwrap();
    assert_vec_close!(
        "transpose csr gather matches serial",
        &y_serial,
        &y_parallel
    );
}

#[cfg(feature = "rayon")]
#[test]
fn spmm_csr_block_matches_serial() {
    // 2x3 matrix [[1,2,0],[0,3,4]]
    let csr = CsrMatrix::from_csr(
        2,
        3,
        vec![0, 2, 4],
        vec![0, 1, 1, 2],
        vec![
            S::from_real(1.0),
            S::from_real(2.0),
            S::from_real(3.0),
            S::from_real(4.0),
        ],
    );
    let x1 = vec![S::from_real(1.0), S::from_real(2.0), S::from_real(3.0)];
    let x2 = vec![S::from_real(4.0), S::from_real(5.0), S::from_real(6.0)];
    let mut y1 = vec![S::zero(); 2];
    let mut y2 = vec![S::zero(); 2];
    csr.spmv(&x1, &mut y1);
    csr.spmv(&x2, &mut y2);

    let mut y1b = vec![S::zero(); 2];
    let mut y2b = vec![S::zero(); 2];
    let x_cols = [&x1[..], &x2[..]];
    let mut y_cols = [&mut y1b[..], &mut y2b[..]];
    spmv::spmm_csr_block(&csr, 2, &x_cols, &mut y_cols).unwrap();
    assert_vec_close!("block column 0", &y1, y_cols[0]);
    assert_vec_close!("block column 1", &y2, y_cols[1]);
}
