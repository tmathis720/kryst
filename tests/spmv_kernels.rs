#[cfg(feature = "rayon")]
use kryst::matrix::{spmv, CsrMatrix};

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
    spmv::spmv_csr_parallel(&csr, &x, &mut y_parallel).unwrap();
    assert_eq!(y_serial, y_parallel);
}

#[cfg(feature = "rayon")]
#[test]
fn t_spmv_csr_parallel_matches_serial_csc_backend() {
    // 2x3 matrix [[1,2,0],[0,3,4]]
    let csr = CsrMatrix::from_csr(
        2,
        3,
        vec![0, 2, 4],
        vec![0, 1, 1, 2],
        vec![1.0, 2.0, 3.0, 4.0],
    );
    let csc = csr.to_csc_cached(0.0);
    let x = vec![5.0, 6.0];
    let mut y_serial = vec![0.0; 3];
    csc.t_matvec(&x, &mut y_serial);
    let mut y_parallel = vec![0.0; 3];
    spmv::t_spmv_csr_parallel(
        &csr,
        spmv::TBackend::Csc(&csc),
        &x,
        &mut y_parallel,
    )
    .unwrap();
    assert_eq!(y_serial, y_parallel);
}

#[cfg(feature = "rayon")]
#[test]
fn t_spmv_csr_parallel_matches_serial_gather() {
    // 2x3 matrix [[1,2,0],[0,3,4]]
    let csr = CsrMatrix::from_csr(
        2,
        3,
        vec![0, 2, 4],
        vec![0, 1, 1, 2],
        vec![1.0, 2.0, 3.0, 4.0],
    );
    let x = vec![5.0, 6.0];
    let mut y_serial = vec![0.0; 3];
    let csc = csr.to_csc_cached(0.0);
    csc.t_matvec(&x, &mut y_serial);
    let mut y_parallel = vec![0.0; 3];
    spmv::t_spmv_csr_parallel(&csr, spmv::TBackend::CsrGather, &x, &mut y_parallel).unwrap();
    assert_eq!(y_serial, y_parallel);
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
        vec![1.0, 2.0, 3.0, 4.0],
    );
    let x1 = vec![1.0, 2.0, 3.0];
    let x2 = vec![4.0, 5.0, 6.0];
    let mut y1 = vec![0.0; 2];
    let mut y2 = vec![0.0; 2];
    csr.spmv(&x1, &mut y1);
    csr.spmv(&x2, &mut y2);

    let mut y1b = vec![0.0; 2];
    let mut y2b = vec![0.0; 2];
    let x_cols = [&x1[..], &x2[..]];
    let mut y_cols = [&mut y1b[..], &mut y2b[..]];
    spmv::spmm_csr_block(&csr, 2, &x_cols, &mut y_cols).unwrap();
    assert_eq!(y1, y_cols[0]);
    assert_eq!(y2, y_cols[1]);
}

