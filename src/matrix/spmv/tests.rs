#[cfg(feature = "simd")]
use crate::matrix::spmv::{SpmvTuning, sellc, simd_csr};
use crate::matrix::{
    csc::CscMatrix,
    sparse::{CsrMatrix, SparseMatrix},
    spmv::{
        TBackend, spmm_csr_block, spmv_csr_parallel, spmv_scaled_csr, spmv_t_scaled_csr,
        t_spmv_csr_parallel,
    },
};

#[test]
fn scalar_kernel_matches_matrix_apply() {
    let a = CsrMatrix::from_csr(
        3,
        3,
        vec![0, 2, 4, 5],
        vec![0, 2, 1, 2, 0],
        vec![1.0, -1.0, 2.0, 3.0, 4.0],
    );
    let x = vec![1.0, 0.5, -2.0];
    let mut y_ref = vec![0.0; 3];
    a.spmv_scaled(1.0, &x, 0.0, &mut y_ref).unwrap();

    let mut y = vec![1.5; 3];
    spmv_scaled_csr(
        a.nrows(),
        a.row_ptr(),
        a.col_idx(),
        a.values(),
        1.0,
        &x,
        0.0,
        &mut y,
    );
    assert_eq!(y, y_ref);

    // Check scaling branch as well.
    let mut y_scale = vec![2.0; 3];
    spmv_scaled_csr(
        a.nrows(),
        a.row_ptr(),
        a.col_idx(),
        a.values(),
        0.5,
        &x,
        2.0,
        &mut y_scale,
    );
    for (lhs, rhs) in y_scale.iter().zip(y_ref.iter()) {
        assert!((lhs - (2.0 * 2.0 + 0.5 * rhs)).abs() < 1e-12);
    }
}

#[test]
fn scalar_kernel_transpose_matches_matrix_apply() {
    let a = CsrMatrix::from_csr(
        2,
        3,
        vec![0, 2, 4],
        vec![0, 1, 1, 2],
        vec![1.0, 2.0, 3.0, 4.0],
    );
    let x = vec![1.0, -1.0];
    let mut y_ref = vec![0.0; 3];
    a.spmv_transpose_scaled(1.0, &x, 0.0, &mut y_ref).unwrap();

    let mut y = vec![0.0; 3];
    spmv_t_scaled_csr(
        a.nrows(),
        a.row_ptr(),
        a.col_idx(),
        a.values(),
        1.0,
        &x,
        0.0,
        &mut y,
    );
    assert_eq!(y, y_ref);
}

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

#[cfg(feature = "simd")]
#[test]
fn simd_gather_matches_scalar_kernel() {
    let a = CsrMatrix::from_csr(
        4,
        4,
        vec![0, 3, 5, 7, 9],
        vec![0, 1, 3, 0, 2, 1, 3, 0, 2],
        vec![2.0, -1.0, 0.5, 3.0, 4.0, -2.0, 1.5, 0.25, 2.25],
    );
    let x = vec![0.5, -1.0, 2.0, 1.5];
    let mut y_scalar = vec![0.0; a.nrows()];
    spmv_scaled_csr(
        a.nrows(),
        a.row_ptr(),
        a.col_idx(),
        a.values(),
        1.0,
        &x,
        0.0,
        &mut y_scalar,
    );

    let mut y_simd = vec![0.0; a.nrows()];
    simd_csr::spmv_scaled_csr_simd_gather(
        a.nrows(),
        a.row_ptr(),
        a.col_idx(),
        a.values(),
        1.0,
        &x,
        0.0,
        &mut y_simd,
        2,
    );

    for (lhs, rhs) in y_scalar.iter().zip(y_simd.iter()) {
        assert!((lhs - rhs).abs() <= 1e-12);
    }
}

#[cfg(feature = "simd")]
#[test]
fn sellc_kernel_matches_scalar() {
    let a = CsrMatrix::from_csr(
        5,
        5,
        vec![0, 2, 5, 7, 9, 11],
        vec![0, 3, 1, 2, 4, 0, 3, 1, 4, 0, 2],
        vec![1.0, 0.5, -1.0, 2.5, -0.5, 3.0, 0.75, -2.0, 1.5, 0.8, 2.2],
    );
    let x = vec![1.0, -0.5, 0.25, 1.5, -1.25];
    let mut y_scalar = vec![0.0; a.nrows()];
    spmv_scaled_csr(
        a.nrows(),
        a.row_ptr(),
        a.col_idx(),
        a.values(),
        1.0,
        &x,
        0.0,
        &mut y_scalar,
    );

    let storage = sellc::csr_to_sellc(
        a.nrows(),
        a.ncols(),
        a.row_ptr(),
        a.col_idx(),
        a.values(),
        4,
        8,
    );
    let mut y_sellc = vec![0.0; a.nrows()];
    sellc::spmv_scaled_sellc(&storage, 1.0, &x, 0.0, &mut y_sellc, 2);

    for (lhs, rhs) in y_scalar.iter().zip(y_sellc.iter()) {
        assert!((lhs - rhs).abs() <= 1e-12);
    }
}

#[cfg(feature = "simd")]
#[test]
fn plan_apply_matches_scalar_results() {
    let mut a = CsrMatrix::from_csr(
        3,
        3,
        vec![0, 2, 4, 5],
        vec![0, 2, 1, 2, 0],
        vec![1.0, -1.0, 2.0, 3.0, 4.0],
    );
    let tuning = SpmvTuning {
        allow_simd: true,
        prefer_sellc: false,
        sell_c: 4,
        sell_sigma: 8,
        bench_nsamples: 0,
        min_nnz_for_simd: 0,
    };
    a.build_spmv_plan(&tuning);

    let x = vec![0.5, -2.0, 1.0];
    let mut y_plan = vec![0.0; a.nrows()];
    a.spmv_scaled(1.0, &x, 0.0, &mut y_plan).unwrap();

    let mut y_scalar = vec![0.0; a.nrows()];
    spmv_scaled_csr(
        a.nrows(),
        a.row_ptr(),
        a.col_idx(),
        a.values(),
        1.0,
        &x,
        0.0,
        &mut y_scalar,
    );

    for (lhs, rhs) in y_plan.iter().zip(y_scalar.iter()) {
        assert!((lhs - rhs).abs() <= 1e-12);
    }
}
