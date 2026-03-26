#![cfg(feature = "backend-faer")]

use crate::error::KError;
use crate::matrix::spmv::SpmvTuning;
#[cfg(all(feature = "simd", not(feature = "complex")))]
use crate::matrix::spmv::{sellc, simd_csr};
use crate::matrix::{
    csc::CscMatrix,
    csr::CsrMatrix as GenericCsrMatrix,
    sparse::CsrMatrix,
    spmv::{
        TBackend, spmm_csr_block, spmm_csr_dense, spmv_csr_parallel, spmv_scaled_csr,
        spmv_t_scaled_csr, t_spmv_csr_parallel,
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
fn spmv_plan_scalar_matches_kernel() {
    use crate::matrix::spmv::plan;

    let matrix = GenericCsrMatrix::new(
        4,
        4,
        vec![0, 2, 4, 7, 8],
        vec![0, 3, 1, 2, 0, 2, 3, 1],
        vec![1.0, -2.0, 3.0, 4.0, 0.5, -1.5, 2.0, 1.25],
    );
    let tuning = SpmvTuning {
        allow_simd: false,
        ..Default::default()
    };
    let plan = plan::build(&matrix, &tuning);

    let x = vec![0.75, -1.0, 0.5, 2.0];
    let mut y_plan = vec![0.0; matrix.nrows];
    plan.apply_scaled(1.0, &x, 0.0, &mut y_plan);

    let mut y_ref = vec![0.0; matrix.nrows];
    crate::matrix::spmv::spmv_scaled_csr(
        matrix.nrows,
        &matrix.rowptr,
        &matrix.colind,
        &matrix.values,
        1.0,
        &x,
        0.0,
        &mut y_ref,
    );

    for (lhs, rhs) in y_plan.iter().zip(y_ref.iter()) {
        assert!((lhs - rhs).abs() < 1e-12);
    }
}

#[test]
fn spmv_csr_scalar_matches_real_reference() {
    use crate::algebra::prelude::*;

    let a_real = GenericCsrMatrix::new(
        3,
        3,
        vec![0, 2, 3, 4],
        vec![0, 2, 1, 2],
        vec![1.0, 2.0, -3.0, 4.0],
    );
    let x_real = vec![1.0, 2.0, -1.0];
    let mut y_real = vec![0.0; 3];
    spmv_scaled_csr(
        a_real.nrows,
        &a_real.rowptr,
        &a_real.colind,
        &a_real.values,
        1.0,
        &x_real,
        0.0,
        &mut y_real,
    );

    let a_scalar = GenericCsrMatrix::new(
        3,
        3,
        vec![0, 2, 3, 4],
        vec![0, 2, 1, 2],
        vec![
            S::from_real(1.0),
            S::from_real(2.0),
            S::from_real(-3.0),
            S::from_real(4.0),
        ],
    );
    let x_scalar: Vec<S> = x_real.iter().copied().map(S::from_real).collect();
    let mut y_scalar = vec![S::zero(); 3];
    crate::matrix::spmv::spmv_csr_scalar(&a_scalar, &x_scalar, &mut y_scalar);

    for i in 0..3 {
        assert!((y_scalar[i].real() - y_real[i]).abs() < 1e-12);
        #[cfg(feature = "complex")]
        {
            assert_eq!(y_scalar[i], y_scalar[i].conj());
        }
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
fn csr_matvec_matches_sparse_method() {
    use crate::algebra::prelude::*;

    // 2×3 matrix [[1,2,0],[0,3,4]]
    let a = CsrMatrix::from_csr(
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
    let x = vec![S::one(), S::one(), S::one()];
    let mut y_api = vec![S::zero(); 2];
    let mut y_method = vec![S::zero(); 2];

    crate::matrix::spmv::csr_matvec(&a, &x, &mut y_api).unwrap();
    a.spmv(&x, &mut y_method);

    assert_eq!(y_api, y_method);
}

#[test]
fn csr_t_matvec_matches_transpose_method() {
    use crate::algebra::prelude::*;

    // 2×3 matrix [[1,2,0],[0,3,4]]; transpose is 3×2
    let a = CsrMatrix::from_csr(
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
    let x = vec![S::from_real(1.0), S::from_real(2.0)];
    let mut y_api = vec![S::zero(); 3];
    let mut y_method = vec![S::zero(); 3];

    crate::matrix::spmv::csr_t_matvec(&a, &x, &mut y_api).unwrap();
    a.spmv_transpose_scaled(S::one(), &x, S::zero(), &mut y_method)
        .unwrap();

    assert_eq!(y_api, y_method);
}

#[test]
fn csr_matvec_dim_mismatch_errors() {
    use crate::algebra::prelude::*;

    let a = CsrMatrix::from_csr(1, 2, vec![0, 1], vec![0], vec![S::from_real(1.0)]);
    let mut y = vec![S::zero(); 1];
    let err =
        crate::matrix::spmv::csr_matvec(&a, &[S::one(), S::one(), S::one()], &mut y).unwrap_err();
    assert!(matches!(err, KError::InvalidInput(_)));
}

#[cfg(feature = "rayon")]
#[test]
fn csr_matvec_par_dim_mismatch_errors() {
    use crate::algebra::prelude::*;

    let a = CsrMatrix::from_csr(1, 1, vec![0, 1], vec![0], vec![S::from_real(2.0)]);
    let mut y = vec![S::zero(); 1];
    let err = crate::matrix::spmv::csr_matvec_par(&a, &[], &mut y).unwrap_err();
    assert!(matches!(err, KError::InvalidInput(_)));
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

fn make_block_spmm_fixture(n: usize) -> CsrMatrix {
    let mut rp = Vec::with_capacity(n + 1);
    let mut ci = Vec::new();
    let mut vv = Vec::new();
    rp.push(0);
    for i in 0..n {
        ci.push(i);
        vv.push(2.0);
        if i + 1 < n {
            ci.push(i + 1);
            vv.push(-0.25);
        }
        if i > 0 {
            ci.push(i - 1);
            vv.push(0.5);
        }
        rp.push(ci.len());
    }
    CsrMatrix::from_csr(n, n, rp, ci, vv)
}

fn fill_block_input(n: usize, p: usize) -> crate::context::ksp_context::BlockVec {
    let mut x = crate::context::ksp_context::BlockVec::new(n, p);
    for col in 0..p {
        let xcol = x.col_mut(col);
        for (i, xi) in xcol.iter_mut().enumerate() {
            *xi = ((i + 1 + col) as f64) / ((col + 2) as f64);
        }
    }
    x
}

#[test]
fn spmm_csr_dense_matches_columnwise_spmv_for_block_gmres_sizes() {
    let n = 96;
    let a = make_block_spmm_fixture(n);
    for &p in &[1usize, 2, 4, 8] {
        let x = fill_block_input(n, p);
        let mut y = crate::context::ksp_context::BlockVec::new(n, p);
        spmm_csr_dense(&a, &x, &mut y).expect("spmm_csr_dense should succeed");

        for col in 0..p {
            let mut y_ref = vec![0.0; n];
            a.spmv(x.col(col), &mut y_ref);
            assert_eq!(y.col(col), &y_ref);
        }
    }
}

#[cfg(feature = "rayon")]
#[test]
fn spmm_csr_dense_parallel_tuning_paths_match_serial_for_arnoldi_block_sizes() {
    use crate::algebra::parallel_cfg::{parallel_tune, serial_guard, set_parallel_tune};

    let n = 160;
    let a = make_block_spmm_fixture(n);
    for &p in &[2usize, 4, 8] {
        let x = fill_block_input(n, p);
        let mut y_parallel = crate::context::ksp_context::BlockVec::new(n, p);
        let mut y_serial = crate::context::ksp_context::BlockVec::new(n, p);

        let orig = parallel_tune();
        let mut tuned = orig;
        tuned.min_work_spmm_dense = 1;
        tuned.chunk_rows_spmm_dense = 16;
        tuned.chunk_cols_spmm_dense = 2;
        set_parallel_tune(tuned);

        spmm_csr_dense(&a, &x, &mut y_parallel).expect("parallel spmm should succeed");
        let _guard = serial_guard(true);
        spmm_csr_dense(&a, &x, &mut y_serial).expect("serial spmm should succeed");
        set_parallel_tune(orig);

        assert_eq!(y_parallel.as_slice(), y_serial.as_slice());
    }
}

#[cfg(all(feature = "simd", not(feature = "complex")))]
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

#[cfg(all(feature = "simd", not(feature = "complex")))]
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

#[cfg(all(feature = "simd", not(feature = "complex")))]
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
