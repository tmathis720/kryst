use crate::matrix::sparse::CsrMatrix;
use crate::preconditioner::Preconditioner;
use crate::preconditioner::ilu_csr::{
    IluCsr, IluCsrConfig, IluKind, PivotStrategy, IlutParams, PivotPolicy, Pivoting,
};

fn tridiag_csr(n: usize, a: f64, b: f64, c: f64) -> CsrMatrix<f64> {
    // main diag b, subdiag a, superdiag c
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::with_capacity(3 * n);
    let mut vals = Vec::with_capacity(3 * n);
    row_ptr.push(0);
    for i in 0..n {
        if i > 0 {
            col_idx.push(i - 1);
            vals.push(a);
        }
        col_idx.push(i);
        vals.push(b);
        if i + 1 < n {
            col_idx.push(i + 1);
            vals.push(c);
        }
        row_ptr.push(col_idx.len());
    }
    CsrMatrix::from_csr(n, n, row_ptr, col_idx, vals)
}

#[test]
fn iluk_basic_pivots_nonzero() {
    let n = 8;
    let a = tridiag_csr(n, -1.0, 4.0, -1.0);
    let cfg = IluCsrConfig {
        kind: IluKind::Iluk { k: 1 },
        pivot: PivotStrategy::DiagonalPerturbation,
        pivot_threshold: 1e-12,
        diag_perturb_factor: 1e-10,
        level_sched: cfg!(feature = "rayon"),
        numeric_update_fixed: true,
        logging: 0,
    };
    let mut pc = IluCsr::new_with_config(cfg);
    pc.setup(&a).unwrap();

    // Check diagonal entries in U are finite/nonzero
    for i in 0..n {
        let dix = pc.u_diag_ix()[i];
        let d = pc.u_val()[dix];
        assert!(d.is_finite());
        assert_ne!(d, 0.0);
    }

    // Apply on a vector and ensure finite results
    let x = vec![1.0; n];
    let mut y = vec![0.0; n];
    pc.apply(crate::preconditioner::PcSide::Left, &x, &mut y)
        .unwrap();
    assert!(y.iter().all(|v| v.is_finite()));
}

#[test]
fn ilut_basic_and_numeric_update_keeps_pattern() {
    let n = 10;
    let a = tridiag_csr(n, -1.0, 4.0, -1.0);
    let params = IlutParams {
        droptol_abs: 1e-6,
        droptol_rel: 0.0,
        p_l: 2,
        p_u: 2,
        early_drop: true,
        pivot: PivotPolicy::DiagonalPerturbation,
        pivot_tau: 1e-12,
        reproducible_order: true,
        pivoting: Pivoting::None,
    };
    let cfg = IluCsrConfig {
        kind: IluKind::Ilut { params },
        pivot: PivotStrategy::DiagonalPerturbation,
        pivot_threshold: 1e-12,
        diag_perturb_factor: 1e-10,
        level_sched: false,
        numeric_update_fixed: true,
        logging: 0,
    };
    let mut pc = IluCsr::new_with_config(cfg);
    pc.setup(&a).unwrap();

    // Snapshot structure
    let lcol0 = pc.l_col().to_vec();
    let ucol0 = pc.u_col().to_vec();
    let lrow0 = pc.l_row().to_vec();
    let urow0 = pc.u_row().to_vec();
    let udiag0 = pc.u_diag_ix().to_vec();

    // Build a numerically changed matrix with identical structure: scale values
    let mut a2 = a.clone();
    for v in a2.values_mut() {
        *v *= 2.0;
    }

    // Update numeric only
    pc.update_numeric(&a2).unwrap();

    // Structure unchanged
    assert_eq!(lcol0, pc.l_col());
    assert_eq!(ucol0, pc.u_col());
    assert_eq!(lrow0, pc.l_row());
    assert_eq!(urow0, pc.u_row());
    assert_eq!(udiag0, pc.u_diag_ix());

    // Apply and ensure finite results
    let x = vec![1.0; n];
    let mut y = vec![0.0; n];
    pc.apply(crate::preconditioner::PcSide::Left, &x, &mut y)
        .unwrap();
    assert!(y.iter().all(|v| v.is_finite()));
}

#[test]
fn milu0_preserves_row_sums() {
    use crate::preconditioner::ilu_csr::IluKind;
    // 4x4 "arrow" matrix that induces dropped fill in ILU(0)
    // [2 1 1 1; 1 2 0 0; 1 0 2 0; 1 0 0 2]
    let n = 4;
    let row_ptr = vec![0, 4, 6, 8, 10];
    let col_idx = vec![0, 1, 2, 3, 0, 1, 0, 2, 0, 3];
    let vals = vec![2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0];
    let a = CsrMatrix::from_csr(n, n, row_ptr, col_idx, vals);

    // Row sums of original matrix
    let mut a_row_sum = vec![0.0; n];
    for i in 0..n {
        let rs = a.row_ptr()[i];
        let re = a.row_ptr()[i + 1];
        for p in rs..re {
            a_row_sum[i] += a.values()[p];
        }
    }

    // Factor with MILU(0)
    let cfg = IluCsrConfig {
        kind: IluKind::Milu0,
        pivot: PivotStrategy::DiagonalPerturbation,
        pivot_threshold: 1e-12,
        diag_perturb_factor: 1e-10,
        level_sched: false,
        numeric_update_fixed: true,
        logging: 0,
    };
    let mut pc = IluCsr::new_with_config(cfg);
    pc.setup(&a).unwrap();

    // Compute M*1 where M = L*U
    let ones = vec![1.0; n];
    // z = U * 1
    let mut z = vec![0.0; n];
    for i in 0..n {
        for p in pc.u_row()[i]..pc.u_row()[i + 1] {
            let j = pc.u_col()[p];
            z[i] += pc.u_val()[p] * ones[j];
        }
    }
    // y = L * z (L has unit diagonal)
    let mut y = z.clone();
    for i in 0..n {
        for p in pc.l_row()[i]..pc.l_row()[i + 1] {
            let j = pc.l_col()[p];
            y[i] += pc.l_val()[p] * z[j];
        }
    }

    for i in 0..n {
        let diff = (y[i] - a_row_sum[i]).abs();
        assert!(diff < 1e-9, "row {} diff {}", i, diff);
    }
}
