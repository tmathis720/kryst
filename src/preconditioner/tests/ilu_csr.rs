use crate::algebra::prelude::*;
use crate::matrix::sparse::CsrMatrix;
use crate::matrix::utils::poisson_2d;
use crate::preconditioner::Preconditioner;
use crate::preconditioner::ilu_csr::{
    IluCsr, IluCsrConfig, IluKind, IlutParams, PivotPolicy, PivotStrategy, Pivoting,
    ReorderingKind, ReorderingOptions,
};

fn tridiag_csr(n: usize, a: R, b: R, c: R) -> CsrMatrix<R> {
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
fn ilu0_single_rank_matches_serial_ref() {
    use crate::preconditioner::PcSide;
    use oorandom::Rand64;

    let a = poisson_2d(10, 10);

    let cfg_serial = IluCsrConfig {
        kind: IluKind::Ilu0,
        pivot: PivotStrategy::DiagonalPerturbation,
        pivot_threshold: R::from(1e-12),
        diag_perturb_factor: R::from(1e-10),
        level_sched: false,
        numeric_update_fixed: true,
        logging: 0,
        reordering: ReorderingOptions::default(),
    };
    let mut cfg_par = cfg_serial.clone();
    cfg_par.level_sched = cfg!(feature = "rayon");

    let mut ilu_ref = IluCsr::new_with_config(cfg_serial);
    ilu_ref.setup(&a).unwrap();
    let mut ilu_par = IluCsr::new_with_config(cfg_par);
    ilu_par.setup(&a).unwrap();

    let mut rng = Rand64::new(123);
    for _ in 0..4 {
        let b: Vec<R> = (0..a.nrows())
            .map(|_| R::from(rng.rand_float()) - R::from(0.5))
            .collect();
        let mut y_ref = vec![R::default(); b.len()];
        let mut y_par = vec![R::default(); b.len()];
        ilu_ref.apply(PcSide::Left, &b, &mut y_ref).unwrap();
        ilu_par.apply(PcSide::Left, &b, &mut y_par).unwrap();
        let num = y_ref
            .iter()
            .zip(&y_par)
            .map(|(a, b)| (*a - *b) * (*a - *b))
            .sum::<R>()
            .sqrt();
        let den = y_ref
            .iter()
            .map(|a| *a * *a)
            .sum::<R>()
            .sqrt()
            .max(R::from(1e-30));
        assert!(num / den < R::from(5e-14), "relative diff {}", num / den);
    }
}

#[test]
fn iluk_basic_pivots_nonzero() {
    let n = 8;
    let a = tridiag_csr(n, R::from(-1.0), R::from(4.0), R::from(-1.0));
    let cfg = IluCsrConfig {
        kind: IluKind::Iluk { k: 1 },
        pivot: PivotStrategy::DiagonalPerturbation,
        pivot_threshold: R::from(1e-12),
        diag_perturb_factor: R::from(1e-10),
        level_sched: cfg!(feature = "rayon"),
        numeric_update_fixed: true,
        logging: 0,
        reordering: ReorderingOptions::default(),
    };
    let mut pc = IluCsr::new_with_config(cfg);
    pc.setup(&a).unwrap();

    // Check diagonal entries in U are finite/nonzero
    for i in 0..n {
        let dix = pc.u_diag_ix()[i];
        let d = pc.u_val()[dix];
        assert!(d.is_finite());
        assert!(d.abs() > R::from(1e-30));
    }

    // Apply on a vector and ensure finite results
    let x = vec![R::from(1.0); n];
    let mut y = vec![R::default(); n];
    pc.apply(crate::preconditioner::PcSide::Left, &x, &mut y)
        .unwrap();
    assert!(y.iter().all(|v| v.is_finite()));
}

#[test]
fn ilut_basic_and_numeric_update_keeps_pattern() {
    let n = 10;
    let a = tridiag_csr(n, R::from(-1.0), R::from(4.0), R::from(-1.0));
    let params = IlutParams {
        droptol_abs: R::from(1e-6),
        droptol_rel: R::default(),
        p_l: 2,
        p_u: 2,
        early_drop: true,
        pivot: PivotPolicy::DiagonalPerturbation,
        pivot_tau: R::from(1e-12),
        reproducible_order: true,
        pivoting: Pivoting::None,
    };
    let cfg = IluCsrConfig {
        kind: IluKind::Ilut { params },
        pivot: PivotStrategy::DiagonalPerturbation,
        pivot_threshold: R::from(1e-12),
        diag_perturb_factor: R::from(1e-10),
        level_sched: false,
        numeric_update_fixed: true,
        logging: 0,
        reordering: ReorderingOptions::default(),
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
        *v *= R::from(2.0);
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
    let x = vec![R::from(1.0); n];
    let mut y = vec![R::default(); n];
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
    let vals = vec![
        R::from(2.0),
        R::from(1.0),
        R::from(1.0),
        R::from(1.0),
        R::from(1.0),
        R::from(2.0),
        R::from(1.0),
        R::from(2.0),
        R::from(1.0),
        R::from(2.0),
    ];
    let a = CsrMatrix::from_csr(n, n, row_ptr, col_idx, vals);

    // Row sums of original matrix
    let mut a_row_sum = vec![R::default(); n];
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
        pivot_threshold: R::from(1e-12),
        diag_perturb_factor: R::from(1e-10),
        level_sched: false,
        numeric_update_fixed: true,
        logging: 0,
        reordering: ReorderingOptions::default(),
    };
    let mut pc = IluCsr::new_with_config(cfg);
    pc.setup(&a).unwrap();

    // Compute M*1 where M = L*U
    let ones: Vec<S> = vec![S::from_real(1.0); n];
    // z = U * 1
    let mut z = vec![S::zero(); n];
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
        let diff = (y[i].real() - a_row_sum[i]).abs();
        assert!(diff < R::from(1e-9), "row {} diff {}", i, diff);
        assert!(
            y[i].imag().abs() < R::from(1e-12),
            "row {} imag {}",
            i,
            y[i].imag()
        );
    }
}

#[test]
fn ilu0_with_rcm_setup() {
    let n = 8;
    let a = tridiag_csr(n, R::from(-1.0), R::from(4.0), R::from(-1.0));
    let cfg = IluCsrConfig {
        kind: IluKind::Ilu0,
        pivot: PivotStrategy::DiagonalPerturbation,
        pivot_threshold: R::from(1e-12),
        diag_perturb_factor: R::from(1e-10),
        level_sched: false,
        numeric_update_fixed: true,
        logging: 0,
        reordering: ReorderingOptions {
            kind: ReorderingKind::Rcm,
            symmetric: true,
            deterministic: true,
        },
    };
    let mut pc = IluCsr::new_with_config(cfg);
    pc.setup(&a).unwrap();
    let x = vec![R::from(1.0); n];
    let mut y = vec![R::default(); n];
    pc.apply(crate::preconditioner::PcSide::Left, &x, &mut y)
        .unwrap();
    assert!(y.iter().all(|v| v.is_finite()));
}

#[cfg(feature = "complex")]
#[test]
fn ilu_csr_apply_s_matches_real_path() {
    use crate::algebra::bridge::BridgeScratch;
    use crate::algebra::prelude::*;
    use crate::ops::kpc::KPreconditioner;
    use crate::preconditioner::PcSide;

    let n = 5;
    let a = tridiag_csr(n, R::from(-1.0), R::from(4.0), R::from(-1.0));
    let cfg = IluCsrConfig {
        kind: IluKind::Ilu0,
        pivot: PivotStrategy::DiagonalPerturbation,
        pivot_threshold: R::from(1e-12),
        diag_perturb_factor: R::from(1e-10),
        level_sched: false,
        numeric_update_fixed: true,
        logging: 0,
        reordering: ReorderingOptions::default(),
    };
    let mut pc = IluCsr::new_with_config(cfg);
    pc.setup(&a).expect("ilu setup");

    let rhs_real: Vec<R> = (0..n).map(|i| R::from((i + 1) as f64)).collect();
    let mut out_real = vec![R::default(); n];
    pc.apply(PcSide::Left, &rhs_real, &mut out_real)
        .expect("ilu apply real");

    let rhs_s: Vec<S> = rhs_real.iter().copied().map(S::from_real).collect();
    let mut out_s = vec![S::zero(); n];
    let mut scratch = BridgeScratch::default();
    pc.apply_s(PcSide::Left, &rhs_s, &mut out_s, &mut scratch)
        .expect("ilu apply_s");

    for (ys, yr) in out_s.iter().zip(out_real.iter()) {
        assert!(
            (ys.real() - yr).abs() < R::from(1e-12),
            "expected real match, got {} vs {}",
            ys,
            yr
        );
    }
}
