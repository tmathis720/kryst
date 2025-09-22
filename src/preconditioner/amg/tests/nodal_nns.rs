use super::super::AMGBuilder;
use super::super::prolong::{
    TentativeNodal, TentativeP, smooth_tentative_sa_mf, smooth_tentative_sa_multi,
};
use crate::matrix::sparse::{CsrMatrix, SparseMatrix};
use crate::preconditioner::Preconditioner;
use faer::Mat;

fn sample_block_matrix() -> CsrMatrix<f64> {
    let row_ptr = vec![0, 3, 6, 9, 12];
    let col_idx = vec![
        0, 1, 2, // row 0
        0, 1, 3, // row 1
        0, 2, 3, // row 2
        1, 2, 3, // row 3
    ];
    let vals = vec![
        4.0, -1.0, -0.5, // row 0
        -1.0, 5.0, -1.0, // row 1
        -0.5, -1.0, 5.0, // row 2
        -1.0, -0.5, 4.0, // row 3
    ];
    CsrMatrix::from_csr(4, 4, row_ptr, col_idx, vals)
}

#[test]
fn nodal_row_basis_is_locally_orthonormal() {
    let _guard = super::relax_lock()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());

    let a = sample_block_matrix();

    let mut amg = AMGBuilder::new()
        .nodal(true, 2)
        .num_functions(2)
        .strong_threshold(0.25)
        .coarse_threshold(1)
        .max_coarse_size(1)
        .max_levels(2)
        .require_spd(false)
        .build(&Mat::<f64>::zeros(0, 0))
        .unwrap();
    amg.setup(&a).unwrap();

    let h = amg.state.as_ref().unwrap();
    let level0 = &h.levels[0];
    assert_eq!(level0.num_functions, 2);
    let row_basis = level0
        .row_basis
        .as_ref()
        .expect("row_basis populated for nodal level");
    let agg_of = &level0.agg_of;
    let diag_inv = &level0.diag_inv;
    let mut diag = vec![0.0; diag_inv.len()];
    for (i, &inv) in diag_inv.iter().enumerate() {
        diag[i] = if inv.abs() > 0.0 { 1.0 / inv } else { 0.0 };
    }
    let mfun = level0.num_functions;
    let n_agg = 1 + agg_of.iter().copied().max().unwrap_or(0);
    for g in 0..n_agg {
        let rows: Vec<usize> = agg_of
            .iter()
            .enumerate()
            .filter_map(|(i, &agg)| if agg == g { Some(i) } else { None })
            .collect();
        if rows.is_empty() {
            continue;
        }
        let mut gram = vec![0.0f64; mfun * mfun];
        for (f1, f2) in (0..mfun).flat_map(|i| (0..mfun).map(move |j| (i, j))) {
            let mut sum = 0.0;
            for &i in &rows {
                let w = diag[i].abs().max(1e-30);
                let v1 = row_basis[i * mfun + f1];
                let v2 = row_basis[i * mfun + f2];
                sum += w * v1 * v2;
            }
            gram[f1 * mfun + f2] = sum;
        }
        for f in 0..mfun {
            let norm = gram[f * mfun + f];
            if norm > 1e-9 {
                assert!(
                    (norm - 1.0).abs() <= 1e-9,
                    "aggregate {g} column {f} norm {norm}"
                );
            } else {
                assert!(norm.abs() <= 1e-12);
            }
            for f2 in (f + 1)..mfun {
                let off = gram[f * mfun + f2];
                assert!(
                    off.abs() <= 1e-9,
                    "aggregate {g} columns {f}/{f2} not orthogonal: {off}"
                );
            }
        }
    }
}

#[test]
fn nodal_tentative_is_deterministic() {
    let _guard = super::relax_lock()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());

    let a = sample_block_matrix();

    let build_once = || {
        let mut amg = AMGBuilder::new()
            .nodal(true, 2)
            .num_functions(2)
            .strong_threshold(0.25)
            .coarse_threshold(1)
            .max_coarse_size(1)
            .max_levels(2)
            .require_spd(false)
            .build(&Mat::<f64>::zeros(0, 0))
            .unwrap();
        amg.setup(&a).unwrap();
        let level0 = &amg.state.as_ref().unwrap().levels[0];
        (
            level0.p.row_ptr().to_vec(),
            level0.p.col_idx().to_vec(),
            level0.p.values().to_vec(),
        )
    };

    let (rp1, ci1, vv1) = build_once();
    let (rp2, ci2, vv2) = build_once();

    assert_eq!(rp1, rp2, "row_ptr differs between builds");
    assert_eq!(ci1, ci2, "col_idx differs between builds");
    assert_eq!(vv1.len(), vv2.len());
    for (k, (&v1, &v2)) in vv1.iter().zip(vv2.iter()).enumerate() {
        assert!(
            (v1 - v2).abs() <= 1e-12,
            "value mismatch at entry {k}: {v1} vs {v2}"
        );
    }
}

#[test]
fn restriction_respects_component_modes() {
    let _guard = super::relax_lock()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());

    let a = sample_block_matrix();
    let mut amg = AMGBuilder::new()
        .nodal(true, 2)
        .num_functions(2)
        .strong_threshold(0.25)
        .coarse_threshold(1)
        .max_coarse_size(1)
        .max_levels(2)
        .require_spd(false)
        .build(&Mat::<f64>::zeros(0, 0))
        .unwrap();
    amg.setup(&a).unwrap();

    let state = amg.state.as_ref().unwrap();
    let level0 = &state.levels[0];
    let layout = level0
        .layout
        .as_ref()
        .expect("layout present for nodal hierarchy");
    let mfun = level0.num_functions;
    assert_eq!(mfun, 2);
    let mut yc = vec![0.0f64; level0.p.ncols()];

    for k in 0..mfun {
        let mut phi = vec![0.0f64; a.nrows()];
        for i in 0..phi.len() {
            if layout.comp_of[i] == k {
                phi[i] = 1.0;
            }
        }
        level0.r.spmv(&phi, &mut yc);
        for (idx, &val) in yc.iter().enumerate() {
            let comp = idx % mfun;
            if comp == k {
                assert!(
                    val.abs() > 1e-8,
                    "component {k} lost mass at coarse index {idx}: {val}"
                );
            } else {
                assert!(
                    val.abs() <= 1e-12,
                    "component {k} leaked into column {idx} with value {val}"
                );
            }
        }
    }
}

#[test]
fn scalar_and_nodal_singleton_match() {
    let _guard = super::relax_lock()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());

    let row_ptr = vec![0, 4, 8, 12, 16];
    let col_idx = vec![
        0, 1, 2, 3, // row 0
        0, 1, 2, 3, // row 1
        0, 1, 2, 3, // row 2
        0, 1, 2, 3, // row 3
    ];
    let vals = vec![
        6.0, -1.0, -0.8, -0.2, // row 0
        -1.0, 5.5, -0.7, -0.4, // row 1
        -0.8, -0.7, 5.8, -1.1, // row 2
        -0.2, -0.4, -1.1, 5.2, // row 3
    ];
    let a = CsrMatrix::from_csr(4, 4, row_ptr, col_idx, vals);
    let mut d_inv = vec![0.0f64; a.nrows()];
    for i in 0..a.nrows() {
        let mut diag: f64 = 0.0;
        for p in a.row_ptr()[i]..a.row_ptr()[i + 1] {
            if a.col_idx()[p] == i {
                diag = a.values()[p];
                break;
            }
        }
        assert!(diag.abs() > 1e-12, "diagonal vanished at row {i}");
        d_inv[i] = 1.0 / diag;
    }

    let agg_of = vec![0, 0, 1, 1];
    let tp = TentativeP {
        agg_of: agg_of.clone(),
        n_coarse: 2,
        num_functions: 1,
        nns: Some(vec![vec![1.0; a.nrows()]]),
        comp_of: None,
    };
    let tn = TentativeNodal {
        agg_of: agg_of.clone(),
        n_agg: 2,
        mfun: 1,
        row_basis: vec![1.0; a.nrows()],
    };

    let omega = 0.8;
    let drop_tol = 0.0;
    let max_per_row = 0usize;
    let trunc_rel = 0.0;

    let p_scalar =
        smooth_tentative_sa_multi(&a, &d_inv, &tp, omega, drop_tol, max_per_row, trunc_rel);
    let p_nodal = smooth_tentative_sa_mf(&a, &d_inv, &tn, omega, drop_tol, max_per_row);

    assert_eq!(p_scalar.row_ptr, p_nodal.row_ptr);
    assert_eq!(p_scalar.col_idx, p_nodal.col_idx);
    assert_eq!(p_scalar.vals.len(), p_nodal.vals.len());
    for (k, (&vs, &vn)) in p_scalar.vals.iter().zip(p_nodal.vals.iter()).enumerate() {
        assert!(
            (vs - vn).abs() <= 1e-12,
            "entry {k} mismatch: scalar={vs}, nodal={vn}"
        );
    }
}

#[test]
fn nodal_numeric_refresh_updates_values() {
    let _guard = super::relax_lock()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());

    let a = sample_block_matrix();
    let mut amg = AMGBuilder::new()
        .nodal(true, 2)
        .num_functions(2)
        .strong_threshold(0.25)
        .coarse_threshold(1)
        .max_coarse_size(1)
        .max_levels(2)
        .require_spd(false)
        .build(&Mat::<f64>::zeros(0, 0))
        .unwrap();
    amg.setup(&a).unwrap();

    let state_before = amg.state.as_ref().unwrap();
    let p_vals_before = state_before.levels[0].p.values().to_vec();
    let coarse_vals_before = state_before.levels[1].a.values().to_vec();
    let p_pattern = (
        state_before.levels[0].p.row_ptr().to_vec(),
        state_before.levels[0].p.col_idx().to_vec(),
    );

    let mut a_scaled = a.clone();
    {
        let rp = a_scaled.row_ptr().to_vec();
        let cj = a_scaled.col_idx().to_vec();
        let nrows = a_scaled.nrows();
        let vals = a_scaled.values_mut();
        for i in 0..nrows {
            for p in rp[i]..rp[i + 1] {
                if cj[p] == i {
                    let factor = 1.05 + 0.1 * (i as f64);
                    vals[p] *= factor;
                } else {
                    let factor = 1.1 + 0.05 * ((i + cj[p]) as f64);
                    vals[p] *= factor;
                }
            }
        }
    }

    amg.update_numeric(&a_scaled).unwrap();

    let state_after = amg.state.as_ref().unwrap();
    let p_after = &state_after.levels[0].p;
    assert_eq!(p_pattern.0, p_after.row_ptr());
    assert_eq!(p_pattern.1, p_after.col_idx());

    let max_diff_p = p_vals_before
        .iter()
        .zip(p_after.values())
        .map(|(&before, &after)| (before - after).abs())
        .fold(0.0f64, f64::max);

    let coarse_after = state_after.levels[1].a.values();
    let max_diff_coarse = coarse_vals_before
        .iter()
        .zip(coarse_after.iter())
        .map(|(&before, &after)| (before - after).abs())
        .fold(0.0f64, f64::max);

    assert!(
        max_diff_p > 1e-12 || max_diff_coarse > 1e-12,
        "numeric refresh did not change P or A_c"
    );
}
