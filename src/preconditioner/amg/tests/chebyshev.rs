use super::*;
use crate::preconditioner::PcSide;
use crate::preconditioner::chebyshev::{chebyshev_smooth_csr, estimate_lmax_sym};
use faer::Mat;

#[test]
fn chebyshev_damps_high_frequency_mode() -> Result<(), KError> {
    let n = 64;
    let a = poisson1d(n);
    let diag_inv = diag_inv_from_csr(&a)?;
    let cfg = AMGConfig::default();
    let d_sqrt = make_d_sqrt_inv(&diag_inv);
    let cheb = compute_cheb_data(&cfg, &a, &diag_inv, &d_sqrt)?;
    let bounds = ChebBounds {
        lam_max: cheb.lambda_max,
        lam_min: cheb.lambda_min,
    };

    let mut rhs = vec![0.0; n];
    for i in 0..n {
        rhs[i] = if i % 2 == 0 { 1.0 } else { -1.0 };
    }
    let norm0 = l2_norm(&rhs);
    let mut z = vec![0.0; n];
    let mut ws = AMGWorkspace::new(n);
    let degree = 4;
    chebyshev_smooth_csr(
        &a,
        &diag_inv,
        &rhs,
        &mut z,
        degree,
        &bounds,
        &mut ws.residual[..n],
        &mut ws.temp[..n],
        &mut ws.work[..n],
    )?;

    let mut az = vec![0.0; n];
    a.spmv_scaled(1.0, &z, 0.0, &mut az)?;
    let mut res = vec![0.0; n];
    for i in 0..n {
        res[i] = rhs[i] - az[i];
    }
    let norm1 = l2_norm(&res);
    assert!(
        norm1 < 0.8 * norm0,
        "Chebyshev smoothing failed to damp residual: {norm1} vs {norm0}"
    );

    Ok(())
}

#[test]
fn chebyshev_recompute_toggle_and_safety() -> Result<(), KError> {
    let a = poisson1d(8);
    let mut amg = AMGBuilder::new()
        .relaxation_type(RelaxType::Chebyshev)
        .grid_relax_type_all(RelaxType::Chebyshev)
        .chebyshev_degree(2)
        .chebyshev_recompute_esteig(false)
        .chebyshev_safety(1.15)
        .build(&Mat::<f64>::zeros(0, 0))?;
    amg.setup(&a)?;

    let lam0 = amg.state.as_ref().unwrap().levels[0]
        .cheb
        .as_ref()
        .unwrap()
        .lambda_max;

    let mut a_diag = a.clone();
    {
        let row_ptr = a_diag.row_ptr().to_vec();
        let col_idx = a_diag.col_idx().to_vec();
        let nrows = a_diag.nrows();
        let vals = a_diag.values_mut();
        for i in 0..nrows {
            for p in row_ptr[i]..row_ptr[i + 1] {
                if col_idx[p] == i {
                    vals[p] *= 1.3;
                }
            }
        }
    }
    amg.update_numeric(&a_diag)?;
    let lam_no_recompute = amg.state.as_ref().unwrap().levels[0]
        .cheb
        .as_ref()
        .unwrap()
        .lambda_max;
    assert!(
        (lam_no_recompute - lam0).abs() < 1e-10,
        "lambda_max changed despite recompute=false: {lam_no_recompute} vs {lam0}"
    );

    amg.cfg.chebyshev_recompute_esteig = true;
    amg.update_numeric(&a_diag)?;
    let lam_new = amg.state.as_ref().unwrap().levels[0]
        .cheb
        .as_ref()
        .unwrap()
        .lambda_max;
    assert!(
        (lam_new - lam0).abs() > 1e-6,
        "lambda_max did not change after recompute: {lam_new} vs {lam0}"
    );

    let diag_inv2 = diag_inv_from_csr(&a_diag)?;
    let d_sqrt2 = make_d_sqrt_inv(&diag_inv2);
    let raw = estimate_lmax_sym(&a_diag, &d_sqrt2, amg.cfg.chebyshev_power_steps)?;
    assert!(
        lam_new >= raw * amg.cfg.chebyshev_safety * 0.98,
        "lambda_max {lam_new} does not respect safety factor over raw {raw}"
    );

    Ok(())
}

#[test]
fn pcg_with_chebyshev_smoother_converges() -> Result<(), KError> {
    let _guard = super::relax_lock()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let n = 64;
    let a = poisson1d(n);
    let mut amg = AMGBuilder::new()
        .relaxation_type(RelaxType::Chebyshev)
        .grid_relax_type_all(RelaxType::Chebyshev)
        .chebyshev_degree(4)
        .require_spd(true)
        .build(&Mat::<f64>::zeros(0, 0))?;
    amg.setup(&a)?;

    let rhs = vec![1.0; n];
    let mut x = vec![0.0; n];
    pcg_left_precond(&a, &rhs, &mut x, 1e-10, 200, |r, z| {
        amg.apply(PcSide::Left, r, z)
    })?;

    let mut ax = vec![0.0; n];
    a.spmv_scaled(1.0, &x, 0.0, &mut ax)?;
    let mut res_norm2 = 0.0;
    for i in 0..n {
        let diff = rhs[i] - ax[i];
        res_norm2 += diff * diff;
    }
    assert!(
        res_norm2.sqrt() < 1e-6,
        "PCG residual too large: {}",
        res_norm2.sqrt()
    );

    Ok(())
}

#[test]
fn chebyshev_reuses_workspace_buffers() -> Result<(), KError> {
    let n = 40;
    let a = poisson1d(n);
    let diag_inv = diag_inv_from_csr(&a)?;
    let cfg = AMGConfig::default();
    let d_sqrt = make_d_sqrt_inv(&diag_inv);
    let cheb = compute_cheb_data(&cfg, &a, &diag_inv, &d_sqrt)?;
    let bounds = ChebBounds {
        lam_max: cheb.lambda_max,
        lam_min: cheb.lambda_min,
    };
    let mut ws = AMGWorkspace::new(n);
    ws.ensure(n);
    let caps_before = (
        ws.temp.capacity(),
        ws.work.capacity(),
        ws.residual.capacity(),
        ws.coarse_rhs.capacity(),
        ws.fine_corr.capacity(),
    );

    let mut rhs = vec![0.0; n];
    for i in 0..n {
        rhs[i] = ((i * 37) as f64).sin();
    }
    let mut z = vec![0.0; n];
    chebyshev_smooth_csr(
        &a,
        &diag_inv,
        &rhs,
        &mut z,
        3,
        &bounds,
        &mut ws.residual[..n],
        &mut ws.temp[..n],
        &mut ws.work[..n],
    )?;

    let caps_after = (
        ws.temp.capacity(),
        ws.work.capacity(),
        ws.residual.capacity(),
        ws.coarse_rhs.capacity(),
        ws.fine_corr.capacity(),
    );
    assert_eq!(
        caps_before, caps_after,
        "workspace vectors reallocated during smoothing"
    );

    Ok(())
}
