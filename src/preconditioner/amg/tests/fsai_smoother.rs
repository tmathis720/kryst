use super::*;
use crate::preconditioner::PcSide;

fn poisson2d_anisotropic(nx: usize, ny: usize, eps: f64) -> CsrMatrix<f64> {
    let n = nx * ny;
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::new();
    let mut vals = Vec::new();
    row_ptr.push(0);
    for iy in 0..ny {
        for ix in 0..nx {
            let idx = iy * nx + ix;
            let mut entries: Vec<(usize, f64)> = Vec::with_capacity(5);
            if ix > 0 {
                entries.push((idx - 1, -1.0));
            }
            if ix + 1 < nx {
                entries.push((idx + 1, -1.0));
            }
            if iy > 0 {
                entries.push((idx - nx, -eps));
            }
            if iy + 1 < ny {
                entries.push((idx + nx, -eps));
            }
            entries.push((idx, 2.0 * (1.0 + eps)));
            entries.sort_unstable_by(|a, b| a.0.cmp(&b.0));
            for (c, v) in entries {
                col_idx.push(c);
                vals.push(v);
            }
            row_ptr.push(col_idx.len());
        }
    }
    CsrMatrix::from_csr(n, n, row_ptr, col_idx, vals)
}

#[test]
fn fsai_damps_high_frequency_mode() -> Result<(), KError> {
    let n = 64;
    let a = poisson1d(n);
    let mut amg = AMGBuilder::new()
        .relaxation_type(RelaxType::Fsai)
        .grid_relax_type_all(RelaxType::Fsai)
        .build(&Mat::<f64>::zeros(0, 0))?;
    amg.setup(&a)?;

    let lvl = &amg.state.as_ref().unwrap().levels[0];
    let fsai = lvl.fsai.as_ref().expect("missing FSAI data");
    let mut rhs = vec![0.0; n];
    for i in 0..n {
        rhs[i] = if i % 2 == 0 { 1.0 } else { -1.0 };
    }
    let norm0 = l2_norm(&rhs);

    let mut z_fsai = vec![0.0; n];
    let mut ws_fsai = AMGWorkspace::new(n);
    AMG::fsai_smooth(
        &fsai.g,
        &fsai.gt,
        &lvl.a,
        &rhs,
        &mut z_fsai,
        amg.cfg.fsai_damping,
        &mut ws_fsai,
    )?;
    AMG::fsai_smooth(
        &fsai.g,
        &fsai.gt,
        &lvl.a,
        &rhs,
        &mut z_fsai,
        amg.cfg.fsai_damping,
        &mut ws_fsai,
    )?;
    let mut az = vec![0.0; n];
    lvl.a.spmv_scaled(1.0, &z_fsai, 0.0, &mut az)?;
    let mut res_fsai = vec![0.0; n];
    for i in 0..n {
        res_fsai[i] = rhs[i] - az[i];
    }
    let norm_fsai = l2_norm(&res_fsai);

    let mut z_jacobi = vec![0.0; n];
    let mut ws_j = AMGWorkspace::new(n);
    AMG::jacobi_smooth_sparse(
        amg.cfg.jacobi_omega,
        &lvl.a,
        &lvl.diag_inv,
        &rhs,
        &mut z_jacobi,
        2,
        &mut ws_j,
    )?;
    lvl.a.spmv_scaled(1.0, &z_jacobi, 0.0, &mut az)?;
    let mut res_jacobi = vec![0.0; n];
    for i in 0..n {
        res_jacobi[i] = rhs[i] - az[i];
    }
    let norm_jacobi = l2_norm(&res_jacobi);

    assert!(
        norm_fsai < 0.75 * norm0,
        "FSAI failed to damp residual: {norm_fsai} vs {norm0}"
    );
    assert!(
        norm_fsai <= 5.0 * norm_jacobi,
        "FSAI residual {norm_fsai} exceeds tolerance relative to Jacobi {norm_jacobi}"
    );

    Ok(())
}

#[test]
fn fsai_handles_anisotropy() -> Result<(), KError> {
    let a = poisson2d_anisotropic(12, 12, 1e-2);
    let n = a.nrows();
    let mut amg = AMGBuilder::new()
        .relaxation_type(RelaxType::Fsai)
        .grid_relax_type_all(RelaxType::Fsai)
        .fsai_use_strength(false)
        .build(&Mat::<f64>::zeros(0, 0))?;
    amg.setup(&a)?;

    let lvl = &amg.state.as_ref().unwrap().levels[0];
    let fsai = lvl.fsai.as_ref().expect("missing FSAI data");
    let mut rhs = vec![0.0; n];
    for i in 0..n {
        rhs[i] = ((i * 7) as f64).sin();
    }
    let mut z = vec![0.0; n];
    let mut ws = AMGWorkspace::new(n);
    let tau = amg.cfg.fsai_damping;
    AMG::fsai_smooth(&fsai.g, &fsai.gt, &lvl.a, &rhs, &mut z, tau, &mut ws)?;
    AMG::fsai_smooth(&fsai.g, &fsai.gt, &lvl.a, &rhs, &mut z, tau, &mut ws)?;
    let mut az = vec![0.0; n];
    lvl.a.spmv_scaled(1.0, &z, 0.0, &mut az)?;
    let mut res = vec![0.0; n];
    for i in 0..n {
        res[i] = rhs[i] - az[i];
    }
    let norm_initial = l2_norm(&rhs);
    let norm_final = l2_norm(&res);
    assert!(
        norm_final < 0.6 * norm_initial,
        "Anisotropic residual not reduced enough: {norm_final} vs {norm_initial}"
    );
    Ok(())
}

#[test]
fn pcg_with_fsai_converges() -> Result<(), KError> {
    let _guard = super::relax_lock().lock().unwrap();
    let n = 64;
    let a = poisson1d(n);
    let mut amg = AMGBuilder::new()
        .relaxation_type(RelaxType::Fsai)
        .grid_relax_type_all(RelaxType::Fsai)
        .build(&Mat::<f64>::zeros(0, 0))?;
    amg.setup(&a)?;

    let rhs = vec![1.0; n];
    let mut x = vec![0.0; n];
    pcg_left_precond(&a, &rhs, &mut x, 1e-10, 200, |r, z| {
        amg.apply(PcSide::Left, r, z)
    })?;
    let mut ax = vec![0.0; n];
    a.spmv_scaled(1.0, &x, 0.0, &mut ax)?;
    let mut res2 = 0.0;
    for i in 0..n {
        let diff = rhs[i] - ax[i];
        res2 += diff * diff;
    }
    assert!(
        res2.sqrt() < 1e-6,
        "PCG residual too large: {}",
        res2.sqrt()
    );
    Ok(())
}

#[test]
fn fsai_numeric_refresh_updates_values() -> Result<(), KError> {
    let n = 40;
    let a = poisson1d(n);
    let mut amg = AMGBuilder::new()
        .relaxation_type(RelaxType::Fsai)
        .grid_relax_type_all(RelaxType::Fsai)
        .build(&Mat::<f64>::zeros(0, 0))?;
    amg.setup(&a)?;
    let g_before = amg.state.as_ref().unwrap().levels[0]
        .fsai
        .as_ref()
        .unwrap()
        .g
        .clone();

    let mut a_scaled = a.clone();
    for v in a_scaled.values_mut() {
        *v *= 1.1;
    }
    amg.update_numeric(&a_scaled)?;
    let fsai_after = amg.state.as_ref().unwrap().levels[0].fsai.as_ref().unwrap();
    assert_eq!(g_before.row_ptr(), fsai_after.g.row_ptr());
    assert_eq!(g_before.col_idx(), fsai_after.g.col_idx());
    let mut diff_norm = 0.0;
    for (old, new) in g_before.values().iter().zip(fsai_after.g.values().iter()) {
        diff_norm += (*old - *new) * (*old - *new);
    }
    assert!(
        diff_norm.sqrt() > 1e-6,
        "FSAI values did not change appreciably"
    );
    Ok(())
}

#[test]
fn fsai_build_is_deterministic() -> Result<(), KError> {
    let a = poisson1d(48);
    let make_amg = || {
        AMGBuilder::new()
            .relaxation_type(RelaxType::Fsai)
            .grid_relax_type_all(RelaxType::Fsai)
            .fsai_use_strength(false)
            .build(&Mat::<f64>::zeros(0, 0))
    };
    let mut amg1 = make_amg()?;
    let mut amg2 = make_amg()?;
    amg1.setup(&a)?;
    amg2.setup(&a)?;
    let fsai1 = amg1.state.as_ref().unwrap().levels[0]
        .fsai
        .as_ref()
        .unwrap();
    let fsai2 = amg2.state.as_ref().unwrap().levels[0]
        .fsai
        .as_ref()
        .unwrap();
    assert_eq!(fsai1.g.row_ptr(), fsai2.g.row_ptr());
    assert_eq!(fsai1.g.col_idx(), fsai2.g.col_idx());
    assert_eq!(fsai1.g.values(), fsai2.g.values());
    Ok(())
}
