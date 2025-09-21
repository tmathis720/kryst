use super::*;
use crate::preconditioner::PcSide;

#[test]
fn mixed_precision_jacobi_matches_baseline() -> Result<(), KError> {
    let _guard = super::relax_lock()
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let n = 40;
    let a = poisson1d(n);

    let mut amg_ref = AMGBuilder::new()
        .relaxation_type(RelaxType::Jacobi)
        .grid_relax_type_all(RelaxType::Jacobi)
        .build(&Mat::<f64>::zeros(0, 0))?;
    amg_ref.setup(&a)?;

    let mut amg_mixed = AMGBuilder::new()
        .relaxation_type(RelaxType::Jacobi)
        .grid_relax_type_all(RelaxType::Jacobi)
        .mixed_precision(Some(MixedPrecision {
            smooth: true,
            residual: false,
        }))
        .build(&Mat::<f64>::zeros(0, 0))?;
    amg_mixed.setup(&a)?;

    let rhs = vec![1.0; n];
    let mut z_ref = vec![0.0; n];
    let mut z_mixed = vec![0.0; n];
    amg_ref.apply(PcSide::Left, &rhs, &mut z_ref)?;
    amg_mixed.apply(PcSide::Left, &rhs, &mut z_mixed)?;

    let mut max_diff = 0.0f64;
    for i in 0..n {
        max_diff = max_diff.max((z_ref[i] - z_mixed[i]).abs());
    }
    assert!(
        max_diff < 1e-5,
        "Mixed-precision Jacobi deviates by {max_diff}",
    );
    Ok(())
}

#[test]
fn mixed_precision_residual_matches_baseline_norm() -> Result<(), KError> {
    let _guard = super::relax_lock()
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let n = 48;
    let a = poisson1d(n);

    let mut amg_ref = AMGBuilder::new()
        .relaxation_type(RelaxType::Jacobi)
        .grid_relax_type_all(RelaxType::Jacobi)
        .build(&Mat::<f64>::zeros(0, 0))?;
    amg_ref.setup(&a)?;

    let mut amg_mp = AMGBuilder::new()
        .relaxation_type(RelaxType::Jacobi)
        .grid_relax_type_all(RelaxType::Jacobi)
        .mixed_precision(Some(MixedPrecision {
            smooth: true,
            residual: true,
        }))
        .build(&Mat::<f64>::zeros(0, 0))?;
    amg_mp.setup(&a)?;

    let mut rhs = vec![0.0; n];
    for i in 0..n {
        rhs[i] = ((i * 17) as f64).sin();
    }
    let mut z_ref = vec![0.0; n];
    let mut z_mp = vec![0.0; n];
    amg_ref.apply(PcSide::Left, &rhs, &mut z_ref)?;
    amg_mp.apply(PcSide::Left, &rhs, &mut z_mp)?;

    let mut az_ref = vec![0.0; n];
    let mut az_mp = vec![0.0; n];
    a.spmv_scaled(1.0, &z_ref, 0.0, &mut az_ref)?;
    a.spmv_scaled(1.0, &z_mp, 0.0, &mut az_mp)?;
    let mut res_ref = 0.0;
    let mut res_mp = 0.0;
    for i in 0..n {
        let diff_ref = rhs[i] - az_ref[i];
        let diff_mp = rhs[i] - az_mp[i];
        res_ref += diff_ref * diff_ref;
        res_mp += diff_mp * diff_mp;
    }
    let res_ref = res_ref.sqrt();
    let res_mp = res_mp.sqrt();
    assert!(
        (res_ref - res_mp).abs() < 1e-5,
        "Residual norms diverged: {res_ref} vs {res_mp}",
    );
    Ok(())
}

#[test]
fn mixed_precision_transient_storage_runs() -> Result<(), KError> {
    let _guard = super::relax_lock()
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let n = 32;
    let a = poisson1d(n);

    let mut amg_cached = AMGBuilder::new()
        .relaxation_type(RelaxType::Jacobi)
        .grid_relax_type_all(RelaxType::Jacobi)
        .mixed_precision(Some(MixedPrecision {
            smooth: true,
            residual: true,
        }))
        .mixed_storage(MixedStorage::Cached)
        .build(&Mat::<f64>::zeros(0, 0))?;
    amg_cached.setup(&a)?;

    let mut amg_transient = AMGBuilder::new()
        .relaxation_type(RelaxType::Jacobi)
        .grid_relax_type_all(RelaxType::Jacobi)
        .mixed_precision(Some(MixedPrecision {
            smooth: true,
            residual: true,
        }))
        .mixed_storage(MixedStorage::Transient)
        .build(&Mat::<f64>::zeros(0, 0))?;
    amg_transient.setup(&a)?;

    let rhs = vec![1.0; n];
    let mut z_cached = vec![0.0; n];
    let mut z_transient = vec![0.0; n];
    amg_cached.apply(PcSide::Left, &rhs, &mut z_cached)?;
    amg_transient.apply(PcSide::Left, &rhs, &mut z_transient)?;

    let mut az_cached = vec![0.0; n];
    let mut az_transient = vec![0.0; n];
    a.spmv_scaled(1.0, &z_cached, 0.0, &mut az_cached)?;
    a.spmv_scaled(1.0, &z_transient, 0.0, &mut az_transient)?;

    let res_cached = az_cached
        .iter()
        .zip(rhs.iter())
        .map(|(az, b)| {
            let diff = b - az;
            diff * diff
        })
        .sum::<f64>()
        .sqrt();
    let res_transient = az_transient
        .iter()
        .zip(rhs.iter())
        .map(|(az, b)| {
            let diff = b - az;
            diff * diff
        })
        .sum::<f64>()
        .sqrt();

    assert!(
        res_transient <= res_cached * 1.5 + 1e-6,
        "Transient storage residual {res_transient} exceeds cached residual {res_cached}"
    );
    Ok(())
}

#[test]
fn mixed_precision_refresh_updates_cached_values() -> Result<(), KError> {
    let _guard = super::relax_lock()
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let n = 20;
    let a = poisson1d(n);

    let mut amg = AMGBuilder::new()
        .relaxation_type(RelaxType::Jacobi)
        .grid_relax_type_all(RelaxType::Jacobi)
        .mixed_precision(Some(MixedPrecision {
            smooth: true,
            residual: true,
        }))
        .mixed_storage(MixedStorage::Cached)
        .build(&Mat::<f64>::zeros(0, 0))?;
    amg.setup(&a)?;

    let lvl0 = &amg.state.as_ref().unwrap().levels[0];
    let original = lvl0
        .a_vals_f32
        .as_ref()
        .expect("missing cached values")
        .clone();

    let mut a_scaled = a.clone();
    for val in a_scaled.values_mut() {
        *val *= 1.2;
    }
    AMG::refresh_numeric(&mut amg, &a_scaled)?;
    let lvl_after = &amg.state.as_ref().unwrap().levels[0];
    let updated = lvl_after
        .a_vals_f32
        .as_ref()
        .expect("missing cached values after refresh")
        .clone();
    assert_ne!(original, updated, "Cached f32 values not refreshed");
    Ok(())
}
