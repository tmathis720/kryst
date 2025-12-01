use super::*;
use crate::preconditioner::{PcSide, Preconditioner};
use faer::Mat;

#[test]
#[cfg(not(feature = "complex"))]
fn v_cycle_policy_reports_single_visit() -> Result<(), KError> {
    let a = poisson1d(16);
    let mut amg = AMGBuilder::new()
        .relaxation_type(RelaxType::Jacobi)
        .grid_relax_type_all(RelaxType::Jacobi)
        .cycle_v()
        .build(&Mat::<f64>::zeros(0, 0))?;
    amg.setup(&a)?;
    assert_eq!(amg.cycle_policy.gamma_visits(0), 1);
    Ok(())
}

#[test]
#[cfg(not(feature = "complex"))]
fn w_cycle_policy_uses_configured_gamma() -> Result<(), KError> {
    let a = poisson1d(16);
    let mut amg = AMGBuilder::new()
        .relaxation_type(RelaxType::Jacobi)
        .grid_relax_type_all(RelaxType::Jacobi)
        .cycle_w(3)
        .build(&Mat::<f64>::zeros(0, 0))?;
    amg.setup(&a)?;
    assert_eq!(amg.cycle_policy.gamma_visits(0), 3);
    Ok(())
}

#[test]
#[cfg(not(feature = "complex"))]
fn kcycle_postsmoother_reduces_residual() -> Result<(), KError> {
    let a = poisson1d(64);
    let rhs = vec![1.0; a.nrows()];
    let mut res = vec![0.0; a.nrows()];

    let mut amg_v = AMGBuilder::new()
        .relaxation_type(RelaxType::Jacobi)
        .grid_relax_type_all(RelaxType::Jacobi)
        .cycle_v()
        .build(&Mat::<f64>::zeros(0, 0))?;
    amg_v.setup(&a)?;
    let mut z_v = vec![0.0; a.nrows()];
    amg_v.apply(PcSide::Left, &rhs, &mut z_v)?;
    a.spmv_scaled(1.0, &z_v, 0.0, &mut res)?;
    for i in 0..a.nrows() {
        res[i] = rhs[i] - res[i];
    }
    let norm_v = l2_norm(&res);

    let kc = KCycle {
        levels: vec![0],
        iters: 2,
        algo: KrylovAlgo::FCG,
        place_post: true,
        place_pre: false,
    };
    let mut amg_k = AMGBuilder::new()
        .relaxation_type(RelaxType::Jacobi)
        .grid_relax_type_all(RelaxType::Jacobi)
        .cycle_v()
        .kcycle(kc)
        .build(&Mat::<f64>::zeros(0, 0))?;
    amg_k.setup(&a)?;
    let mut z_k = vec![0.0; a.nrows()];
    res.fill(0.0);
    amg_k.apply(PcSide::Left, &rhs, &mut z_k)?;
    a.spmv_scaled(1.0, &z_k, 0.0, &mut res)?;
    for i in 0..a.nrows() {
        res[i] = rhs[i] - res[i];
    }
    let norm_k = l2_norm(&res);

    assert!(
        norm_k <= norm_v * 0.95,
        "K-cycle residual norm {norm_k} should improve over V-cycle {norm_v}"
    );
    Ok(())
}

#[test]
#[cfg(not(feature = "complex"))]
fn kcycle_is_deterministic() -> Result<(), KError> {
    let a = poisson1d(32);
    let kc = KCycle {
        levels: vec![0, 1],
        iters: 2,
        algo: KrylovAlgo::FCG,
        place_post: true,
        place_pre: true,
    };
    let mut amg = AMGBuilder::new()
        .relaxation_type(RelaxType::Jacobi)
        .grid_relax_type_all(RelaxType::Jacobi)
        .cycle_w(2)
        .kcycle(kc)
        .build(&Mat::<f64>::zeros(0, 0))?;
    amg.setup(&a)?;

    let rhs = vec![1.0; a.nrows()];
    let mut z1 = vec![0.0; a.nrows()];
    let mut z2 = vec![0.0; a.nrows()];

    amg.apply(PcSide::Left, &rhs, &mut z1)?;
    amg.apply(PcSide::Left, &rhs, &mut z2)?;

    for i in 0..a.nrows() {
        assert!(
            (z1[i] - z2[i]).abs() < 1e-12,
            "solution mismatch at row {i}"
        );
    }
    Ok(())
}
