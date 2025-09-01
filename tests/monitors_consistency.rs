use faer::Mat;
use kryst::context::ksp_context::{KspContext, SolverType};
use kryst::context::pc_context::PcType;
use kryst::preconditioner::PcSide;
use kryst::matrix::op::LinOp;
use std::sync::Arc;

fn diag_mat(vals: &[f64]) -> Mat<f64> {
    let n = vals.len();
    Mat::from_fn(n, n, |i, j| if i == j { vals[i] } else { 0.0 })
}

fn nrm2(x: &[f64]) -> f64 {
    x.iter().map(|&v| v * v).sum::<f64>().sqrt()
}

#[test]
fn monitors_reported_norms_and_final_true_residual() {
    // Small diagonal system so Jacobi is exact
    let d = [2.0, 3.0, 4.0, 5.0, 6.0];
    let a = diag_mat(&d);
    let n = d.len();
    let b = vec![1.0; n];

    // Expected norms
    let bnorm = nrm2(&b);
    let mut minv_b = vec![0.0; n];
    for i in 0..n { minv_b[i] = b[i] / d[i]; }
    let minv_b_norm = nrm2(&minv_b);

    // Helper to run a solver and capture first monitor value and final residual
    // Helper to recompute true residual norm
    let true_res_norm = |a: &Mat<f64>, b: &[f64], x: &[f64]| -> f64 {
        let n = b.len();
        let mut ax = vec![0.0; n];
        a.matvec(x, &mut ax);
        let mut r = vec![0.0; n];
        for i in 0..n { r[i] = b[i] - ax[i]; }
        nrm2(&r)
    };

    let run = |solver: SolverType, side: PcSide| -> (f64, f64, Vec<f64>) {
        let mut ksp = KspContext::new();
        ksp.set_type(solver).unwrap();
        ksp.set_pc_type(PcType::Jacobi, None).unwrap();
        ksp.pc_side = side;
        let amat: Arc<dyn LinOp<S = f64>> = Arc::new(a.clone());
        ksp.set_operators(amat, None);

        use std::sync::{Arc, Mutex};
        let first: Arc<Mutex<Option<f64>>> = Arc::new(Mutex::new(None));
        let first_clone = Arc::clone(&first);
        ksp.add_monitor(move |iter, res| {
            if iter == 0 {
                let mut slot = first_clone.lock().unwrap();
                *slot = Some(res);
            }
        });

        let mut x = vec![0.0; n];
        let stats = ksp.solve(&b, &mut x).unwrap();
        (first.lock().unwrap().unwrap_or(0.0), stats.final_residual, x)
    };

    // CG (Left) with no PC: monitors fall back to true residual, final is true
    let mut ksp = KspContext::new();
    ksp.set_tolerances(1e-8, 1e-50, 1e5, 10000);
    ksp.set_type(SolverType::Cg).unwrap();
    ksp.set_pc_type(PcType::None, None).unwrap();
    ksp.pc_side = PcSide::Left;
    let amat: Arc<dyn LinOp<S = f64>> = Arc::new(a.clone());
    ksp.set_operators(amat, None);
    use std::sync::{Arc as SArc, Mutex as SMutex};
    let first_cg: SArc<SMutex<Option<f64>>> = SArc::new(SMutex::new(None));
    let first_cg_cl = SArc::clone(&first_cg);
    ksp.add_monitor(move |iter, res| {
        if iter == 0 { *first_cg_cl.lock().unwrap() = Some(res); }
    });
    let mut x = vec![0.0; n];
    let stats = ksp.solve(&b, &mut x).unwrap();
    let first = first_cg.lock().unwrap().unwrap_or(0.0);
    assert!((first - bnorm).abs() < 1e-12);
    let final_r = stats.final_residual;
    
    // CG should converge to exact on diagonal with Jacobi
    assert!(final_r <= 1e-5);

    // GMRES Left: monitors preconditioned; GMRES Right: monitors true
    let (first_l, final_l, x_l) = run(SolverType::Gmres, PcSide::Left);
    assert!((first_l - minv_b_norm).abs() < 1e-12);
    let res_true_l = true_res_norm(&a, &b, &x_l);
    assert!((final_l - res_true_l).abs() < 1e-5);

    let (first_r, final_r2, x_r) = run(SolverType::Gmres, PcSide::Right);
    assert!((first_r - bnorm).abs() < 1e-12);
    let res_true_r = true_res_norm(&a, &b, &x_r);
    assert!((final_r2 - res_true_r).abs() < 1e-5);

    // CGS and QMR: monitors true residual
    let (cgs_first, _cgs_final) = {
        // run CGS with no PC
        let mut ksp = KspContext::new();
        ksp.set_type(SolverType::Cgs).unwrap();
        ksp.set_pc_type(PcType::None, None).unwrap();
        ksp.pc_side = PcSide::Left;
        let amat: Arc<dyn LinOp<S = f64>> = Arc::new(a.clone());
        ksp.set_operators(amat, None);
        let first: SArc<SMutex<Option<f64>>> = SArc::new(SMutex::new(None));
        let first_cl = SArc::clone(&first);
        ksp.add_monitor(move |iter, res| { if iter == 0 { *first_cl.lock().unwrap() = Some(res); } });
        let mut x = vec![0.0; n];
        let stats = ksp.solve(&b, &mut x).unwrap();
        let res_true = true_res_norm(&a, &b, &x);
        assert!((stats.final_residual - res_true).abs() < 1e-12);
        (first.lock().unwrap().unwrap_or(0.0), stats.final_residual)
    };
    assert!((cgs_first - bnorm).abs() < 1e-12);

    let (qmr_first, _qmr_final) = {
        // run QMR with no PC
        let mut ksp = KspContext::new();
        ksp.set_type(SolverType::Qmr).unwrap();
        ksp.set_pc_type(PcType::None, None).unwrap();
        ksp.pc_side = PcSide::Left;
        let amat: Arc<dyn LinOp<S = f64>> = Arc::new(a.clone());
        ksp.set_operators(amat, None);
        let first: SArc<SMutex<Option<f64>>> = SArc::new(SMutex::new(None));
        let first_cl = SArc::clone(&first);
        ksp.add_monitor(move |iter, res| { if iter == 0 { *first_cl.lock().unwrap() = Some(res); } });
        let mut x = vec![0.0; n];
        let stats = ksp.solve(&b, &mut x).unwrap();
        let res_true = true_res_norm(&a, &b, &x);
        assert!((stats.final_residual - res_true).abs() < 1e-12);
        (first.lock().unwrap().unwrap_or(0.0), stats.final_residual)
    };
    assert!((qmr_first - bnorm).abs() < 1e-12);
}
