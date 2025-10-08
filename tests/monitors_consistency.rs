use faer::Mat;
use kryst::algebra::prelude::*;
use kryst::context::ksp_context::{KspContext, SolverType};
use kryst::context::pc_context::PcType;
use kryst::matrix::op::LinOp;
use kryst::preconditioner::PcSide;
use kryst::{assert_s_close, testkit};
use std::sync::Arc;

fn diag_mat(vals: &[R]) -> Mat<R> {
    let n = vals.len();
    Mat::from_fn(n, n, |i, j| if i == j { vals[i] } else { R::default() })
}

fn nrm2(x: &[R]) -> R {
    x.iter().map(|&v| v * v).sum::<R>().sqrt()
}

#[test]
fn monitors_reported_norms_and_final_true_residual() {
    // Small diagonal system so Jacobi is exact
    let d = [
        S::from_real(2.0).real(),
        S::from_real(3.0).real(),
        S::from_real(4.0).real(),
        S::from_real(5.0).real(),
        S::from_real(6.0).real(),
    ];
    let a = diag_mat(&d);
    let n = d.len();
    let b = vec![S::one().real(); n];

    // Expected norms
    let bnorm = nrm2(&b);
    let mut minv_b = vec![R::default(); n];
    for i in 0..n {
        minv_b[i] = b[i] / d[i];
    }
    let minv_b_norm = nrm2(&minv_b);

    // Helper to run a solver and capture first monitor value and final residual
    // Helper to recompute true residual norm
    let true_res_norm = |a: &Mat<R>, b: &[R], x: &[R]| -> R {
        let n = b.len();
        let mut ax = vec![R::default(); n];
        a.matvec(x, &mut ax);
        let mut r = vec![R::default(); n];
        for i in 0..n {
            r[i] = b[i] - ax[i];
        }
        nrm2(&r)
    };

    let run = |solver: SolverType, side: PcSide| -> (R, R, Vec<R>) {
        let mut ksp = KspContext::new();
        ksp.set_type(solver).unwrap();
        ksp.set_pc_type(PcType::Jacobi, None).unwrap();
        ksp.pc_side = side;
        let amat: Arc<dyn LinOp<S = f64>> = Arc::new(a.clone());
        ksp.set_operators(amat, None);

        use std::sync::{Arc, Mutex};
        let first: Arc<Mutex<Option<R>>> = Arc::new(Mutex::new(None));
        let first_clone = Arc::clone(&first);
        ksp.add_monitor(move |iter, res| {
            if iter == 0 {
                let mut slot = first_clone.lock().unwrap();
                *slot = Some(res);
            }
        });

        let mut x = vec![R::default(); n];
        let stats = ksp.solve(&b, &mut x).unwrap();
        (
            first.lock().unwrap().unwrap_or(R::default()),
            stats.final_residual,
            x,
        )
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
        if iter == 0 {
            *first_cg_cl.lock().unwrap() = Some(res);
        }
    });
    let mut x = vec![R::default(); n];
    let stats = ksp.solve(&b, &mut x).unwrap();
    let first = first_cg.lock().unwrap().unwrap_or(R::default());
    assert_s_close!("cg monitor first", S::from_real(first), S::from_real(bnorm));
    let final_r = stats.final_residual;

    // CG should converge to exact on diagonal with Jacobi
    assert!(final_r <= 1e-5);

    // GMRES Left: monitors preconditioned; GMRES Right: monitors true
    let (first_l, final_l, x_l) = run(SolverType::Gmres, PcSide::Left);
    assert_s_close!(
        "gmres left monitor",
        S::from_real(first_l),
        S::from_real(minv_b_norm)
    );
    let res_true_l = true_res_norm(&a, &b, &x_l);
    testkit::assert_s_close(
        "gmres left final",
        S::from_real(final_l),
        S::from_real(res_true_l),
        1e-5,
        testkit::RTOL,
    );

    let (first_r, final_r2, x_r) = run(SolverType::Gmres, PcSide::Right);
    assert_s_close!(
        "gmres right monitor",
        S::from_real(first_r),
        S::from_real(bnorm)
    );
    let res_true_r = true_res_norm(&a, &b, &x_r);
    testkit::assert_s_close(
        "gmres right final",
        S::from_real(final_r2),
        S::from_real(res_true_r),
        1e-5,
        testkit::RTOL,
    );

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
        ksp.add_monitor(move |iter, res| {
            if iter == 0 {
                *first_cl.lock().unwrap() = Some(res);
            }
        });
        let mut x = vec![R::default(); n];
        let stats = ksp.solve(&b, &mut x).unwrap();
        let res_true = true_res_norm(&a, &b, &x);
        assert_s_close!(
            "cgs final residual",
            S::from_real(stats.final_residual),
            S::from_real(res_true)
        );
        (
            first.lock().unwrap().unwrap_or(R::default()),
            stats.final_residual,
        )
    };
    assert_s_close!("cgs monitor", S::from_real(cgs_first), S::from_real(bnorm));

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
        ksp.add_monitor(move |iter, res| {
            if iter == 0 {
                *first_cl.lock().unwrap() = Some(res);
            }
        });
        let mut x = vec![R::default(); n];
        let stats = ksp.solve(&b, &mut x).unwrap();
        let res_true = true_res_norm(&a, &b, &x);
        assert_s_close!(
            "qmr final residual",
            S::from_real(stats.final_residual),
            S::from_real(res_true)
        );
        (
            first.lock().unwrap().unwrap_or(R::default()),
            stats.final_residual,
        )
    };
    assert_s_close!("qmr monitor", S::from_real(qmr_first), S::from_real(bnorm));
}
