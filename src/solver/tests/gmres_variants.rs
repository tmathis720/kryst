use crate::algebra::prelude::*;
use crate::context::ksp_context::Workspace;
use crate::error::KError;
use crate::parallel::{NoComm, UniverseComm};
use crate::preconditioner::PcSide;
use crate::solver::gmres::{GmresSolver, GmresVariant};
use crate::solver::{MonitorAction, MonitorCallback};
use std::sync::{Arc, Mutex};

use super::util;

fn solve_with_variant(
    a: &crate::matrix::sparse::CsrMatrix<f64>,
    b: &[R],
    variant: GmresVariant,
    restart: usize,
) -> Result<(Vec<R>, crate::utils::convergence::SolveStats<R>, R), KError> {
    let mut solver = GmresSolver::new(restart, 1e-8, 2_000);
    solver.set_variant(variant);
    let mut x: Vec<R> = vec![R::default(); b.len()];
    let mut ws = Workspace::default();
    let comm = UniverseComm::NoComm(NoComm);
    let stats = solver.solve_f64(a, None, b, &mut x, PcSide::Left, &comm, None, Some(&mut ws))?;
    let rtrue = util::true_residual_norm(a, &x, b);
    Ok((x, stats, rtrue))
}

fn solve_dense_lu(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
    let n = b.len();
    for k in 0..n {
        let mut pivot = k;
        let mut max_val = a[k][k].abs();
        for (i, row) in a.iter().enumerate().skip(k + 1).take(n - (k + 1)) {
            if row[k].abs() > max_val {
                max_val = row[k].abs();
                pivot = i;
            }
        }
        if pivot != k {
            a.swap(k, pivot);
            b.swap(k, pivot);
        }
        let piv = a[k][k];
        for i in (k + 1)..n {
            let factor = a[i][k] / piv;
            a[i][k] = 0.0;
            for j in (k + 1)..n {
                a[i][j] -= factor * a[k][j];
            }
            b[i] -= factor * b[k];
        }
    }

    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = b[i];
        for (j, &xj) in x.iter().enumerate().skip(i + 1).take(n - (i + 1)) {
            sum -= a[i][j] * xj;
        }
        x[i] = sum / a[i][i];
    }
    x
}

fn dense_to_csr(a: &[Vec<f64>]) -> crate::matrix::sparse::CsrMatrix<f64> {
    let n = a.len();
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::new();
    let mut values = Vec::new();
    row_ptr.push(0);
    for row in a.iter().take(n) {
        for (j, &v) in row.iter().enumerate() {
            if v != 0.0 {
                col_idx.push(j);
                values.push(v);
            }
        }
        row_ptr.push(col_idx.len());
    }
    crate::matrix::sparse::CsrMatrix::from_csr(n, n, row_ptr, col_idx, values)
}

#[test]
fn gmres_pipelined_tracks_classical_convergence() -> Result<(), KError> {
    let a = util::nonsym_convdiff_2d(10, 5.0);
    let b: Vec<R> = util::rhs_random(a.nrows(), 7);
    let restart = 20;
    let bnorm: R = util::vec_norm(&b).max(R::from(1e-32));

    let (_x_classic, stats_classic, res_classic) =
        solve_with_variant(&a, &b, GmresVariant::Classical, restart)?;
    let (_x_pipe, stats_pipe, res_pipe) =
        solve_with_variant(&a, &b, GmresVariant::Pipelined, restart)?;

    assert!(res_classic <= R::from(1e-8) * bnorm + R::from(1e-10));
    assert!(res_pipe <= R::from(1e-8) * bnorm + R::from(1e-10));
    assert!(
        (stats_classic.iterations as isize - stats_pipe.iterations as isize).abs() as usize
            <= restart
    );
    Ok(())
}

// #[cfg(not(feature = "complex"))]
// #[test]
// fn gmres_sstep_converges_on_spd() -> Result<(), KError> {
//     let a = util::spd_poisson2d(6);
//     let b: Vec<R> = util::rhs_random(a.nrows(), 4);
//     let restart = 12;
//     let bnorm: R = util::vec_norm(&b).max(R::from(1e-32));

//     let (_x_classic, stats_classic, res_classic) =
//         solve_with_variant(&a, &b, GmresVariant::Classical, restart)?;
//     assert!(res_classic <= R::from(1e-8) * bnorm + R::from(1e-10));
//     assert!(stats_classic.reason.is_converged());

//     for s in [2usize, 4usize] {
//         let (_x_sstep, stats_sstep, res_sstep) = solve_with_variant(
//             &a,
//             &b,
//             GmresVariant::SStep {
//                 s,
//                 reorth: crate::context::ksp_context::ReorthPolicy::IfNeeded,
//                 max_cond: 1e8,
//             },
//             restart,
//         )?;
//         assert!(stats_sstep.reason.is_converged());
//         let target = (R::from(1e-8) * bnorm + R::from(1e-10)).max(res_classic * R::from(25.0));
//         assert!(
//             res_sstep <= target,
//             "s-step({s}) residual too large: got {res_sstep:e}, target {target:e}, classic {res_classic:e}"
//         );
//         assert!(
//             stats_sstep.counters.num_global_reductions
//                 <= stats_classic.counters.num_global_reductions,
//             "expected s-step({s}) reductions <= classical (sstep={}, classic={})",
//             stats_sstep.counters.num_global_reductions,
//             stats_classic.counters.num_global_reductions
//         );
//     }
//     Ok(())
// }

#[cfg(not(feature = "complex"))]
#[test]
fn gmres_sstep_s1_tracks_classical_on_spd() -> Result<(), KError> {
    let a = util::spd_poisson2d(6);
    let b: Vec<R> = util::rhs_random(a.nrows(), 9);
    let restart = 12;
    let bnorm: R = util::vec_norm(&b).max(R::from(1e-32));

    let (_x_classic, stats_classic, res_classic) =
        solve_with_variant(&a, &b, GmresVariant::Classical, restart)?;
    let (_x_sstep, stats_sstep, res_sstep) = solve_with_variant(
        &a,
        &b,
        GmresVariant::SStep {
            s: 1,
            reorth: crate::context::ksp_context::ReorthPolicy::IfNeeded,
            max_cond: 1e8,
        },
        restart,
    )?;

    let target = (R::from(1e-8) * bnorm + R::from(1e-10)).max(res_classic * R::from(10.0));
    assert!(stats_classic.reason.is_converged());
    assert!(stats_sstep.reason.is_converged());
    assert!(res_sstep <= target);
    assert!(
        (stats_classic.iterations as isize - stats_sstep.iterations as isize).abs() as usize
            <= restart
    );
    Ok(())
}

#[cfg(feature = "complex")]
#[test]
fn gmres_sstep_complex_matches_classical_reason() -> Result<(), KError> {
    let a = util::spd_poisson2d(6);
    let b: Vec<R> = util::rhs_random(a.nrows(), 4);
    let restart = 12;

    let (_x_classic, stats_classic, res_classic) =
        solve_with_variant(&a, &b, GmresVariant::Classical, restart)?;
    let (_x_sstep, stats_sstep, res_sstep) = solve_with_variant(
        &a,
        &b,
        GmresVariant::SStep {
            s: 3,
            reorth: crate::context::ksp_context::ReorthPolicy::IfNeeded,
            max_cond: 1e8,
        },
        restart,
    )?;

    assert!(res_sstep.is_finite());
    assert!(res_classic.is_finite());
    assert_ne!(
        stats_classic.reason,
        crate::utils::convergence::ConvergedReason::Continued
    );
    assert_ne!(
        stats_sstep.reason,
        crate::utils::convergence::ConvergedReason::Continued
    );
    Ok(())
}

#[test]
fn gmres_full_restart_matches_dense_reference_on_small_nonsymmetric() -> Result<(), KError> {
    let n = 8usize;
    let mut dense = vec![vec![0.0; n]; n];
    for (i, row) in dense.iter_mut().enumerate().take(n) {
        for (j, val) in row.iter_mut().enumerate().take(n) {
            let base = ((i * 3 + j * 5 + 7) % 17) as f64;
            *val = if i == j {
                10.0 + base
            } else {
                ((i + 2 * j + 1) as f64).sin() * 0.2 + base * 1e-2
            };
        }
    }
    let a = dense_to_csr(&dense);
    let b: Vec<R> = (1..=n).map(|i| i as R).collect();
    let mut x = vec![0.0; n];

    let mut solver = GmresSolver::new(n, 1e-12, n);
    solver.set_variant(GmresVariant::Classical);
    let comm = UniverseComm::NoComm(NoComm);
    let mut ws = Workspace::default();
    let stats = solver.solve_f64(
        &a,
        None,
        &b,
        &mut x,
        PcSide::Left,
        &comm,
        None,
        Some(&mut ws),
    )?;
    assert!(stats.iterations <= n);

    let x_ref = solve_dense_lu(dense, b.clone());

    let diff = x
        .iter()
        .zip(&x_ref)
        .map(|(xi, xri)| (xi - xri).powi(2))
        .sum::<f64>()
        .sqrt();
    let norm_ref = x_ref.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-30);
    assert!(
        diff / norm_ref < 1e-10,
        "relative solution error={}",
        diff / norm_ref
    );
    Ok(())
}

#[test]
fn gmres_monitor_ids_are_strict_and_respect_max_it() -> Result<(), KError> {
    let a = util::nonsym_convdiff_2d(6, 5.0);
    let n = a.nrows();
    let b: Vec<R> = util::rhs_random(n, 11);
    let max_it = 1usize;

    let mut solver = GmresSolver::new(n, 1e-16, max_it);
    solver.set_variant(GmresVariant::Classical);
    let mut x = vec![0.0; n];
    let mut ws = Workspace::default();
    let comm = UniverseComm::NoComm(NoComm);
    let ids = Arc::new(Mutex::new(Vec::<usize>::new()));
    let ids_cb = Arc::clone(&ids);
    let monitor: Box<MonitorCallback<R>> = Box::new(move |it, _r, _| {
        ids_cb.lock().expect("monitor lock").push(it);
        MonitorAction::Continue
    });
    let monitors = vec![monitor];

    let stats = solver.solve_f64(
        &a,
        None,
        &b,
        &mut x,
        PcSide::Left,
        &comm,
        Some(&monitors),
        Some(&mut ws),
    )?;
    assert!(stats.iterations <= max_it);
    let history = ids.lock().expect("ids lock");
    for w in history.windows(2) {
        assert!(
            w[1] > w[0],
            "monitor iteration IDs must be strictly increasing"
        );
    }
    assert!(
        history.last().copied().unwrap_or(0) <= max_it,
        "monitor IDs exceeded max_it"
    );
    Ok(())
}
