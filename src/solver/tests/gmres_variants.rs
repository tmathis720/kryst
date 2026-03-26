use crate::algebra::prelude::*;
use crate::context::ksp_context::Workspace;
use crate::error::KError;
use crate::parallel::{NoComm, UniverseComm};
use crate::preconditioner::PcSide;
use crate::solver::gmres::{GmresSolver, GmresVariant};

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

#[cfg(not(feature = "complex"))]
#[test]
fn gmres_sstep_converges_on_spd() -> Result<(), KError> {
    let a = util::spd_poisson2d(6);
    let b: Vec<R> = util::rhs_random(a.nrows(), 4);
    let restart = 12;
    let bnorm: R = util::vec_norm(&b).max(R::from(1e-32));

    let (_x_classic, stats_classic, res_classic) =
        solve_with_variant(&a, &b, GmresVariant::Classical, restart)?;
    assert!(res_classic <= R::from(1e-8) * bnorm + R::from(1e-10));
    assert!(stats_classic.reason.is_converged());

    for s in [2usize, 4usize] {
        let (_x_sstep, stats_sstep, res_sstep) = solve_with_variant(
            &a,
            &b,
            GmresVariant::SStep {
                s,
                reorth: crate::context::ksp_context::ReorthPolicy::IfNeeded,
                max_cond: 1e8,
            },
            restart,
        )?;
        assert!(stats_sstep.reason.is_converged());
        let target = (R::from(1e-8) * bnorm + R::from(1e-10)).max(res_classic * R::from(25.0));
        assert!(
            res_sstep <= target,
            "s-step({s}) residual too large: got {res_sstep:e}, target {target:e}, classic {res_classic:e}"
        );
        assert!(
            stats_sstep.counters.num_global_reductions
                <= stats_classic.counters.num_global_reductions,
            "expected s-step({s}) reductions <= classical (sstep={}, classic={})",
            stats_sstep.counters.num_global_reductions,
            stats_classic.counters.num_global_reductions
        );
    }
    Ok(())
}

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
    assert!(
        (stats_classic.iterations as isize - stats_sstep.iterations as isize).abs() as usize
            <= restart
    );
    Ok(())
}
