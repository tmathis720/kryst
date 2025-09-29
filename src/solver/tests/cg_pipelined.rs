use crate::context::ksp_context::Workspace;
use crate::error::KError;
use crate::parallel::{NoComm, UniverseComm};
use crate::preconditioner::PcSide;
use crate::preconditioner::jacobi::Jacobi;
use crate::solver::LinearSolver;
use crate::solver::pcg::{PcgSolver, PcgVariant};

use super::util;

fn solve_with_variant(
    a: &crate::matrix::sparse::CsrMatrix<f64>,
    b: &[f64],
    variant: PcgVariant,
) -> Result<(Vec<f64>, usize, f64), KError> {
    let mut solver = PcgSolver::new(1e-10, 5_000);
    solver.set_variant(variant);
    let mut x = vec![0.0f64; b.len()];
    let mut ws = Workspace::default();
    let mut pc = Jacobi::new();
    let op: &dyn crate::matrix::op::LinOp<S = f64> = a;
    pc.setup(op)?;
    let comm = UniverseComm::NoComm(NoComm);
    let stats = solver.solve(
        op,
        Some(&mut pc),
        b,
        &mut x,
        PcSide::Left,
        &comm,
        None,
        Some(&mut ws),
    )?;
    let rtrue = util::true_residual_norm(op, &x, b);
    Ok((x, stats.iterations, rtrue))
}

#[test]
fn pcg_pipelined_matches_classic_on_spd_gallery() -> Result<(), KError> {
    let sizes = [12usize, 16usize];
    for &n in &sizes {
        let a = util::spd_poisson2d(n);
        let b = util::rhs_random(a.nrows(), 42);
        let bnorm = util::vec_norm(&b).max(1e-32);

        let (x_classic, it_classic, res_classic) = solve_with_variant(&a, &b, PcgVariant::Classic)?;
        let (_x_pipe, it_pipe, res_pipe) = solve_with_variant(
            &a,
            &b,
            PcgVariant::Pipelined {
                replace_every: crate::solver::PCG_PIPELINED_DEFAULT_REPLACE_EVERY,
            },
        )?;

        assert!(res_classic <= 1e-10 * bnorm + 1e-12);
        assert!(res_pipe <= 1e-10 * bnorm + 1e-12);
        assert!((it_classic as isize - it_pipe as isize).abs() <= 1);

        // Ensure the solutions are close in norm.
        let op: &dyn crate::matrix::op::LinOp<S = f64> = &a;
        let r_classic = util::true_residual_norm(op, &x_classic, &b);
        assert!(r_classic <= 1e-10 * bnorm + 1e-12);
    }
    Ok(())
}
