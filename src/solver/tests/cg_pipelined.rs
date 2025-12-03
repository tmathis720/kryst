use crate::algebra::prelude::*;
use crate::context::ksp_context::Workspace;
use crate::error::KError;
use crate::matrix::op::LinOp;
use crate::parallel::{NoComm, UniverseComm};
use crate::preconditioner::PcSide;
use crate::preconditioner::Preconditioner;
use crate::preconditioner::jacobi::Jacobi;
use crate::solver::LinearSolver;
use crate::solver::pcg::{PcgSolver, PcgVariant};

use super::util;

/// Solve with a pipelined PCG variant on any operator that implements `LinOp<S = f64>`.
fn solve_with_variant<A>(a: &A, b: &[R], variant: PcgVariant) -> Result<(Vec<R>, usize, R), KError>
where
    A: LinOp<S = f64> + 'static,
{
    let mut solver = PcgSolver::new(1e-10, 5_000);
    solver.set_variant(variant);

    // In non-complex builds R = f64; in complex builds R is the real scalar type.
    let mut x: Vec<R> = vec![R::default(); b.len()];
    let mut ws = Workspace::default();
    let mut pc = Jacobi::new();

    let op: &dyn LinOp<S = f64> = a;
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
        let bnorm = util::vec_norm(&b).max(R::from(1e-32));

        let (x_classic, it_classic, res_classic) = solve_with_variant(&a, &b, PcgVariant::Classic)?;
        let (_x_pipe, it_pipe, res_pipe) = solve_with_variant(
            &a,
            &b,
            PcgVariant::Pipelined {
                replace_every: crate::solver::PCG_PIPELINED_DEFAULT_REPLACE_EVERY,
            },
        )?;

        assert!(
            res_classic <= R::from(1e-10) * bnorm + R::from(1e-12),
            "classic residual {:.3e} exceeds tolerance with bnorm {:.3e}",
            res_classic,
            bnorm
        );
        // The pipelined variant trades a small loss of accuracy for overlap. In
        // practice the residual can stagnate at ~1e-8 even though the classic
        // solver reaches the tighter 1e-10 tolerance, so we only require
        // agreement to 1e-8 in these checks.
        assert!(
            res_pipe <= R::from(1e-8) * bnorm + R::from(1e-10),
            "pipelined residual {:.3e} exceeds relaxed tolerance with bnorm {:.3e}; classic {:.3e}",
            res_pipe,
            bnorm,
            res_classic
        );
        assert!(
            (it_classic as isize - it_pipe as isize).abs() <= 12,
            "iteration counts diverged: classic={}, pipelined={}",
            it_classic,
            it_pipe
        );

        // Ensure the classic solution also has a small true residual.
        let op: &dyn LinOp<S = f64> = &a;
        let r_classic = util::true_residual_norm(op, &x_classic, &b);
        assert!(r_classic <= R::from(1e-10) * bnorm + R::from(1e-12));
    }
    Ok(())
}
