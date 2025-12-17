#![cfg(all(feature = "backend-faer", not(feature = "complex")))]
use faer::Mat;
use kryst::algebra::prelude::*;
use kryst::assert_s_close;
use kryst::context::ksp_context::Workspace;
use kryst::error::KError;
use kryst::parallel::{NoComm, UniverseComm};
use kryst::preconditioner::{PcSide, Preconditioner};
use kryst::solver::pcg::CgNormType;
use kryst::solver::{LinearSolver, PcgSolver};
use std::sync::{Arc, Mutex};

struct HalfPc;
impl Preconditioner for HalfPc {
    fn setup(&mut self, _a: &dyn kryst::matrix::op::LinOp<S = f64>) -> Result<(), KError> {
        Ok(())
    }
    fn apply(&self, _side: PcSide, x: &[f64], y: &mut [f64]) -> Result<(), KError> {
        for (yi, xi) in y.iter_mut().zip(x) {
            *yi = 0.5 * xi;
        }
        Ok(())
    }
}

fn run(norm: CgNormType, expected: R) {
    let one = S::one().real();
    let zero = S::zero().real();
    let a = Mat::<R>::from_fn(1, 1, |_, _| one);
    let b = vec![S::from_real(2.0).real()];
    let mut x = vec![zero];
    let mut pc = HalfPc;
    let comm = UniverseComm::NoComm(NoComm);
    let mut solver = PcgSolver::new(S::from_real(1e-20).real(), 0).with_norm(norm);
    let mut wk = Workspace::default();
    solver.setup_workspace(&mut wk);

    let log = Arc::new(Mutex::new(Vec::new()));
    {
        let log_clone = log.clone();
        let monitor: Box<dyn Fn(usize, f64) + Send + Sync> = Box::new(move |i, r| {
            if i == 0 {
                log_clone.lock().unwrap().push(r);
            }
        });
        solver
            .solve_with_comm(
                &a,
                Some(&mut pc),
                &b,
                &mut x,
                PcSide::Left,
                &comm,
                Some(&[monitor]),
                Some(&mut wk),
            )
            .unwrap();
    }
    let res0 = log.lock().unwrap()[0];
    let expected_s = S::from_real(expected);
    let res0_s = S::from_real(res0);
    assert_s_close!("pcg norm initial residual", expected_s, res0_s);
}

#[test]
fn pcg_norm_variants() {
    let two = S::from_real(2.0).real();
    let one = S::one().real();
    run(CgNormType::Preconditioned, two.sqrt());
    run(CgNormType::Unpreconditioned, two);
    run(CgNormType::Natural, one);
}
