#![cfg(all(feature = "backend-faer", not(feature = "complex")))]
use faer::Mat;
use kryst::algebra::prelude::*;
use kryst::assert_s_close;
use kryst::context::ksp_context::Workspace;
use kryst::error::KError;
use kryst::ops::wrap::{as_s_op, as_s_pc_mut};
use kryst::parallel::{NoComm, UniverseComm};
use kryst::preconditioner::{PcSide, Preconditioner};
use kryst::solver::cg::CgNormType;
use kryst::solver::{CgSolver, LinearSolver};
use std::sync::{Arc, Mutex};

struct HalfPc;
impl Preconditioner for HalfPc {
    fn setup(&mut self, _a: &dyn kryst::matrix::op::LinOp<S = f64>) -> Result<(), KError> {
        Ok(())
    }
    fn apply(&self, _side: PcSide, x: &[f64], y: &mut [f64]) -> Result<(), KError> {
        for (yi, xi) in y.iter_mut().zip(x) {
            *yi = 0.5 * *xi;
        }
        Ok(())
    }
}

fn run(norm: CgNormType, expected: R) {
    let a = Mat::<f64>::from_fn(1, 1, |_, _| 1.0);
    let b = vec![2.0f64];
    let x = vec![0.0f64];
    let mut pc = HalfPc;
    let comm = UniverseComm::NoComm(NoComm);
    let mut solver = CgSolver::new(1e-20, 0).with_norm(norm);
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
        let op = as_s_op(&a);
        let mut pc_bridge = as_s_pc_mut(&mut pc);
        let b_s: Vec<S> = b.iter().copied().map(S::from_real).collect();
        let mut x_s: Vec<S> = x.iter().copied().map(S::from_real).collect();
        solver
            .solve_with_comm(
                &op,
                Some(&mut pc_bridge),
                &b_s,
                &mut x_s,
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
    assert_s_close!("cg norm initial residual", expected_s, res0_s);
}

#[test]
fn cg_norm_variants() {
    let two = R::from(2.0);
    run(CgNormType::Preconditioned, two.sqrt());
    run(CgNormType::Unpreconditioned, two);
    run(CgNormType::Natural, R::one());
}
