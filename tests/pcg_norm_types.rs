use faer::Mat;
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

fn run(norm: CgNormType, expected: f64) {
    let a = Mat::<f64>::from_fn(1, 1, |_, _| 1.0);
    let b = vec![2.0];
    let mut x = vec![0.0];
    let mut pc = HalfPc;
    let comm = UniverseComm::NoComm(NoComm);
    let mut solver = PcgSolver::new(1e-20, 0).with_norm(norm);
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
    assert!((res0 - expected).abs() < 1e-12);
}

#[test]
fn pcg_norm_variants() {
    run(CgNormType::Preconditioned, (2.0_f64).sqrt());
    run(CgNormType::Unpreconditioned, 2.0);
    run(CgNormType::Natural, 1.0);
}

