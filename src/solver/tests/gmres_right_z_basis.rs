#[cfg(test)]
mod tests_gmres_z_basis {
    use crate::algebra::prelude::*;
    use crate::context::ksp_context::{KspContext, SolverType};
    use crate::error::KError;
    use crate::matrix::op::LinOp;
    use crate::preconditioner::{PcSide, Preconditioner};
    use faer::Mat;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct CountingIdentityPc {
        hits: Arc<AtomicUsize>,
    }
    impl CountingIdentityPc {
        fn new(hits: Arc<AtomicUsize>) -> Self {
            Self { hits }
        }
    }
    impl Preconditioner for CountingIdentityPc {
        fn setup(&mut self, _a: &dyn LinOp<S = f64>) -> Result<(), KError> {
            Ok(())
        }
        fn apply(&self, _side: PcSide, x: &[f64], y: &mut [f64]) -> Result<(), KError> {
            self.hits.fetch_add(1, Ordering::Relaxed);
            y.copy_from_slice(x);
            Ok(())
        }
    }

    #[test]
    fn gmres_right_uses_z_basis_and_calls_pc_per_step() -> Result<(), KError> {
        // A: simple nonsymmetric to avoid early convergence in one step
        let a = Mat::from_fn(4, 4, |i, j| {
            if i == j {
                R::from(3.0)
            } else if j + 1 == i {
                R::from(-1.0)
            } else {
                R::from(0.2)
            }
        });
        let b = [R::from(1.0), R::default(), R::default(), R::default()];
        let amat: Arc<dyn LinOp<S = f64>> = Arc::new(a);

        // Counting identity PC
        let hits = Arc::new(AtomicUsize::new(0));
        let pc = Box::new(CountingIdentityPc::new(hits.clone())) as Box<dyn Preconditioner>;

        // GMRES with a small restart to make a clear "cycle"
        let mut ksp = KspContext::new();
        ksp.set_type(SolverType::Gmres)?;
        ksp.set_operators(amat.clone(), None);
        ksp.pc_side = PcSide::Right;
        ksp.restart = 3; // keep small for predictable cycle
        ksp.rtol = 1e-12; // try to make it finish inside a couple cycles
        ksp.set_pc_box_for_tests(pc);

        let mut x = [R::default(); 4];
        let stats = ksp.solve(&b, &mut x)?;

        let w = ksp.debug_workspace().expect("workspace present");
        assert!(
            w.has_z(),
            "Right-preconditioned GMRES must allocate Z basis"
        );

        // Counting PC should be called once per step in the last cycle at least.
        let calls = hits.load(Ordering::Relaxed);
        let last_cycle_steps = {
            let iters = stats.iterations % ksp.restart;
            if iters == 0 { ksp.restart } else { iters }
        };
        assert!(
            calls >= last_cycle_steps,
            "PC apply calls ({calls}) must be >= last-cycle steps ({})",
            last_cycle_steps
        );

        Ok(())
    }
}
