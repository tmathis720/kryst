#[cfg(feature = "complex")]
use kryst::prelude::*;

#[cfg(not(feature = "complex"))]
fn main() {
    eprintln!("complex_gmres requires --features complex");
}

#[cfg(feature = "complex")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use kryst::algebra::bridge::BridgeScratch;
    use kryst::ops::klinop::KLinOp;
    use kryst::ops::kpc::KPreconditioner;
    use kryst::parallel::{NoComm, UniverseComm};
    use kryst::preconditioner::PcSide;
    use kryst::solver::GmresSolver;

    struct ComplexDiag {
        diag: Vec<S>,
    }

    impl KLinOp for ComplexDiag {
        type Scalar = S;

        fn dims(&self) -> (usize, usize) {
            (self.diag.len(), self.diag.len())
        }

        fn matvec_s(&self, x: &[S], y: &mut [S], _scratch: &mut BridgeScratch) {
            for i in 0..self.diag.len() {
                y[i] = self.diag[i] * x[i];
            }
        }
    }

    let n = 4;
    let diag = vec![S::from_parts(2.0, 0.5); n];
    let op = ComplexDiag { diag };
    let b = vec![S::from_real(1.0); n];
    let mut x = vec![S::zero(); n];

    let comm = UniverseComm::NoComm(NoComm);
    let pc: Option<&dyn KPreconditioner<Scalar = S>> = None;

    let mut solver = GmresSolver::new(8, 1e-10, 50);
    let stats = solver.solve(&op, pc, &b, &mut x, PcSide::Left, &comm, None, None)?;

    println!(
        "iters={} reason={:?} final_residual={:.3e}",
        stats.iterations, stats.reason, stats.final_residual
    );

    Ok(())
}
