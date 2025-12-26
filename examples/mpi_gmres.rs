#[cfg(feature = "mpi")]
use kryst::matrix::MatShell;
#[cfg(feature = "mpi")]
use kryst::prelude::*;
#[cfg(feature = "mpi")]
use std::sync::Arc;

#[cfg(not(feature = "mpi"))]
fn main() {
    eprintln!("mpi_gmres requires --features mpi");
}

#[cfg(feature = "mpi")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use kryst::parallel::MpiComm;

    let comm = UniverseComm::Mpi(Arc::new(MpiComm::new()));
    let rank = comm.rank();
    let size = comm.size();

    let n = 8;
    let op = MatShell::new(n, n, move |x, y| {
        for i in 0..n {
            let mut sum = 2.0 * x[i];
            if i > 0 {
                sum -= x[i - 1];
            }
            if i + 1 < n {
                sum -= x[i + 1];
            }
            y[i] = sum;
        }
    });

    let op = Arc::new(op) as Arc<dyn LinOp<S = R>>;
    let b = vec![S::from_real(1.0); n];
    let mut x = vec![S::zero(); n];

    let mut ksp = KspContext::new();
    ksp.set_type(SolverType::Gmres)?;
    ksp.set_pc_type(PcType::None, None)?;
    ksp.try_set_operators_with_comm(op, None, comm.clone())?;
    ksp.setup()?;

    let stats = ksp.solve(&b, &mut x)?;
    println!(
        "rank {}/{}: iters={} reason={:?} final_residual={:.3e}",
        rank,
        size,
        stats.iterations,
        stats.reason,
        stats.final_residual
    );

    Ok(())
}
