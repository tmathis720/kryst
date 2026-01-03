// run with `cargo run --example basic_gmres`
#[cfg(feature = "complex")]
fn main() {
    eprintln!("basic_gmres.rs is unavailable when built with --features complex");
}

use kryst::matrix::MatShell;
use kryst::prelude::*;
use std::sync::Arc;

#[cfg(not(feature = "complex"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n = 8;
    let op = MatShell::<f64>::new(n, n, move |x, y| {
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
    ksp.set_operators(op, None);
    ksp.setup()?;

    let stats = ksp.solve(&b, &mut x)?;
    println!(
        "iters={} reason={:?} final_residual={:.3e}",
        stats.iterations, stats.reason, stats.final_residual
    );

    Ok(())
}
