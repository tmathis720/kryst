#[path = "infra/datasets.rs"]
mod datasets;

use criterion::{Criterion, criterion_group, criterion_main};
use kryst::context::ksp_context::Workspace;
use kryst::matrix::op::LinOp;
use kryst::parallel::{NoComm, UniverseComm};
use kryst::preconditioner::PcSide;
use kryst::solver::fgmres::FgmresSolver;

fn run_case(a: &dyn LinOp<S = f64>, b: &[f64], x: &mut [f64], ws: &mut Workspace) {
    let comm = UniverseComm::NoComm(NoComm);
    let mut solver = FgmresSolver::new(1e-8, 200, 30);
    let _ = solver
        .solve_f64(a, None, b, x, PcSide::Right, &comm, None, Some(ws))
        .expect("FGMRES benchmark solve failed");
}

fn bench_fgmres_rayon_threshold(c: &mut Criterion) {
    for (label, grid) in [("small", 20usize), ("large", 80usize)] {
        let a = datasets::poisson2d_csr(grid);
        let n = a.nrows();
        let b = vec![1.0f64; n];
        let mut x = vec![0.0f64; n];
        let mut ws = Workspace::default();

        c.bench_function(&format!("fgmres_{label}_g{grid}"), |ben| {
            ben.iter(|| {
                x.fill(0.0);
                run_case(&a, &b, &mut x, &mut ws);
            });
        });
    }
}

criterion_group!(benches, bench_fgmres_rayon_threshold);
criterion_main!(benches);
