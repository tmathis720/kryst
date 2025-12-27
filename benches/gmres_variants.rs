#[path = "infra/datasets.rs"]
mod datasets;

use criterion::{Criterion, criterion_group, criterion_main};
use kryst::context::ksp_context::Workspace;
use kryst::parallel::{NoComm, UniverseComm};
use kryst::preconditioner::PcSide;
use kryst::solver::gmres::{GmresSolver, GmresVariant};

fn bench_gmres_variants(c: &mut Criterion) {
    let grid = 40usize;
    let a = datasets::poisson2d_csr(grid);
    let n = a.nrows();
    let b = vec![1.0f64; n];
    let comm = UniverseComm::NoComm(NoComm);

    c.bench_function("gmres_classic", |ben| {
        let mut solver = GmresSolver::new(30, 1e-8, 200);
        solver.set_variant(GmresVariant::Classical);
        let mut ws = Workspace::default();
        let mut x = vec![0.0f64; n];
        ben.iter(|| {
            x.fill(0.0);
            let _ = solver
                .solve_f64(
                    &a,
                    None,
                    &b,
                    &mut x,
                    PcSide::Left,
                    &comm,
                    None,
                    Some(&mut ws),
                )
                .unwrap();
        });
    });

    #[cfg(not(feature = "complex"))]
    c.bench_function("gmres_pipelined", |ben| {
        let mut solver = GmresSolver::new(30, 1e-8, 200);
        solver.set_variant(GmresVariant::Pipelined);
        let mut ws = Workspace::default();
        let mut x = vec![0.0f64; n];
        ben.iter(|| {
            x.fill(0.0);
            let _ = solver
                .solve_f64(
                    &a,
                    None,
                    &b,
                    &mut x,
                    PcSide::Left,
                    &comm,
                    None,
                    Some(&mut ws),
                )
                .unwrap();
        });
    });
}

criterion_group!(benches, bench_gmres_variants);
criterion_main!(benches);
