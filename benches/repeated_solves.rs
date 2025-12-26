#[path = "infra/datasets.rs"]
mod datasets;

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use kryst::context::ksp_context::{KspContext, SolverType};
use kryst::matrix::op::CsrOp;
use kryst::parallel::{NoComm, UniverseComm};
use kryst::preconditioner::PcSide;
use std::sync::Arc;

fn bench_repeated_solves(c: &mut Criterion) {
    let grid = 40usize;
    let a = datasets::poisson2d_csr(grid);
    let n = a.nrows();
    let b = vec![1.0f64; n];
    let comm = UniverseComm::NoComm(NoComm);
    let op = Arc::new(CsrOp::new(Arc::new(a)).with_comm(comm.clone()));

    c.bench_function("gmres_reuse_context", |ben| {
        let mut ksp = KspContext::new();
        ksp.set_type(SolverType::Gmres).unwrap();
        ksp.try_set_pc_side(PcSide::Left).unwrap();
        ksp.set_restart(30);
        ksp.set_tolerances(1e-8, 1e-12, 1e6, 200);
        ksp.set_operators(op.clone(), None);
        ksp.setup().unwrap();
        let mut x = vec![0.0f64; n];
        ben.iter(|| {
            for _ in 0..5 {
                x.fill(0.0);
                let _ = ksp.solve(&b, &mut x).unwrap();
            }
        });
    });

    c.bench_function("gmres_new_context", |ben| {
        ben.iter_batched(
            || {
                let mut ksp = KspContext::new();
                ksp.set_type(SolverType::Gmres).unwrap();
                ksp.try_set_pc_side(PcSide::Left).unwrap();
                ksp.set_restart(30);
                ksp.set_tolerances(1e-8, 1e-12, 1e6, 200);
                ksp.set_operators(op.clone(), None);
                ksp.setup().unwrap();
                (ksp, vec![0.0f64; n])
            },
            |(mut ksp, mut x)| {
                let _ = ksp.solve(&b, &mut x).unwrap();
            },
            BatchSize::SmallInput,
        );
    });
}

criterion_group!(benches, bench_repeated_solves);
criterion_main!(benches);
