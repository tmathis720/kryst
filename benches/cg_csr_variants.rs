use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use kryst::algebra::prelude::*;
use kryst::config::options::{CgVariant, KspOptions};
use kryst::context::ksp_context::{KspContext, SolverType};
use kryst::context::pc_context::PcType;
use kryst::matrix::csr::CsrMatrix as ScalarCsrMatrix;
use kryst::matrix::op::{GenericCsrOp, LinOp};
use kryst::matrix::spmv::plan::SpmvTuning;
use kryst::matrix::utils::poisson_3d;
use std::hint::black_box;
use std::sync::Arc;

fn csr_operator(grid: usize) -> Arc<dyn LinOp<S = S>> {
    let real = poisson_3d(grid, grid, 1);
    let mut values = Vec::with_capacity(real.nnz());
    for row in 0..real.nrows() {
        for index in real.row_ptr()[row]..real.row_ptr()[row + 1] {
            let col = real.col_idx()[index];
            let value = real.values()[index];
            if row == col {
                values.push(S::from_real(value + 1.0));
            } else {
                let imag = if col > row { 0.02 } else { -0.02 };
                values.push(S::from_parts(value, imag));
            }
        }
    }

    let matrix = ScalarCsrMatrix::new(
        real.nrows(),
        real.ncols(),
        real.row_ptr().to_vec(),
        real.col_idx().to_vec(),
        values,
    );
    Arc::new(GenericCsrOp::new(Arc::new(matrix), &SpmvTuning::default()))
}

fn rhs_for(op: &dyn LinOp<S = S>) -> Vec<S> {
    let x_true: Vec<S> = (0..op.dims().1)
        .map(|i| S::from_parts(1.0 + (i % 7) as f64 * 0.1, (i % 5) as f64 * 0.03))
        .collect();
    let mut rhs = vec![S::zero(); x_true.len()];
    op.matvec(&x_true, &mut rhs);
    rhs
}

fn bench_csr_variants(c: &mut Criterion) {
    let mut group = c.benchmark_group("cg_csr_classic_vs_pipelined");
    for grid in [32usize, 64] {
        let op = csr_operator(grid);
        let rhs = rhs_for(op.as_ref());
        let n = rhs.len();

        for solver_type in [SolverType::Cg, SolverType::Pcg] {
            for variant in [CgVariant::Classic, CgVariant::Pipelined] {
                let id = BenchmarkId::new(
                    format!("{solver_type:?}_{variant:?}"),
                    format!("csr_{grid}x{grid}"),
                );
                group.bench_function(id, |bencher| {
                    bencher.iter(|| {
                        let mut ksp = KspContext::new();
                        ksp.set_type(solver_type).expect("set CG-family solver");
                        ksp.set_from_options(&KspOptions {
                            cg_variant: Some(variant),
                            cg_replace_every: Some(0),
                            ..KspOptions::default()
                        })
                        .expect("set CG options");
                        ksp.set_pc_type(PcType::Jacobi, None)
                            .expect("set Jacobi preconditioner");
                        ksp.set_tolerances(1e-8, 1e-12, 1e8, 4 * n);
                        ksp.set_operators(op.clone(), None);

                        let mut x = vec![S::zero(); n];
                        let stats = ksp.solve(&rhs, &mut x).expect("CSR CG solve");
                        assert!(stats.reason.is_converged(), "stats={stats:?}");
                        let _ = black_box(stats);
                    });
                });
            }
        }
    }
    group.finish();
}

criterion_group!(benches, bench_csr_variants);
criterion_main!(benches);
