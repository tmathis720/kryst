use criterion::{Criterion, criterion_group, criterion_main};
use kryst::config::options::{CgVariant, KspOptions};
use kryst::context::ksp_context::{KspContext, SolverType};
use kryst::matrix::op::{CsrOp, LinOp};
use kryst::matrix::sparse::CsrMatrix;
use kryst::parallel::NoComm;
use kryst::parallel::UniverseComm;
use kryst::preconditioner::PcSide;
use std::hint::black_box;
use std::sync::Arc;

#[cfg(feature = "rayon")]
use num_cpus;

fn make_local_operator(n: usize, half_bandwidth: usize) -> Arc<dyn LinOp<S = f64>> {
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::new();
    let mut values = Vec::new();
    row_ptr.push(0);

    for i in 0..n {
        let start = i.saturating_sub(half_bandwidth);
        let end = (i + half_bandwidth).min(n - 1);
        let neighbors = (end - start) as f64;
        for j in start..=end {
            col_idx.push(j);
            if i == j {
                values.push(neighbors + 2.0);
            } else {
                values.push(-1.0);
            }
        }
        row_ptr.push(col_idx.len());
    }

    let csr = Arc::new(CsrMatrix::from_csr(n, n, row_ptr, col_idx, values));
    let op = CsrOp::new(csr).with_comm(UniverseComm::NoComm(NoComm));
    Arc::new(op)
}

fn bench_local(c: &mut Criterion) {
    let cases = [(10_000usize, 1usize), (50_000usize, 2usize)];
    for &(n, half_band) in &cases {
        let op = make_local_operator(n, half_band);
        let label = format!("cg_local_n{n}_bw{}", 2 * half_band + 1);
        c.bench_function(&label, |b| {
            b.iter(|| {
                let mut ksp = KspContext::new();
                ksp.set_type(SolverType::Pcg).unwrap();
                ksp.try_set_pc_side(PcSide::Left).unwrap();

                let mut opts = KspOptions::default();
                opts.cg_variant = Some(CgVariant::Classic);
                #[cfg(feature = "rayon")]
                {
                    opts.threads = Some(num_cpus::get_physical());
                }
                ksp.set_from_options(&opts).unwrap();
                ksp.set_tolerances(1e-8, 0.0, 1e6, 200);
                ksp.set_operators(op.clone(), None);

                let rhs = vec![1.0f64; n];
                let mut x = vec![0.0f64; n];

                let stats = ksp.solve(&rhs, &mut x).unwrap();
                let _ = black_box(stats);
            });
        });
    }
}

criterion_group!(benches, bench_local);
criterion_main!(benches);
