#[path = "infra/datasets.rs"]
mod datasets;

use criterion::{Criterion, criterion_group, criterion_main};
use faer::Mat;
use kryst::algebra::prelude::*;
use kryst::matrix::sparse::CsrMatrix;
use kryst::preconditioner::mg::MgPc;
use kryst::preconditioner::sor::{MatSorType, SorPc};
use kryst::preconditioner::{
    AmgCoarseSpace, DeflationOptions, DeflationPC, Jacobi, PcSide, Preconditioner, ZSource,
};

#[cfg(feature = "complex")]
#[inline]
fn s_from(re: f64, im: f64) -> S {
    S::from_parts(re, im)
}

#[cfg(not(feature = "complex"))]
#[inline]
fn s_from(re: f64, _im: f64) -> S {
    S::from_real(re)
}

fn residual_norm(a: &CsrMatrix<S>, b: &[S], x: &[S]) -> R {
    let mut ax = vec![S::zero(); b.len()];
    a.try_spmv(x, &mut ax).expect("spmv");
    let mut acc = R::zero();
    for i in 0..b.len() {
        let ri = b[i] - ax[i];
        acc += ri.abs2();
    }
    acc.sqrt()
}

fn to_complex_like(a: &CsrMatrix<f64>) -> CsrMatrix<S> {
    let values = a
        .values()
        .iter()
        .copied()
        .map(|v| s_from(v, 0.05 * v.signum()))
        .collect();
    CsrMatrix::from_csr(
        a.nrows(),
        a.ncols(),
        a.row_ptr().to_vec(),
        a.col_idx().to_vec(),
        values,
    )
}

fn bench_residual_reduction_complex(c: &mut Criterion) {
    let a_real = datasets::poisson2d_csr(24);
    let a = to_complex_like(&a_real);
    let n = a.nrows();
    let b = vec![s_from(1.0, -0.2); n];

    let mut sor = SorPc::new(1.0, 2, MatSorType::SYMMETRIC_SWEEP, 0.0);
    sor.setup(&a).expect("sor setup");

    let mut mg = MgPc::new(
        3,
        Some("v".into()),
        Some("sor".into()),
        Some(1),
        Some("linear".into()),
        Some("linear".into()),
        Some("full_weighting".into()),
        Some("jacobi".into()),
        None,
        None,
        None,
    );
    mg.setup(&a).expect("mg setup");

    let mut z = Mat::<f64>::zeros(n, 4);
    for i in 0..n {
        z[(i, i % 4)] = 1.0;
    }
    let coarse = AmgCoarseSpace {
        z,
        local_range: None,
    };
    let defl_opts = DeflationOptions {
        z_source: ZSource::External,
        cond_cap: None,
        augment_initial_guess: false,
    };
    let mut defl = DeflationPC::new(Jacobi::new(), &a_real, coarse, &defl_opts).expect("defl");
    defl.setup(&a).expect("defl setup");

    let mut group = c.benchmark_group("complex_residual_reduction_per_iter");
    for (name, pc) in [
        ("sor", &sor as &dyn Preconditioner),
        ("mg", &mg as &dyn Preconditioner),
        ("deflation", &defl as &dyn Preconditioner),
    ] {
        let mut x = vec![S::zero(); n];
        let before = residual_norm(&a, &b, &x);
        pc.apply(PcSide::Left, &b, &mut x).expect("apply");
        let after = residual_norm(&a, &b, &x);
        let ratio = if before > 0.0 { after / before } else { 1.0 };
        eprintln!("quality metric [{name}] complex residual ratio after 1 apply: {ratio:.6}");

        group.bench_function(name, |ben| {
            ben.iter(|| {
                x.fill(S::zero());
                pc.apply(PcSide::Left, &b, &mut x).expect("apply");
            });
        });
    }
    group.finish();
}

fn bench_setup_apply_cost_deltas(c: &mut Criterion) {
    let a_real = datasets::poisson2d_csr(32);
    let a_complex = to_complex_like(&a_real);
    let n = a_real.nrows();
    let rhs_real = vec![1.0; n];
    let rhs_complex = vec![s_from(1.0, 0.25); n];

    let mut group = c.benchmark_group("setup_apply_cost_delta_vs_real");

    group.bench_function("sor_real_baseline", |ben| {
        ben.iter(|| {
            let mut pc = SorPc::new(1.0, 2, MatSorType::APPLY_LOWER, 0.0);
            pc.setup(&a_real).expect("setup real");
            let mut y = vec![0.0; n];
            pc.apply(PcSide::Left, &rhs_real, &mut y)
                .expect("apply real");
        });
    });

    group.bench_function("sor_complex_native", |ben| {
        ben.iter(|| {
            let mut pc = SorPc::new(1.0, 2, MatSorType::APPLY_LOWER, 0.0);
            pc.setup(&a_complex).expect("setup complex");
            let mut y = vec![S::zero(); n];
            pc.apply(PcSide::Left, &rhs_complex, &mut y)
                .expect("apply complex");
        });
    });

    group.finish();
}

fn bench_mpi_rayon_scaling_sanity(c: &mut Criterion) {
    #[cfg(feature = "rayon")]
    {
        use kryst::algebra::parallel::set_rayon_threads;
        let a = to_complex_like(&datasets::poisson2d_csr(28));
        let n = a.nrows();
        let b = vec![s_from(1.0, 0.0); n];

        let mut group = c.benchmark_group("mpi_rayon_scaling_sanity_single_rank");
        for t in [1usize, 2, 4] {
            set_rayon_threads(t);
            let mut mg = MgPc::new(
                3,
                Some("v".into()),
                Some("jacobi".into()),
                Some(1),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            );
            mg.setup(&a).expect("mg setup");
            let mut y = vec![S::zero(); n];
            group.bench_function(format!("threads_{t}"), |ben| {
                ben.iter(|| {
                    y.fill(S::zero());
                    mg.apply(PcSide::Left, &b, &mut y).expect("mg apply");
                });
            });
        }
        group.finish();
    }

    #[cfg(not(feature = "rayon"))]
    {
        let _ = c;
        eprintln!("mpi/rayon scaling sanity bench skipped: `rayon` feature is disabled");
    }
}

criterion_group!(
    benches,
    bench_residual_reduction_complex,
    bench_setup_apply_cost_deltas,
    bench_mpi_rayon_scaling_sanity
);
criterion_main!(benches);
