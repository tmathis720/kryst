use criterion::{Criterion, criterion_group, criterion_main};
use kryst::config::options::{KspOptions, PcOptions};
use kryst::context::ksp_context::{KspContext, SolverType};
use kryst::matrix::op::CsrOp;
use kryst::matrix::sparse::CsrMatrix;
use std::sync::Arc;

fn laplace_1d(n: usize) -> CsrMatrix<f64> {
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::with_capacity(3 * n);
    let mut vals = Vec::with_capacity(3 * n);
    row_ptr.push(0);
    for i in 0..n {
        if i > 0 {
            col_idx.push(i - 1);
            vals.push(-1.0);
        }
        col_idx.push(i);
        vals.push(2.5);
        if i + 1 < n {
            col_idx.push(i + 1);
            vals.push(-1.0);
        }
        row_ptr.push(col_idx.len());
    }
    CsrMatrix::from_csr(n, n, row_ptr, col_idx, vals)
}

fn bench_mode(c: &mut Criterion, split: &str, schur: Option<&str>) {
    let n = 200usize;
    let a = Arc::new(CsrOp::new(Arc::new(laplace_1d(n))));
    let mut ksp = KspContext::new();
    ksp.set_type(SolverType::Gmres).unwrap();
    let ksp_opts = KspOptions {
        maxits: Some(40),
        rtol: Some(1e-8),
        ..Default::default()
    };
    let mut pc_opts = PcOptions {
        pc_type: Some("fieldsplit".into()),
        pc_fieldsplit_block_sizes: Some(vec![n / 2, n / 2]),
        pc_fieldsplit_child_pc_type: Some("jacobi".into()),
        pc_fieldsplit_type: Some(split.into()),
        ..Default::default()
    };
    if let Some(route) = schur {
        pc_opts.pc_fieldsplit_schur_fact_type = Some("full".into());
        pc_opts.pc_fieldsplit_schur_precondition = Some("full".into());
        pc_opts.pc_fieldsplit_schur_approx = Some(route.into());
        pc_opts.pc_fieldsplit_comm_schedule = Some("local_first".into());
    }
    ksp.set_from_all_options(&ksp_opts, &pc_opts).unwrap();
    ksp.set_operators(a, None);
    let b = vec![1.0; n];
    let mut x = vec![0.0; n];

    c.bench_function(
        &format!("fieldsplit_{split}_{}", schur.unwrap_or("base")),
        |ben| {
            ben.iter(|| {
                x.fill(0.0);
                let _ = ksp.solve(&b, &mut x).unwrap();
            });
        },
    );
}

fn bench_fieldsplit_modes(c: &mut Criterion) {
    bench_mode(c, "additive", None);
    bench_mode(c, "multiplicative", None);
    bench_mode(c, "schur", Some("dist_full"));
}

criterion_group!(benches, bench_fieldsplit_modes);
criterion_main!(benches);
