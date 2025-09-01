use criterion::{black_box, criterion_group, criterion_main, Criterion};
use kryst::matrix::{spmv, sparse::CsrMatrix};

fn csr_poisson_1d(n: usize) -> CsrMatrix<f64> {
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::new();
    let mut vals = Vec::new();
    row_ptr.push(0);
    for i in 0..n {
        if i > 0 {
            col_idx.push(i - 1);
            vals.push(-1.0);
        }
        col_idx.push(i);
        vals.push(2.0);
        if i + 1 < n {
            col_idx.push(i + 1);
            vals.push(-1.0);
        }
        row_ptr.push(col_idx.len());
    }
    CsrMatrix::from_csr(n, n, row_ptr, col_idx, vals)
}

fn bench_spmv(c: &mut Criterion) {
    let n = 1_000;
    let a = csr_poisson_1d(n);
    let x = vec![1.0; n];
    let mut y = vec![0.0; n];
    c.bench_function("spmv_csr_parallel", |b| {
        b.iter(|| {
            spmv::spmv_csr_parallel(&a, black_box(&x), &mut y).unwrap();
        });
    });
}

criterion_group!(benches, bench_spmv);
criterion_main!(benches);
