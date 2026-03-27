use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use kryst::matrix::{sparse::CsrMatrix, spmv::spmv_csr_parallel};
use rand::{RngExt, SeedableRng};

fn build_random_csr(m: usize, n: usize, nnz_per_row: usize) -> CsrMatrix<f64> {
    let mut rp = Vec::with_capacity(m + 1);
    rp.push(0);
    let mut cj = Vec::with_capacity(m * nnz_per_row);
    let mut vv = Vec::with_capacity(m * nnz_per_row);
    let mut rng = rand::rngs::StdRng::seed_from_u64(123);
    for _ in 0..m {
        let mut cols: Vec<usize> = (0..n)
            .map(|_| rng.random_range(0..n))
            .take(nnz_per_row)
            .collect();
        cols.sort_unstable();
        cols.dedup();
        let k0 = cj.len();
        cj.extend(cols.iter().copied());
        let added = cj.len() - k0;
        vv.extend((0..added).map(|_| rng.random::<f64>()));
        rp.push(cj.len());
    }
    CsrMatrix::from_csr(m, n, rp, cj, vv)
}

fn bench_spmv(c: &mut Criterion) {
    let (m, n, nnz_r) = (200_000, 200_000, 20);
    let a = build_random_csr(m, n, nnz_r);
    let x = vec![1.0; n];
    let mut y = vec![0.0; m];

    let mut group = c.benchmark_group("spmv_csr");
    group.sample_size(10);
    group.throughput(Throughput::Elements(a.nnz() as u64));

    group.bench_function(BenchmarkId::new("naive_serial", a.nnz()), |b| {
        b.iter(|| {
            a.spmv_scaled(1.0, &x, 0.0, &mut y).unwrap();
        });
    });

    group.bench_function(BenchmarkId::new("parallel", a.nnz()), |b| {
        b.iter(|| {
            spmv_csr_parallel(&a, &x, &mut y).unwrap();
        });
    });

    group.finish();
}

criterion_group!(benches, bench_spmv);
criterion_main!(benches);
