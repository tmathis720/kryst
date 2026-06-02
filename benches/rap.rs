use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use kryst::matrix::sparse::CsrMatrix;
use kryst::matrix::utils::{rap_btree, rap_opt};
use rand::{RngExt, SeedableRng};
use std::hint::black_box;

#[path = "infra/datasets.rs"]
mod datasets;

fn validate_equal(a: &CsrMatrix<f64>, b: &CsrMatrix<f64>) {
    assert_eq!(a.nrows(), b.nrows());
    assert_eq!(a.ncols(), b.ncols());
    // Compare after sorting rows if necessary: our generators emit sorted rows already,
    // and both kernels maintain sorted columns per row, so do direct compare.
    assert_eq!(a.row_ptr(), b.row_ptr());
    assert_eq!(a.col_idx(), b.col_idx());
    let av = a.values();
    let bv = b.values();
    assert_eq!(av.len(), bv.len());
    let mut max_diff = 0.0f64;
    for i in 0..av.len() {
        max_diff = max_diff.max((av[i] - bv[i]).abs());
    }
    assert!(max_diff <= 1e-12, "numeric mismatch: max_diff={}", max_diff);
}

fn bench_rap_poisson2d_400(c: &mut Criterion) {
    let n = 400; // N = 160k, nnz ~ 800k
    let (a, p, r) = datasets::rap_triplet_poisson2d(n);

    // Lightweight pre-check
    let c_ref = rap_btree(&r, &a, &p).expect("rap_btree failed");
    let c_opt = rap_opt(&r, &a, &p).expect("rap_opt failed");
    validate_equal(&c_ref, &c_opt);

    let mut group = c.benchmark_group("rap_poisson2d_400");
    group.bench_function("rap_btree", |b| {
        b.iter_batched(
            || (),
            |_| {
                let ch = rap_btree(&r, &a, &p).unwrap();
                black_box(ch)
            },
            BatchSize::SmallInput,
        )
    });
    group.bench_function("rap_opt", |b| {
        b.iter_batched(
            || (),
            |_| {
                let ch = rap_opt(&r, &a, &p).unwrap();
                black_box(ch)
            },
            BatchSize::SmallInput,
        )
    });
    group.finish();
}

fn bench_rap_powerlaw(c: &mut Criterion) {
    let n = 200_000; // average ~10–12 nnz/row
    let a = datasets::random_powerlaw_like(n, 10, 42);
    // Build a skinny P with ~2 nonzeros per row by sampling a smaller coarse set
    let nc = n / 20;
    let mut p_rp = Vec::with_capacity(n + 1);
    let mut p_cj = Vec::with_capacity(2 * n);
    let mut p_vv = Vec::with_capacity(2 * n);
    p_rp.push(0);
    let mut rng = rand::rngs::StdRng::seed_from_u64(123);
    for _i in 0..n {
        let c0 = rng.random_range(0..nc);
        let offset = 1 + rng.random_range(0..3);
        let c1 = if c0 + offset < nc {
            c0 + offset
        } else {
            c0 - offset
        };
        let (a, b) = if c0 <= c1 { (c0, c1) } else { (c1, c0) };
        p_cj.push(a);
        p_vv.push(0.7);
        p_cj.push(b);
        p_vv.push(0.3);
        p_rp.push(p_cj.len());
    }
    let p = CsrMatrix::from_csr(n, nc, p_rp, p_cj, p_vv);

    // R = P^T (simple, not super optimized)
    let mut counts = vec![0usize; nc + 1];
    for &j in p.col_idx() {
        counts[j + 1] += 1;
    }
    for i in 0..nc {
        counts[i + 1] += counts[i];
    }
    let r_rp = counts.clone();
    let mut r_cj = vec![0usize; p.col_idx().len()];
    let mut r_vv = vec![0.0f64; p.values().len()];
    let mut next = r_rp.clone();
    for i in 0..n {
        for t in p.row_ptr()[i]..p.row_ptr()[i + 1] {
            let j = p.col_idx()[t];
            let dst = next[j];
            r_cj[dst] = i;
            r_vv[dst] = p.values()[t];
            next[j] += 1;
        }
    }
    let r = CsrMatrix::from_csr(nc, n, r_rp, r_cj, r_vv);

    // Sanity test on tiny slice to avoid OOM in CI pre-check
    {
        let a_small = datasets::random_powerlaw_like(1_000, 10, 7);
        let nc_small = 100;
        let (p_small, r_small) = {
            let mut pr = Vec::with_capacity(1_000 + 1);
            let mut pc = Vec::with_capacity(2_000);
            let mut pv = Vec::with_capacity(2_000);
            pr.push(0);
            let mut rng = rand::rngs::StdRng::seed_from_u64(99);
            for _ in 0..1_000 {
                let c0 = rng.random_range(0..nc_small);
                let offset = 1 + rng.random_range(0..3);
                let c1 = if c0 + offset < nc_small {
                    c0 + offset
                } else {
                    c0 - offset
                };
                let (a, b) = if c0 <= c1 { (c0, c1) } else { (c1, c0) };
                pc.push(a);
                pv.push(0.7);
                pc.push(b);
                pv.push(0.3);
                pr.push(pc.len());
            }
            let p = CsrMatrix::from_csr(1_000, nc_small, pr, pc, pv);
            // transpose
            let mut counts = vec![0usize; nc_small + 1];
            for &j in p.col_idx() {
                counts[j + 1] += 1;
            }
            for i in 0..nc_small {
                counts[i + 1] += counts[i];
            }
            let r_rp = counts.clone();
            let mut r_cj = vec![0usize; p.col_idx().len()];
            let mut r_vv = vec![0.0f64; p.values().len()];
            let mut next = r_rp.clone();
            for i in 0..1_000 {
                for t in p.row_ptr()[i]..p.row_ptr()[i + 1] {
                    let j = p.col_idx()[t];
                    let dst = next[j];
                    r_cj[dst] = i;
                    r_vv[dst] = p.values()[t];
                    next[j] += 1;
                }
            }
            (p, CsrMatrix::from_csr(nc_small, 1_000, r_rp, r_cj, r_vv))
        };
        let c_ref = rap_btree(&r_small, &a_small, &p_small).unwrap();
        let c_opt = rap_opt(&r_small, &a_small, &p_small).unwrap();
        validate_equal(&c_ref, &c_opt);
    }

    let mut group = c.benchmark_group("rap_powerlaw_200k");
    group.bench_function("rap_btree", |b| {
        b.iter_batched(
            || (),
            |_| black_box(rap_btree(&r, &a, &p).unwrap()),
            BatchSize::SmallInput,
        )
    });
    group.bench_function("rap_opt", |b| {
        b.iter_batched(
            || (),
            |_| black_box(rap_opt(&r, &a, &p).unwrap()),
            BatchSize::SmallInput,
        )
    });
    group.finish();
}

pub fn bench_rap(c: &mut Criterion) {
    bench_rap_poisson2d_400(c);
    // The 200k case is heavier; keep enabled for full runs.
    // Comment out locally if resource-limited.
    bench_rap_powerlaw(c);
}

criterion_group!(benches, bench_rap);
criterion_main!(benches);
