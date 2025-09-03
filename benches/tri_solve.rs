use criterion::{Criterion, black_box, criterion_group, criterion_main};
use kryst::matrix::sparse::CsrMatrix;

#[path = "infra/datasets.rs"]
mod datasets;

fn extract_lower_unit(a: &CsrMatrix<f64>) -> CsrMatrix<f64> {
    let n = a.nrows();
    let mut rp = Vec::with_capacity(n + 1);
    let mut cj = Vec::new();
    let mut vv = Vec::new();
    rp.push(0);
    for i in 0..n {
        for p in a.row_ptr()[i]..a.row_ptr()[i + 1] {
            let j = a.col_idx()[p];
            if j < i {
                cj.push(j);
                vv.push(a.values()[p]);
            }
        }
        // unit diagonal
        cj.push(i);
        vv.push(1.0);
        rp.push(cj.len());
    }
    CsrMatrix::from_csr(n, n, rp, cj, vv)
}

fn extract_upper(a: &CsrMatrix<f64>) -> CsrMatrix<f64> {
    let n = a.nrows();
    let mut rp = Vec::with_capacity(n + 1);
    let mut cj = Vec::new();
    let mut vv = Vec::new();
    rp.push(0);
    for i in 0..n {
        for p in a.row_ptr()[i]..a.row_ptr()[i + 1] {
            let j = a.col_idx()[p];
            if j >= i {
                cj.push(j);
                vv.push(a.values()[p]);
            }
        }
        rp.push(cj.len());
    }
    CsrMatrix::from_csr(n, n, rp, cj, vv)
}

// Serial forward solve for lower triangular with unit diagonal
fn csr_forward_unit_diag(l: &CsrMatrix<f64>, b: &[f64], x: &mut [f64]) {
    let n = l.nrows();
    assert_eq!(b.len(), n);
    assert_eq!(x.len(), n);
    for i in 0..n {
        let mut sum = 0.0;
        for p in l.row_ptr()[i]..l.row_ptr()[i + 1] {
            let j = l.col_idx()[p];
            if j < i {
                sum += l.values()[p] * x[j];
            }
        }
        x[i] = b[i] - sum; // diag = 1
    }
}

// Serial backward solve for upper triangular with explicit diagonal
fn csr_backward(u: &CsrMatrix<f64>, b: &[f64], x: &mut [f64]) {
    let n = u.nrows();
    assert_eq!(b.len(), n);
    assert_eq!(x.len(), n);
    for i in (0..n).rev() {
        let mut sum = 0.0;
        let mut diag = 0.0;
        for p in u.row_ptr()[i]..u.row_ptr()[i + 1] {
            let j = u.col_idx()[p];
            let v = u.values()[p];
            if j == i {
                diag = v;
            } else if j > i {
                sum += v * x[j];
            }
        }
        x[i] = if diag.abs() > 1e-14 {
            (b[i] - sum) / diag
        } else {
            0.0
        };
    }
}

fn bench_tri_solve(c: &mut Criterion) {
    let n = 200_000;
    let a = datasets::random_powerlaw_like(n, 10, 777);
    let l = extract_lower_unit(&a);
    let u = extract_upper(&a);
    let b = vec![1.0; n];
    let mut x = vec![0.0; n];

    c.bench_function("forward_serial", |bch| {
        bch.iter(|| {
            x.fill(0.0);
            csr_forward_unit_diag(&l, &b, &mut x);
            black_box(&x);
        })
    });
    c.bench_function("backward_serial", |bch| {
        bch.iter(|| {
            x.fill(0.0);
            csr_backward(&u, &b, &mut x);
            black_box(&x);
        })
    });
}

criterion_group!(benches, bench_tri_solve);
criterion_main!(benches);
