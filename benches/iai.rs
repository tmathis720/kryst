#![cfg(feature = "iai")]
use iai_callgrind::{black_box, library_benchmark, main};
use kryst::context::ksp_context::{KspContext, SolverType};
use kryst::context::pc_context::PcType;
use kryst::matrix::op::CsrOp;
use kryst::matrix::sparse::CsrMatrix;
use std::sync::Arc;

fn csr_poisson_1d(n: usize) -> CsrMatrix<f64> {
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::new();
    let mut vals = Vec::new();
    row_ptr.push(0);
    for i in 0..n {
        if i > 0 { col_idx.push(i - 1); vals.push(-1.0); }
        col_idx.push(i); vals.push(2.0);
        if i + 1 < n { col_idx.push(i + 1); vals.push(-1.0); }
        row_ptr.push(col_idx.len());
    }
    CsrMatrix::from_csr(n, n, row_ptr, col_idx, vals)
}

fn cg_jacobi_1d400() {
    let n = 400;
    let a = csr_poisson_1d(n);
    let b = vec![1.0; n];
    let aop = CsrOp::new(Arc::new(a));
    let mut ksp = KspContext::new();
    ksp.set_type(SolverType::Cg).unwrap()
       .set_pc_type(PcType::Jacobi, None).unwrap()
       .set_tolerances(1e-6, 1e-12, 1e6, 2000);
    ksp.set_operators(Arc::new(aop), None);
    ksp.setup().unwrap();
    let mut x = vec![0.0; n];
    let _ = ksp.solve(black_box(&b), black_box(&mut x)).unwrap();
}

main!(library_benchmark("cg_jacobi_1d400", cg_jacobi_1d400),);

#[cfg(not(feature = "iai"))]
fn main() {}
