use crate::algebra::prelude::*;
use crate::matrix::sparse::CsrMatrix;
use crate::preconditioner::amg::{AMGBuilder, RelaxType};
use crate::preconditioner::{
    DeflationOptions, DeflationPC, Jacobi, PcSide, Preconditioner, ZSource,
};
use faer::Mat;

fn poisson_1d(n: usize) -> CsrMatrix<R> {
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::new();
    let mut values = Vec::new();
    row_ptr.push(0);
    for i in 0..n {
        if i > 0 {
            col_idx.push(i - 1);
            values.push(R::from(-1.0));
        }
        col_idx.push(i);
        values.push(R::from(2.0));
        if i + 1 < n {
            col_idx.push(i + 1);
            values.push(R::from(-1.0));
        }
        row_ptr.push(col_idx.len());
    }
    CsrMatrix::from_csr(n, n, row_ptr, col_idx, values)
}

#[test]
fn extract_coarse_space_identity() {
    let a = poisson_1d(12);
    let mut amg = AMGBuilder::new()
        .relaxation_type(RelaxType::Jacobi)
        .grid_relax_type_all(RelaxType::Jacobi)
        .build(&Mat::<R>::zeros(0, 0))
        .unwrap();
    amg.setup(&a).unwrap();

    let opts = DeflationOptions::default();
    let cs = amg.extract_coarse_space(&opts).unwrap();
    assert_eq!(cs.z.nrows(), a.nrows());
    assert!(cs.z.ncols() > 0);
}

#[test]
fn deflation_preconditioner_applies() {
    let a = poisson_1d(10);
    let mut amg = AMGBuilder::new()
        .relaxation_type(RelaxType::Jacobi)
        .grid_relax_type_all(RelaxType::Jacobi)
        .require_spd(true)
        .build(&Mat::<R>::zeros(0, 0))
        .unwrap();
    amg.setup(&a).unwrap();
    let opts = DeflationOptions::default();
    let cs = amg.extract_coarse_space(&opts).unwrap();
    let mut def = DeflationPC::new(amg, &a, cs, &opts).unwrap();
    def.setup(&a).unwrap();

    let rhs: Vec<R> = (0..a.nrows()).map(|i| R::from((i as f64).sin())).collect();
    let mut out = vec![R::default(); rhs.len()];
    def.apply(PcSide::Left, &rhs, &mut out).unwrap();
    assert!(out.iter().all(|v| v.is_finite()));
    assert_eq!(def.fine_dim(), rhs.len());
}

#[test]
fn gram_schmidt_drops_dependent_columns() {
    let n = 5;
    let diag = CsrMatrix::from_csr(
        n,
        n,
        (0..=n).collect(),
        (0..n).collect(),
        vec![R::from(1.0); n],
    );
    let mut z = Mat::<R>::zeros(n, 2);
    for i in 0..n {
        z[(i, 0)] = R::from(1.0);
        z[(i, 1)] = R::from(1.0);
    }
    let coarse = crate::preconditioner::AmgCoarseSpace {
        z,
        local_range: None,
    };
    let opts = DeflationOptions {
        z_source: ZSource::External,
        cond_cap: Some(R::from(10.0)),
        augment_initial_guess: false,
    };
    let jacobi = Jacobi::new();
    let def = DeflationPC::new(jacobi, &diag, coarse, &opts).unwrap();
    assert_eq!(def.coarse_dim(), 1);
}
