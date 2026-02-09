use std::sync::Arc;

use kryst::config::options::PcOptions;
use kryst::context::pc_context::PcFactory;
use kryst::KError;
use kryst::matrix::op::CsrOp;
use kryst::matrix::sparse::CsrMatrix;
use kryst::prelude::*;

fn diag_csr(diag: &[f64]) -> CsrMatrix<S> {
    let n = diag.len();
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::with_capacity(n);
    let mut values = Vec::with_capacity(n);
    row_ptr.push(0);
    for (i, val) in diag.iter().enumerate() {
        col_idx.push(i);
        values.push(S::from_real(*val));
        row_ptr.push(i + 1);
    }
    CsrMatrix::from_csr(n, n, row_ptr, col_idx, values)
}

#[test]
fn fieldsplit_blockdiag_jacobi_scales_blocks() -> Result<(), KError> {
    let csr = diag_csr(&[2.0, 2.0, 4.0, 4.0]);
    let op = CsrOp::new(Arc::new(csr));
    let opts = PcOptions {
        pc_type: Some("fieldsplit".into()),
        pc_fieldsplit_block_sizes: Some(vec![2, 2]),
        pc_fieldsplit_child_pc_type: Some("jacobi".into()),
        pc_fieldsplit_type: Some("additive".into()),
        ..Default::default()
    };
    let mut pc = PcFactory::create_from_options(&opts)?;
    pc.setup(&op)?;

    let x = vec![
        S::from_real(2.0),
        S::from_real(4.0),
        S::from_real(8.0),
        S::from_real(12.0),
    ];
    let mut y = vec![S::zero(); 4];
    pc.apply(PcSide::Left, &x, &mut y)?;

    let expected = vec![
        S::from_real(1.0),
        S::from_real(2.0),
        S::from_real(2.0),
        S::from_real(3.0),
    ];
    assert_eq!(y, expected);
    Ok(())
}

#[test]
fn fieldsplit_schur_lower_uses_offdiag_blocks() -> Result<(), KError> {
    let csr = CsrMatrix::from_csr(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![
            S::from_real(2.0),
            S::from_real(1.0),
            S::from_real(1.0),
            S::from_real(3.0),
        ],
    );
    let op = CsrOp::new(Arc::new(csr));
    let opts = PcOptions {
        pc_type: Some("fieldsplit".into()),
        pc_fieldsplit_block_sizes: Some(vec![1, 1]),
        pc_fieldsplit_child_pc_type: Some("jacobi".into()),
        pc_fieldsplit_type: Some("schur".into()),
        pc_fieldsplit_schur_fact_type: Some("lower".into()),
        pc_fieldsplit_schur_precondition: Some("self".into()),
        ..Default::default()
    };
    let mut pc = PcFactory::create_from_options(&opts)?;
    pc.setup(&op)?;

    let x = vec![S::from_real(2.0), S::from_real(3.0)];
    let mut y = vec![S::zero(); 2];
    pc.apply(PcSide::Left, &x, &mut y)?;

    assert!((y[0] - S::from_real(1.0)).abs() < 1e-12);
    assert!((y[1] - S::from_real(2.0 / 3.0)).abs() < 1e-12);
    Ok(())
}
