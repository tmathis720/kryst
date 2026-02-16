use std::sync::Arc;

use kryst::KError;
use kryst::config::options::PcOptions;
use kryst::context::pc_context::PcFactory;
use kryst::matrix::op::CsrOp;
use kryst::matrix::op::DistLayout;
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
fn fieldsplit_schur_factorization_variants_apply() -> Result<(), KError> {
    let csr = CsrMatrix::from_csr(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![
            S::from_real(4.0),
            S::from_real(1.0),
            S::from_real(1.0),
            S::from_real(3.0),
        ],
    );
    let op = CsrOp::new(Arc::new(csr));
    for fact in ["diag", "lower", "upper", "full"] {
        let opts = PcOptions {
            pc_type: Some("fieldsplit".into()),
            pc_fieldsplit_block_sizes: Some(vec![1, 1]),
            pc_fieldsplit_child_pc_type: Some("jacobi".into()),
            pc_fieldsplit_type: Some("schur".into()),
            pc_fieldsplit_schur_fact_type: Some(fact.into()),
            pc_fieldsplit_schur_precondition: Some("self".into()),
            ..Default::default()
        };
        let mut pc = PcFactory::create_from_options(&opts)?;
        pc.setup(&op)?;

        let x = vec![S::from_real(2.0), S::from_real(3.0)];
        let mut y = vec![S::zero(); 2];
        pc.apply(PcSide::Left, &x, &mut y)?;
        assert!(y.iter().all(|v| v.abs() > 0.0));
    }
    Ok(())
}

#[test]
fn fieldsplit_schur_full_precondition_and_nested_children() -> Result<(), KError> {
    let csr = CsrMatrix::from_csr(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![
            S::from_real(2.0),
            S::from_real(1.0),
            S::from_real(1.0),
            S::from_real(2.0),
        ],
    );
    let op = CsrOp::new(Arc::new(csr));
    let args = [
        "-pc_type",
        "fieldsplit",
        "-pc_fieldsplit_block_sizes",
        "1,1",
        "-pc_fieldsplit_type",
        "schur",
        "-pc_fieldsplit_schur_fact_type",
        "full",
        "-pc_fieldsplit_schur_precondition",
        "full",
        "-pc_fieldsplit_prefixes",
        "pc_fieldsplit_0_,pc_fieldsplit_1_",
        "-pc_fieldsplit_0_pc_type",
        "jacobi",
        "-pc_fieldsplit_1_pc_type",
        "none",
    ];
    let opts = PcOptions::from_args(&args)?;
    let mut pc = PcFactory::create_from_options(&opts)?;
    pc.setup(&op)?;
    let x = vec![S::from_real(2.0), S::from_real(2.0)];
    let mut y = vec![S::zero(); 2];
    pc.apply(PcSide::Left, &x, &mut y)?;
    assert!(y[0].abs() > 0.0);
    Ok(())
}

#[test]
fn fieldsplit_layout_rejects_mixed_block_sizes() {
    let csr = diag_csr(&[1.0, 2.0, 3.0]);
    let op = CsrOp::new(Arc::new(csr)).with_layout(DistLayout {
        global_rows: 5,
        global_cols: 5,
        row_start: 0,
        row_end: 3,
        col_start: 0,
        col_end: 3,
    });
    let opts = PcOptions {
        pc_type: Some("fieldsplit".into()),
        pc_fieldsplit_block_sizes: Some(vec![2, 2]),
        pc_fieldsplit_child_pc_type: Some("jacobi".into()),
        pc_fieldsplit_type: Some("additive".into()),
        ..Default::default()
    };
    let mut pc = PcFactory::create_from_options(&opts).expect("create fieldsplit");
    let err = pc.setup(&op).expect_err("mixed local/global must fail");
    let msg = err.to_string();
    assert!(msg.contains("mixed local/global") || msg.contains("must sum to local"));
}

#[test]
fn fieldsplit_layout_rejects_inconsistent_dist_layout() {
    let csr = diag_csr(&[1.0, 2.0]);
    let op = CsrOp::new(Arc::new(csr)).with_layout(DistLayout {
        global_rows: 4,
        global_cols: 4,
        row_start: 1,
        row_end: 4,
        col_start: 1,
        col_end: 3,
    });
    let opts = PcOptions {
        pc_type: Some("fieldsplit".into()),
        pc_fieldsplit_block_sizes: Some(vec![1, 1]),
        pc_fieldsplit_child_pc_type: Some("jacobi".into()),
        ..Default::default()
    };
    let mut pc = PcFactory::create_from_options(&opts).expect("create fieldsplit");
    let err = pc
        .setup(&op)
        .expect_err("inconsistent distributed layout must fail");
    assert!(err.to_string().contains("layout/local row mismatch"));
}

#[cfg(feature = "complex")]
#[test]
fn fieldsplit_complex_rejects_diag_schur_precondition() {
    let csr = diag_csr(&[1.0, 1.0]);
    let op = CsrOp::new(Arc::new(csr));
    let opts = PcOptions {
        pc_type: Some("fieldsplit".into()),
        pc_fieldsplit_block_sizes: Some(vec![1, 1]),
        pc_fieldsplit_child_pc_type: Some("jacobi".into()),
        pc_fieldsplit_type: Some("schur".into()),
        pc_fieldsplit_schur_fact_type: Some("diag".into()),
        pc_fieldsplit_schur_precondition: Some("diag".into()),
        ..Default::default()
    };
    let mut pc = PcFactory::create_from_options(&opts).expect("create fieldsplit");
    let err = pc.setup(&op).expect_err("complex diag schur should fail");
    assert!(
        err.to_string()
            .contains("not supported for complex scalars")
    );
}
