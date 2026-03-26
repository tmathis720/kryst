#![cfg(feature = "backend-faer")]
use std::sync::Arc;

use kryst::KError;
use kryst::config::options::PcOptions;
use kryst::context::pc_context::PcFactory;
use kryst::matrix::op::CsrOp;
use kryst::matrix::op::DistLayout;
use kryst::matrix::sparse::CsrMatrix;
use kryst::preconditioner::fieldsplit::FieldSplitPc;
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
        for pre in [
            "self",
            "selfp",
            "diag",
            "a11",
            "full",
            "full_matfree",
            "user",
        ] {
            let opts = PcOptions {
                pc_type: Some("fieldsplit".into()),
                pc_fieldsplit_block_sizes: Some(vec![1, 1]),
                pc_fieldsplit_child_pc_type: Some("jacobi".into()),
                pc_fieldsplit_type: Some("schur".into()),
                pc_fieldsplit_schur_fact_type: Some(fact.into()),
                pc_fieldsplit_schur_precondition: Some(pre.into()),
                ..Default::default()
            };
            let mut pc = PcFactory::create_from_options(&opts)?;
            pc.setup(&op)?;

            let x = vec![S::from_real(2.0), S::from_real(3.0)];
            let mut y = vec![S::zero(); 2];
            pc.apply(PcSide::Left, &x, &mut y)?;
            assert!(y.iter().all(|v| v.abs() > 0.0));
        }
    }
    Ok(())
}

#[test]
fn fieldsplit_extended_split_aliases_apply() -> Result<(), KError> {
    let csr = diag_csr(&[2.0, 3.0, 4.0, 5.0]);
    let op = CsrOp::new(Arc::new(csr));
    for split in ["basic", "gs", "symmetric_multiplicative"] {
        let opts = PcOptions {
            pc_type: Some("fieldsplit".into()),
            pc_fieldsplit_block_sizes: Some(vec![2, 2]),
            pc_fieldsplit_child_pc_type: Some("jacobi".into()),
            pc_fieldsplit_type: Some(split.into()),
            ..Default::default()
        };
        let mut pc = PcFactory::create_from_options(&opts)?;
        pc.setup(&op)?;
        let x = vec![
            S::from_real(2.0),
            S::from_real(4.0),
            S::from_real(6.0),
            S::from_real(8.0),
        ];
        let mut y = vec![S::zero(); x.len()];
        pc.apply(PcSide::Left, &x, &mut y)?;
        assert!(y.iter().all(|v| v.abs() > 0.0));
    }
    Ok(())
}

#[test]
fn fieldsplit_composite_aliases_apply() -> Result<(), KError> {
    let csr = diag_csr(&[2.0, 3.0, 4.0, 5.0]);
    let op = CsrOp::new(Arc::new(csr));
    for split in [
        "composite_additive",
        "composite_multiplicative",
        "composite_symmetric_multiplicative",
    ] {
        let opts = PcOptions {
            pc_type: Some("fieldsplit".into()),
            pc_fieldsplit_block_sizes: Some(vec![2, 2]),
            pc_fieldsplit_child_pc_type: Some("jacobi".into()),
            pc_fieldsplit_type: Some(split.into()),
            ..Default::default()
        };
        let mut pc = PcFactory::create_from_options(&opts)?;
        pc.setup(&op)?;
        let x = vec![
            S::from_real(2.0),
            S::from_real(6.0),
            S::from_real(8.0),
            S::from_real(10.0),
        ];
        let mut y = vec![S::zero(); x.len()];
        pc.apply(PcSide::Left, &x, &mut y)?;
        assert!(y.iter().all(|v| v.abs() > 0.0));
    }
    Ok(())
}

#[test]
fn fieldsplit_subksp_and_child_pc_prefixes_parse() {
    let args = [
        "-pc_type",
        "fieldsplit",
        "-pc_fieldsplit_block_sizes",
        "1,1",
        "-pc_fieldsplit_prefixes",
        "pc_fieldsplit_0_,pc_fieldsplit_1_",
        "-pc_fieldsplit_0_pc_type",
        "ksp",
        "-pc_fieldsplit_0_pc_ksp_ksp_type",
        "gmres",
        "-pc_fieldsplit_0_pc_ksp_pc_type",
        "jacobi",
        "-pc_fieldsplit_1_pc_type",
        "ilu",
    ];
    let opts = PcOptions::from_args(&args).expect("parse fieldsplit scoped args");
    let c0 = opts
        .scoped_child("pc_fieldsplit_0_")
        .expect("first child options");
    let c1 = opts
        .scoped_child("pc_fieldsplit_1_")
        .expect("second child options");
    assert_eq!(c0.pc_type.as_deref(), Some("ksp"));
    assert_eq!(
        c0.pc_ksp_ksp_options
            .as_ref()
            .and_then(|k| k.ksp_type.as_deref()),
        Some("gmres")
    );
    assert_eq!(
        c0.pc_ksp_pc_options
            .as_ref()
            .and_then(|pc| pc.pc_type.as_deref()),
        Some("jacobi")
    );
    assert_eq!(c1.pc_type.as_deref(), Some("ilu"));
}

#[test]
fn fieldsplit_extraction_modes_parse_and_apply() -> Result<(), KError> {
    let csr = diag_csr(&[2.0, 2.0]);
    let op = CsrOp::new(Arc::new(csr));
    for mode in ["extract", "cached", "zero_copy"] {
        let args = [
            "-pc_type",
            "fieldsplit",
            "-pc_fieldsplit_block_sizes",
            "2",
            "-pc_fieldsplit_type",
            "additive",
            "-pc_fieldsplit_child_pc_type",
            "jacobi",
            "-pc_fieldsplit_extraction",
            mode,
        ];
        let opts = PcOptions::from_args(&args)?;
        let mut pc = PcFactory::create_from_options(&opts)?;
        pc.setup(&op)?;
        let x = vec![S::from_real(2.0), S::from_real(4.0)];
        let mut y = vec![S::zero(); 2];
        pc.apply(PcSide::Left, &x, &mut y)?;
        assert_eq!(y, vec![S::from_real(1.0), S::from_real(2.0)]);
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
fn fieldsplit_schur_accepts_matfree_aliases() -> Result<(), KError> {
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
    for pre in ["self_p", "matfree"] {
        let opts = PcOptions {
            pc_type: Some("fieldsplit".into()),
            pc_fieldsplit_block_sizes: Some(vec![1, 1]),
            pc_fieldsplit_child_pc_type: Some("jacobi".into()),
            pc_fieldsplit_type: Some("schur".into()),
            pc_fieldsplit_schur_fact_type: Some("full".into()),
            pc_fieldsplit_schur_precondition: Some(pre.into()),
            ..Default::default()
        };
        let mut pc = PcFactory::create_from_options(&opts)?;
        pc.setup(&op)?;
        let x = vec![S::from_real(1.0), S::from_real(2.0)];
        let mut y = vec![S::zero(); 2];
        pc.apply(PcSide::Left, &x, &mut y)?;
        assert!(y.iter().all(|v| v.abs().is_finite()));
    }
    Ok(())
}

#[test]
fn fieldsplit_schur_distributed_global_block_sizes_apply() -> Result<(), KError> {
    let csr = CsrMatrix::from_csr(
        4,
        4,
        vec![0, 2, 5, 8, 10],
        vec![0, 2, 1, 2, 3, 0, 2, 3, 1, 3],
        vec![
            S::from_real(4.0),
            S::from_real(-1.0),
            S::from_real(3.0),
            S::from_real(-1.0),
            S::from_real(-0.5),
            S::from_real(-1.0),
            S::from_real(4.0),
            S::from_real(-1.0),
            S::from_real(-1.0),
            S::from_real(3.0),
        ],
    );
    let op = CsrOp::new(Arc::new(csr)).with_layout(DistLayout {
        global_rows: 8,
        global_cols: 8,
        row_start: 2,
        row_end: 6,
        col_start: 2,
        col_end: 6,
    });
    let opts = PcOptions {
        pc_type: Some("fieldsplit".into()),
        pc_fieldsplit_block_sizes: Some(vec![4, 4]),
        pc_fieldsplit_child_pc_type: Some("jacobi".into()),
        pc_fieldsplit_type: Some("schur".into()),
        pc_fieldsplit_schur_fact_type: Some("upper".into()),
        pc_fieldsplit_schur_precondition: Some("user".into()),
        ..Default::default()
    };
    let mut pc = PcFactory::create_from_options(&opts)?;
    pc.setup(&op)?;
    let x = vec![
        S::from_real(1.0),
        S::from_real(2.0),
        S::from_real(3.0),
        S::from_real(4.0),
    ];
    let mut y = vec![S::zero(); x.len()];
    pc.apply(PcSide::Left, &x, &mut y)?;
    assert!(y.iter().all(|v| v.abs().is_finite()));
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

#[cfg(feature = "complex")]
#[test]
fn fieldsplit_complex_allows_schur_self_precondition() {
    let csr = diag_csr(&[2.0, 3.0]);
    let op = CsrOp::new(Arc::new(csr));
    let opts = PcOptions {
        pc_type: Some("fieldsplit".into()),
        pc_fieldsplit_block_sizes: Some(vec![1, 1]),
        pc_fieldsplit_child_pc_type: Some("jacobi".into()),
        pc_fieldsplit_type: Some("schur".into()),
        pc_fieldsplit_schur_fact_type: Some("full".into()),
        pc_fieldsplit_schur_precondition: Some("self".into()),
        ..Default::default()
    };
    let mut pc = PcFactory::create_from_options(&opts).expect("create fieldsplit");
    pc.setup(&op).expect("complex-safe schur self setup");
}

#[test]
fn fieldsplit_strategy_and_schur_approx_parse() {
    let args = [
        "-pc_type",
        "fieldsplit",
        "-pc_fieldsplit_block_sizes",
        "1,1",
        "-pc_fieldsplit_strategy",
        "schur",
        "-pc_fieldsplit_schur_approx",
        "full",
    ];
    let opts = PcOptions::from_args(&args).expect("parse fieldsplit strategy/schur approx");
    assert_eq!(opts.resolved_fieldsplit_type(), "schur");
    assert_eq!(opts.resolved_fieldsplit_schur_approx(), "full");
}

#[test]
fn fieldsplit_comm_schedule_parse() {
    let args = [
        "-pc_type",
        "fieldsplit",
        "-pc_fieldsplit_block_sizes",
        "1,1",
        "-pc_fieldsplit_comm_schedule",
        "local_first",
    ];
    let opts = PcOptions::from_args(&args).expect("parse fieldsplit comm schedule");
    assert_eq!(opts.resolved_fieldsplit_comm_schedule(), "local_first");
}

#[test]
fn fieldsplit_split_diagnostics_record_exchange() -> Result<(), KError> {
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
    let opts = PcOptions {
        pc_fieldsplit_type: Some("schur".into()),
        pc_fieldsplit_schur_fact_type: Some("full".into()),
        pc_fieldsplit_schur_precondition: Some("full".into()),
        pc_fieldsplit_schur_approx: Some("distributed_full".into()),
        ..Default::default()
    };
    let mut pc = FieldSplitPc::new(vec![1, 1], Some("jacobi".into()), opts)?;
    pc.setup(&op)?;
    let x = vec![S::from_real(1.0), S::from_real(2.0)];
    let mut y = vec![S::zero(); 2];
    pc.apply(PcSide::Left, &x, &mut y)?;
    let diags = pc.split_diagnostics();
    assert_eq!(diags.len(), 2);
    assert!(diags.iter().map(|d| d.reduction_count).sum::<usize>() >= 1);
    Ok(())
}

#[cfg(feature = "mpi")]
#[test]
fn fieldsplit_mpi_layout_schur_exchange_path() -> Result<(), KError> {
    let csr = CsrMatrix::from_csr(
        4,
        4,
        vec![0, 2, 4, 6, 8],
        vec![0, 2, 1, 3, 0, 2, 1, 3],
        vec![
            S::from_real(3.0),
            S::from_real(-1.0),
            S::from_real(3.0),
            S::from_real(-1.0),
            S::from_real(-1.0),
            S::from_real(3.0),
            S::from_real(-1.0),
            S::from_real(3.0),
        ],
    );
    let op = CsrOp::new(Arc::new(csr)).with_layout(DistLayout {
        global_rows: 8,
        global_cols: 8,
        row_start: 2,
        row_end: 6,
        col_start: 2,
        col_end: 6,
    });
    let opts = PcOptions {
        pc_fieldsplit_block_sizes: Some(vec![4, 4]),
        pc_fieldsplit_child_pc_type: Some("jacobi".into()),
        pc_fieldsplit_type: Some("schur".into()),
        pc_fieldsplit_schur_fact_type: Some("lower".into()),
        pc_fieldsplit_schur_precondition: Some("full_matfree".into()),
        pc_fieldsplit_schur_approx: Some("dist_diag".into()),
        pc_fieldsplit_comm_schedule: Some("local_first".into()),
        ..Default::default()
    };
    let mut pc = FieldSplitPc::new(vec![4, 4], Some("jacobi".into()), opts)?;
    pc.setup(&op)?;
    let x = vec![
        S::from_real(1.0),
        S::from_real(0.5),
        S::from_real(2.0),
        S::from_real(1.5),
    ];
    let mut y = vec![S::zero(); 4];
    pc.apply(PcSide::Left, &x, &mut y)?;
    assert!(y.iter().all(|v| v.abs().is_finite()));
    Ok(())
}
