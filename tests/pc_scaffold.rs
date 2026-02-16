#![cfg(feature = "backend-faer")]

use std::sync::Arc;

use faer::Mat;
use kryst::context::pc_context::PcFactory;
use kryst::error::KError;
use kryst::matrix::op::DenseOp;
use kryst::prelude::*;

#[test]
fn pc_options_parse_scaffold_prefixes() {
    let args = vec![
        "-pc_type",
        "fieldsplit",
        "-pc_fieldsplit_block_sizes",
        "2,2",
        "-pc_fieldsplit_child_pc_type",
        "jacobi",
        "-pc_fieldsplit_type",
        "additive",
        "-pc_ksp_type",
        "gmres",
        "-pc_ksp_pc_type",
        "jacobi",
        "-pc_ksp_maxits",
        "2",
        "-pc_ksp_rtol",
        "1e-4",
        "-pc_mg_levels",
        "3",
        "-pc_mg_cycle_type",
        "v",
        "-pc_mg_smoother",
        "jacobi",
        "-pc_mg_smoother_steps",
        "2",
        "-pc_shell_name",
        "demo",
        "-pc_bddc_coarse_ksp_type",
        "cg",
        "-pc_bddc_coarse_pc_type",
        "jacobi",
        "-pc_bddc_use_vertices",
        "-pc_gamg_type",
        "agg",
        "-pc_gamg_threshold",
        "0.1",
        "-pc_gamg_levels",
        "3",
        "-pc_gamg_coarsen_type",
        "pmis",
        "-pc_gamg_interp_type",
        "standard",
        "-pc_gamg_aggressive_levels",
        "2",
        "-pc_gamg_aggressive_mis_k",
        "5",
    ];
    let opts = PcOptions::from_args(&args).unwrap();
    assert_eq!(opts.pc_fieldsplit_block_sizes, Some(vec![2, 2]));
    assert_eq!(opts.pc_fieldsplit_child_pc_type.as_deref(), Some("jacobi"));
    assert_eq!(opts.pc_fieldsplit_type.as_deref(), Some("additive"));
    assert_eq!(opts.pc_ksp_ksp_type.as_deref(), Some("gmres"));
    assert_eq!(opts.pc_ksp_pc_type.as_deref(), Some("jacobi"));
    assert_eq!(opts.pc_ksp_maxits, Some(2));
    assert_eq!(opts.pc_ksp_rtol, Some(1e-4));
    assert_eq!(opts.pc_mg_levels, Some(3));
    assert_eq!(opts.pc_mg_cycle_type.as_deref(), Some("v"));
    assert_eq!(opts.pc_mg_smoother.as_deref(), Some("jacobi"));
    assert_eq!(opts.pc_mg_smoother_steps, Some(2));
    assert_eq!(opts.pc_shell_name.as_deref(), Some("demo"));
    assert_eq!(opts.pc_bddc_coarse_ksp_type.as_deref(), Some("cg"));
    assert_eq!(opts.pc_bddc_coarse_pc_type.as_deref(), Some("jacobi"));
    assert_eq!(opts.pc_bddc_use_vertices, Some(true));
    assert_eq!(opts.pc_gamg_type.as_deref(), Some("agg"));
    assert_eq!(opts.pc_gamg_threshold, Some(0.1));
    assert_eq!(opts.pc_gamg_levels, Some(3));
    assert_eq!(opts.pc_gamg_coarsen_type.as_deref(), Some("pmis"));
    assert_eq!(opts.pc_gamg_interp_type.as_deref(), Some("standard"));
    assert_eq!(opts.pc_gamg_aggressive_levels, Some(2));
    assert_eq!(opts.pc_gamg_aggressive_mis_k, Some(5));
}

#[test]
fn pc_scaffold_apply_smoke() -> Result<(), KError> {
    let mat = Mat::<R>::identity(4, 4);
    let op = DenseOp::<S>::new(Arc::new(mat));
    let x = vec![
        S::from_real(1.0),
        S::from_real(2.0),
        S::from_real(3.0),
        S::from_real(4.0),
    ];

    let fieldsplit_opts = PcOptions {
        pc_type: Some("fieldsplit".into()),
        pc_fieldsplit_block_sizes: Some(vec![2, 2]),
        pc_fieldsplit_child_pc_type: Some("none".into()),
        ..Default::default()
    };
    let mut fieldsplit = PcFactory::create_from_options(&fieldsplit_opts)?;
    fieldsplit.setup(&op)?;
    let mut y = vec![S::zero(); 4];
    fieldsplit.apply(PcSide::Left, &x, &mut y)?;
    assert_eq!(y, x);

    let mg_opts = PcOptions {
        pc_type: Some("mg".into()),
        pc_mg_levels: Some(2),
        ..Default::default()
    };
    let mut mg = PcFactory::create_from_options(&mg_opts)?;
    mg.setup(&op)?;
    let mut y = vec![S::zero(); 4];
    mg.apply(PcSide::Left, &x, &mut y)?;
    assert_eq!(y, x);

    let ksp_opts = PcOptions {
        pc_type: Some("ksp".into()),
        pc_ksp_pc_type: Some("none".into()),
        pc_ksp_maxits: Some(1),
        ..Default::default()
    };
    let mut ksp_pc = PcFactory::create_from_options(&ksp_opts)?;
    ksp_pc.setup(&op)?;
    let mut y = vec![S::zero(); 4];
    ksp_pc.apply(PcSide::Left, &x, &mut y)?;
    assert_eq!(y, x);

    kryst::preconditioner::shell::register_shell_callback(
        "double",
        kryst::preconditioner::shell::shell_apply(|_side, x: &[S], y: &mut [S]| {
            for (yi, xi) in y.iter_mut().zip(x.iter()) {
                *yi = *xi + *xi;
            }
            Ok(())
        }),
    );
    let shell_opts = PcOptions {
        pc_type: Some("shell".into()),
        pc_shell_name: Some("double".into()),
        ..Default::default()
    };
    let mut shell = PcFactory::create_from_options(&shell_opts)?;
    shell.setup(&op)?;
    let mut y = vec![S::zero(); 4];
    shell.apply(PcSide::Left, &x, &mut y)?;
    assert_eq!(
        y,
        vec![
            S::from_real(2.0),
            S::from_real(4.0),
            S::from_real(6.0),
            S::from_real(8.0)
        ]
    );

    Ok(())
}

#[test]
fn pc_scaffold_unsupported_types() {
    let bddc_opts = PcOptions {
        pc_type: Some("bddc".into()),
        ..Default::default()
    };
    PcFactory::create_from_options(&bddc_opts).expect("expected bddc scaffold to construct");

    let gamg_opts = PcOptions {
        pc_type: Some("gamg".into()),
        ..Default::default()
    };
    PcFactory::create_from_options(&gamg_opts).expect("expected gamg scaffold to construct");
}
