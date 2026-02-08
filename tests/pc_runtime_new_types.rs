#![cfg(all(feature = "backend-faer", not(feature = "complex")))]

use kryst::algebra::prelude::S;
use kryst::algebra::scalar::KrystScalar;
use kryst::config::options::{KspOptions, PcOptions};
use kryst::context::pc_context::PcFactory;
use kryst::matrix::op::DenseOp;
use kryst::preconditioner::{PcSide, shell::register_shell_callback};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

fn eye_op(n: usize) -> Arc<DenseOp<S>> {
    let mat = faer::Mat::<S>::from_fn(
        n,
        n,
        |i, j| if i == j { S::from_real(1.0) } else { S::zero() },
    );
    Arc::new(DenseOp::new(Arc::new(mat)))
}

fn diag_op(diag: &[S]) -> Arc<DenseOp<S>> {
    let n = diag.len();
    let mat = faer::Mat::<S>::from_fn(n, n, |i, j| if i == j { diag[i] } else { S::zero() });
    Arc::new(DenseOp::new(Arc::new(mat)))
}

fn rel_error(x: &[S], x_true: &[S]) -> f64 {
    let mut num = 0.0;
    let mut denom = 0.0;
    for (xi, ti) in x.iter().zip(x_true.iter()) {
        let diff = *xi - *ti;
        num += diff.abs2();
        denom += ti.abs2();
    }
    if denom == 0.0 {
        0.0
    } else {
        (num / denom).sqrt()
    }
}

#[test]
fn options_parse_new_pc_knobs() {
    let args = vec![
        "-pc_type",
        "fieldsplit",
        "-pc_fieldsplit_block_sizes",
        "2,2",
        "-pc_fieldsplit_child_pc_type",
        "jacobi",
        "-pc_ksp_maxits",
        "4",
        "-pc_ksp_rtol",
        "1e-3",
        "-pc_mg_levels",
        "3",
    ];
    let opts = PcOptions::from_args(&args).expect("parse");
    assert_eq!(opts.pc_type.as_deref(), Some("fieldsplit"));
    assert_eq!(opts.pc_fieldsplit_block_sizes, Some(vec![2, 2]));
    assert_eq!(opts.pc_fieldsplit_child_pc_type.as_deref(), Some("jacobi"));
    assert_eq!(opts.pc_ksp_maxits, Some(4));
    assert_eq!(opts.pc_mg_levels, Some(3));
}

#[test]
fn smoke_fieldsplit_shell_ksp_mg_and_bddc_placeholder() {
    let op = eye_op(4);

    let mut fs = PcFactory::create_from_options(&PcOptions {
        pc_type: Some("fieldsplit".into()),
        pc_fieldsplit_block_sizes: Some(vec![2, 2]),
        pc_fieldsplit_child_pc_type: Some("jacobi".into()),
        ..Default::default()
    })
    .expect("fieldsplit create");
    fs.setup(op.as_ref()).expect("fieldsplit setup");
    let x = vec![S::from_real(1.0); 4];
    let mut y = vec![S::zero(); 4];
    fs.apply(PcSide::Left, &x, &mut y)
        .expect("fieldsplit apply");

    register_shell_callback(
        "scale2",
        Arc::new(|_side: PcSide, x: &[S], y: &mut [S]| {
            for (yi, xi) in y.iter_mut().zip(x.iter()) {
                *yi = *xi * S::from_real(2.0);
            }
            Ok(())
        }),
    );
    let mut sh = PcFactory::create_from_options(&PcOptions {
        pc_type: Some("shell".into()),
        pc_shell_name: Some("scale2".into()),
        ..Default::default()
    })
    .expect("shell create");
    sh.setup(op.as_ref()).expect("shell setup");
    sh.apply(PcSide::Left, &x, &mut y).expect("shell apply");
    assert_eq!(y[0], S::from_real(2.0));

    let mut ksp_pc = PcFactory::create_from_options(&PcOptions {
        pc_type: Some("ksp".into()),
        pc_ksp_pc_type: Some("jacobi".into()),
        pc_ksp_maxits: Some(2),
        ..Default::default()
    })
    .expect("ksp pc create");
    ksp_pc.setup(op.as_ref()).expect("ksp setup");
    ksp_pc.apply(PcSide::Left, &x, &mut y).expect("ksp apply");

    let mut mg = PcFactory::create_from_options(&PcOptions {
        pc_type: Some("mg".into()),
        pc_mg_levels: Some(2),
        ..Default::default()
    })
    .expect("mg create");
    mg.setup(op.as_ref()).expect("mg setup");
    mg.apply(PcSide::Left, &x, &mut y).expect("mg apply");

    let err = match PcFactory::create_from_options(&PcOptions {
        pc_type: Some("bddc".into()),
        ..Default::default()
    }) {
        Ok(_) => panic!("bddc unexpectedly succeeded"),
        Err(e) => e,
    };
    assert!(err.to_string().to_lowercase().contains("bddc"));
}

#[test]
fn ksp_pc_nested_converges() {
    let diag = vec![
        S::from_real(2.0),
        S::from_real(3.0),
        S::from_real(4.0),
    ];
    let op = diag_op(&diag);
    let b = diag.clone();
    let x_true = vec![S::from_real(1.0); diag.len()];

    let mut ksp_pc = PcFactory::create_from_options(&PcOptions {
        pc_type: Some("ksp".into()),
        pc_ksp_ksp_type: Some("cg".into()),
        pc_ksp_pc_type: Some("jacobi".into()),
        pc_ksp_maxits: Some(25),
        pc_ksp_rtol: Some(1e-12),
        ..Default::default()
    })
    .expect("ksp pc create");
    ksp_pc.setup(op.as_ref()).expect("ksp setup");
    let mut y = vec![S::zero(); diag.len()];
    ksp_pc.apply(PcSide::Left, &b, &mut y).expect("ksp apply");

    assert!(rel_error(&y, &x_true) < 1e-8);
}

#[test]
fn ksp_pc_scoped_options_override() {
    let diag = vec![
        S::from_real(2.0),
        S::from_real(3.0),
        S::from_real(4.0),
    ];
    let op = diag_op(&diag);
    let b = diag.clone();
    let x_true = vec![S::from_real(1.0); diag.len()];

    let mut base_pc = PcFactory::create_from_options(&PcOptions {
        pc_type: Some("ksp".into()),
        pc_ksp_ksp_type: Some("cg".into()),
        pc_ksp_pc_type: Some("jacobi".into()),
        pc_ksp_maxits: Some(1),
        pc_ksp_rtol: Some(1e-12),
        ..Default::default()
    })
    .expect("base ksp pc create");
    base_pc.setup(op.as_ref()).expect("base ksp setup");
    let mut y_base = vec![S::zero(); diag.len()];
    base_pc
        .apply(PcSide::Left, &b, &mut y_base)
        .expect("base ksp apply");
    let base_err = rel_error(&y_base, &x_true);

    let mut override_pc = PcFactory::create_from_options(&PcOptions {
        pc_type: Some("ksp".into()),
        pc_ksp_ksp_type: Some("cg".into()),
        pc_ksp_pc_type: Some("jacobi".into()),
        pc_ksp_maxits: Some(1),
        pc_ksp_rtol: Some(1e-12),
        pc_ksp_ksp_options: Some(KspOptions {
            maxits: Some(10),
            rtol: Some(1e-12),
            ksp_monitor_rank0: Some(true),
            ..Default::default()
        }),
        ..Default::default()
    })
    .expect("override ksp pc create");
    override_pc.setup(op.as_ref()).expect("override ksp setup");
    let mut y_override = vec![S::zero(); diag.len()];
    override_pc
        .apply(PcSide::Left, &b, &mut y_override)
        .expect("override ksp apply");
    let override_err = rel_error(&y_override, &x_true);

    assert!(override_err < base_err);
}

#[test]
fn ksp_pc_uses_inner_pc_options() {
    let diag = vec![S::from_real(2.0), S::from_real(3.0)];
    let op = diag_op(&diag);
    let b = diag.clone();

    static CALLS: AtomicUsize = AtomicUsize::new(0);
    register_shell_callback(
        "ksp_pc_inner_shell",
        Arc::new(|_side: PcSide, x: &[S], y: &mut [S]| {
            CALLS.fetch_add(1, Ordering::SeqCst);
            y.copy_from_slice(x);
            Ok(())
        }),
    );

    let mut ksp_pc = PcFactory::create_from_options(&PcOptions {
        pc_type: Some("ksp".into()),
        pc_ksp_ksp_type: Some("richardson".into()),
        pc_ksp_maxits: Some(1),
        pc_ksp_rtol: Some(1e-8),
        pc_ksp_pc_options: Some(Box::new(PcOptions {
            pc_type: Some("shell".into()),
            pc_shell_name: Some("ksp_pc_inner_shell".into()),
            ..Default::default()
        })),
        ..Default::default()
    })
    .expect("ksp pc create");
    ksp_pc.setup(op.as_ref()).expect("ksp setup");
    let mut y = vec![S::zero(); diag.len()];
    ksp_pc.apply(PcSide::Left, &b, &mut y).expect("ksp apply");

    assert!(CALLS.load(Ordering::SeqCst) > 0);
}
