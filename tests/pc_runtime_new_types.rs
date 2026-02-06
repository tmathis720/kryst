use kryst::algebra::prelude::S;
use kryst::algebra::scalar::KrystScalar;
use kryst::config::options::PcOptions;
use kryst::context::pc_context::PcFactory;
use kryst::matrix::op::DenseOp;
use kryst::preconditioner::{PcSide, shell::register_shell_callback};
use std::sync::Arc;

fn eye_op(n: usize) -> Arc<DenseOp<S>> {
    let mat = faer::Mat::<S>::from_fn(
        n,
        n,
        |i, j| if i == j { S::from_real(1.0) } else { S::zero() },
    );
    Arc::new(DenseOp::new(Arc::new(mat)))
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
