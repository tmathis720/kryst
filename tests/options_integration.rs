#![cfg(not(feature = "complex"))]
//! Tests for the PETSc-style options parsing and integration.

use kryst::config::options::{CgVariant, KspOptions, PcOptions, PcSide, parse_all_options};
use kryst::context::ksp_context::{KspContext, SolverType};
use kryst::error::KError;

#[test]
fn test_ksp_options_from_args() {
    let args = vec![
        "-ksp_type",
        "gmres",
        "-ksp_rtol",
        "1e-8",
        "-ksp_max_it",
        "500",
    ];
    let opts = KspOptions::from_args(&args).unwrap();

    assert_eq!(opts.ksp_type, Some("gmres".to_string()));
    assert_eq!(opts.rtol, Some(1e-8));
    assert_eq!(opts.maxits, Some(500));
    assert_eq!(opts.atol, None); // Not specified
}

#[test]
fn cg_variant_options_precedence_and_staging() {
    let opts = KspOptions::from_args(&[
        "-ksp_cg_pipelined",
        "false",
        "-ksp_cg_variant",
        "pipelined",
        "-ksp_cg_replace_every",
        "7",
        "-ksp_cg_use_async",
        "false",
        "-ksp_cg_async_min_n",
        "123",
    ])
    .unwrap();
    assert_eq!(opts.cg_variant, Some(CgVariant::Pipelined));
    assert_eq!(opts.cg_pipelined, Some(true));
    assert_eq!(opts.cg_replace_every, Some(7));

    let mut ksp = KspContext::new();
    ksp.set_from_options(&opts).unwrap();
    ksp.set_type(SolverType::Cg).unwrap();
    let view = ksp.view();
    assert_eq!(
        view.solver_config
            .get("cg_variant")
            .and_then(|v| v.as_str()),
        Some("Pipelined")
    );
    assert_eq!(
        view.solver_config
            .get("cg_replace_every")
            .and_then(|v| v.as_u64()),
        Some(7)
    );
    assert_eq!(
        view.solver_config
            .get("cg_async_enabled")
            .and_then(|v| v.as_bool()),
        Some(false)
    );
    assert_eq!(
        view.solver_config
            .get("cg_async_effective")
            .and_then(|v| v.as_bool()),
        Some(false)
    );
    assert_eq!(
        view.solver_config
            .get("cg_async_min_n")
            .and_then(|v| v.as_u64()),
        Some(123)
    );
    assert_eq!(
        view.solver_config
            .get("cg_async_overlap_safe")
            .and_then(|v| v.as_bool()),
        Some(true)
    );
    let model = view
        .solver_config
        .get("cg_reduction_model")
        .and_then(|v| v.as_object())
        .expect("cg reduction model");
    assert_eq!(
        model.get("variant").and_then(|v| v.as_str()),
        Some("cg-pipelined")
    );
    assert_eq!(model.get("startup").and_then(|v| v.as_u64()), Some(1));
    assert_eq!(
        model.get("per_iteration").and_then(|v| v.as_f64()),
        Some(1.0)
    );
}

#[test]
fn pcg_view_reports_cg_wrapper_diagnostics() {
    let opts = KspOptions::from_args(&[
        "-ksp_cg_variant",
        "pipelined",
        "-ksp_cg_replace_every",
        "11",
        "-ksp_cg_use_async",
        "true",
        "-ksp_cg_async_min_n",
        "99",
        "-ksp_reproducible",
        "true",
    ])
    .unwrap();

    let mut ksp = KspContext::new();
    ksp.set_from_options(&opts).unwrap();
    ksp.set_type(SolverType::Pcg).unwrap();
    let view = ksp.view();

    assert_eq!(
        view.solver_config
            .get("cg_variant")
            .and_then(|v| v.as_str()),
        Some("Pipelined")
    );
    assert_eq!(
        view.solver_config
            .get("cg_replace_every")
            .and_then(|v| v.as_u64()),
        Some(11)
    );
    assert_eq!(
        view.solver_config
            .get("cg_async_enabled")
            .and_then(|v| v.as_bool()),
        Some(true)
    );
    assert_eq!(
        view.solver_config
            .get("cg_async_effective")
            .and_then(|v| v.as_bool()),
        Some(false)
    );
    assert_eq!(
        view.solver_config
            .get("cg_async_min_n")
            .and_then(|v| v.as_u64()),
        Some(99)
    );
    assert_eq!(
        view.solver_config
            .get("cg_async_overlap_safe")
            .and_then(|v| v.as_bool()),
        Some(false)
    );
    let model = view
        .solver_config
        .get("cg_reduction_model")
        .and_then(|v| v.as_object())
        .expect("pcg reduction model");
    assert_eq!(
        model.get("variant").and_then(|v| v.as_str()),
        Some("pcg-pipelined")
    );
    assert_eq!(model.get("startup").and_then(|v| v.as_u64()), Some(1));
    assert_eq!(
        model.get("per_iteration").and_then(|v| v.as_f64()),
        Some(1.0)
    );
}

#[test]
fn cg_replace_every_zero_disables_refresh_diagnostic() {
    let opts =
        KspOptions::from_args(&["-ksp_cg_variant", "pipelined", "-ksp_cg_replace_every", "0"])
            .unwrap();
    assert_eq!(opts.cg_replace_every, Some(0));

    let mut cg = KspContext::new();
    cg.set_from_options(&opts).unwrap();
    cg.set_type(SolverType::Cg).unwrap();
    let cg_view = cg.view();
    assert_eq!(
        cg_view
            .solver_config
            .get("cg_variant")
            .and_then(|v| v.as_str()),
        Some("Pipelined")
    );
    assert!(cg_view.solver_config.get("cg_replace_every").is_none());

    let mut pcg = KspContext::new();
    pcg.set_from_options(&opts).unwrap();
    pcg.set_type(SolverType::Pcg).unwrap();
    let pcg_view = pcg.view();
    assert_eq!(
        pcg_view
            .solver_config
            .get("cg_variant")
            .and_then(|v| v.as_str()),
        Some("Pipelined")
    );
    assert!(pcg_view.solver_config.get("cg_replace_every").is_none());
}

#[test]
fn deterministic_reduction_mode_disables_cg_async_overlap_diagnostics() {
    let opts = KspOptions::from_args(&[
        "-ksp_cg_variant",
        "pipelined",
        "-ksp_cg_use_async",
        "true",
        "-ksp_reduction",
        "deterministic",
    ])
    .unwrap();

    let mut ksp = KspContext::new();
    ksp.set_from_options(&opts).unwrap();
    ksp.set_type(SolverType::Cg).unwrap();
    let view = ksp.view();

    assert_eq!(
        view.solver_config
            .get("cg_async_enabled")
            .and_then(|v| v.as_bool()),
        Some(true)
    );
    assert_eq!(
        view.solver_config
            .get("cg_async_overlap_safe")
            .and_then(|v| v.as_bool()),
        Some(false)
    );
    assert_eq!(
        view.solver_config
            .get("cg_async_effective")
            .and_then(|v| v.as_bool()),
        Some(false)
    );
}

#[cfg(feature = "backend-faer")]
#[test]
fn cg_view_reports_csr_operator_route() {
    use std::sync::Arc;

    use kryst::matrix::{CsrMatrix, CsrOp, LinOp};

    let csr = CsrMatrix::from_csr(
        3,
        3,
        vec![0, 2, 4, 5],
        vec![0, 1, 0, 2, 2],
        vec![4.0, 1.0, 1.0, 3.0, 2.0],
    );
    let op: Arc<dyn LinOp<S = f64>> = Arc::new(CsrOp::new(Arc::new(csr)));

    let mut ksp = KspContext::new();
    ksp.set_type(SolverType::Cg).unwrap();
    ksp.set_operators(op, None);

    let view = ksp.view();
    assert_eq!(
        view.solver_config
            .get("operator_format")
            .and_then(|v| v.as_str()),
        Some("Csr")
    );
    assert_eq!(
        view.solver_config
            .get("operator_route")
            .and_then(|v| v.as_str()),
        Some("csr")
    );
    assert_eq!(
        view.solver_config
            .get("operator_comm_size")
            .and_then(|v| v.as_u64()),
        Some(1)
    );
    assert_eq!(
        view.solver_config
            .get("operator_distributed_layout")
            .and_then(|v| v.as_bool()),
        Some(false)
    );
}

#[cfg(feature = "backend-faer")]
#[test]
fn pcg_view_reports_generic_csr_operator_route() {
    use std::sync::Arc;

    use kryst::matrix::csr::CsrMatrix as ScalarCsrMatrix;
    use kryst::matrix::spmv::plan::SpmvTuning;
    use kryst::matrix::{GenericCsrOp, LinOp};

    let csr = ScalarCsrMatrix::new(
        3,
        3,
        vec![0, 2, 4, 5],
        vec![0, 1, 0, 2, 2],
        vec![4.0, 1.0, 1.0, 3.0, 2.0],
    );
    let op: Arc<dyn LinOp<S = f64>> =
        Arc::new(GenericCsrOp::new(Arc::new(csr), &SpmvTuning::default()));

    let mut ksp = KspContext::new();
    ksp.set_type(SolverType::Pcg).unwrap();
    ksp.set_operators(op, None);

    let view = ksp.view();
    assert_eq!(
        view.solver_config
            .get("operator_format")
            .and_then(|v| v.as_str()),
        Some("Csr")
    );
    assert_eq!(
        view.solver_config
            .get("operator_route")
            .and_then(|v| v.as_str()),
        Some("generic-csr")
    );
}

#[test]
fn test_pc_options_from_args() {
    let args = vec!["-pc_type", "jacobi", "-pc_ilu_levels", "2"];
    let opts = PcOptions::from_args(&args).unwrap();

    assert_eq!(opts.pc_type, Some("jacobi".to_string()));
    assert_eq!(opts.ilu_level, Some(2));
    assert_eq!(opts.chebyshev_degree, None); // Not specified
}

#[test]
fn test_mixed_ksp_pc_args() {
    let args = vec!["-ksp_type", "cg", "-pc_type", "ilu0", "-ksp_rtol", "1e-10"];

    let ksp_opts = KspOptions::from_args(&args).unwrap();
    let pc_opts = PcOptions::from_args(&args).unwrap();

    assert_eq!(ksp_opts.ksp_type, Some("cg".to_string()));
    assert_eq!(ksp_opts.rtol, Some(1e-10));
    assert_eq!(pc_opts.pc_type, Some("ilu0".to_string()));
}

#[test]
fn test_invalid_ksp_option() {
    let args = vec!["-ksp_unknown_option", "value"];
    let result = KspOptions::from_args(&args);
    assert!(result.is_err());
}

#[test]
fn test_missing_value() {
    let args = vec!["-ksp_type"]; // Missing value
    let result = KspOptions::from_args(&args);
    assert!(result.is_err());

    if let Err(KError::SolveError(msg)) = result {
        assert!(msg.contains("Missing value for -ksp_type"));
    } else {
        panic!("Expected SolveError with missing value message");
    }
}

#[test]
fn test_invalid_numeric_value() {
    let args = vec!["-ksp_rtol", "not_a_number"];
    let result = KspOptions::from_args(&args);
    assert!(result.is_err());
}

#[test]
fn test_pc_side_parsing() {
    use std::str::FromStr;

    assert_eq!(PcSide::from_str("left").unwrap(), PcSide::Left);
    assert_eq!(PcSide::from_str("right").unwrap(), PcSide::Right);
    assert_eq!(PcSide::from_str("symmetric").unwrap(), PcSide::Symmetric);
    assert_eq!(PcSide::from_str("LEFT").unwrap(), PcSide::Left); // Case insensitive

    assert!(PcSide::from_str("invalid").is_err());
}

#[test]
fn test_ksp_context_set_from_options() {
    let args = vec![
        "-ksp_type",
        "gmres",
        "-ksp_rtol",
        "1e-8",
        "-ksp_max_it",
        "500",
    ];
    let opts = KspOptions::from_args(&args).unwrap();

    let mut ksp = KspContext::new();
    ksp.set_from_options(&opts).unwrap();

    // Verify the context was configured correctly
    assert_eq!(ksp.rtol, 1e-8);
    assert_eq!(ksp.maxits, 500);
    // Note: We can't easily test the solver type without making fields public,
    // but the fact that set_from_options succeeded means the solver was created
}

#[test]
fn test_ksp_context_set_from_all_options() {
    let ksp_args = vec!["-ksp_type", "cg", "-ksp_rtol", "1e-9"];
    let pc_args = vec!["-pc_type", "jacobi"];

    let ksp_opts = KspOptions::from_args(&ksp_args).unwrap();
    let pc_opts = PcOptions::from_args(&pc_args).unwrap();

    let mut ksp = KspContext::new();
    ksp.set_from_all_options(&ksp_opts, &pc_opts).unwrap();

    assert_eq!(ksp.rtol, 1e-9);
}

#[test]
fn test_parse_all_options() {
    let args = vec![
        "-ksp_type".to_string(),
        "gmres".to_string(),
        "-ksp_rtol".to_string(),
        "1e-8".to_string(),
        "-pc_type".to_string(),
        "jacobi".to_string(),
    ];

    let (ksp_opts, pc_opts) = parse_all_options(&args).unwrap();

    assert_eq!(ksp_opts.ksp_type, Some("gmres".to_string()));
    assert_eq!(ksp_opts.rtol, Some(1e-8));
    assert_eq!(pc_opts.pc_type, Some("jacobi".to_string()));
}

#[test]
fn test_empty_args() {
    let args: Vec<&str> = vec![];
    let ksp_opts = KspOptions::from_args(&args).unwrap();
    let pc_opts = PcOptions::from_args(&args).unwrap();

    // All options should be None (using defaults)
    assert_eq!(ksp_opts.ksp_type, None);
    assert_eq!(ksp_opts.rtol, None);
    assert_eq!(pc_opts.pc_type, None);
    assert_eq!(pc_opts.ilu_level, None);
}

#[test]
fn test_non_option_args_ignored() {
    let args = vec![
        "program_name",
        "-ksp_type",
        "cg",
        "some_file.txt",
        "-pc_type",
        "jacobi",
        "--verbose",
    ];

    let ksp_opts = KspOptions::from_args(&args).unwrap();
    let pc_opts = PcOptions::from_args(&args).unwrap();

    assert_eq!(ksp_opts.ksp_type, Some("cg".to_string()));
    assert_eq!(pc_opts.pc_type, Some("jacobi".to_string()));
    // Non-option arguments should be ignored
}

#[test]
fn test_hierarchical_fieldsplit_prefixes_isolate_options() {
    let args = vec![
        "-pc_type",
        "jacobi",
        "-pc_fieldsplit_0_pc_type",
        "ilu",
        "-pc_fieldsplit_0_pc_ilu_levels",
        "2",
        "-pc_fieldsplit_1_pc_type",
        "amg",
        "-pc_fieldsplit_1_pc_amg_levels",
        "3",
    ];
    let opts = PcOptions::from_args(&args).unwrap();
    assert_eq!(opts.pc_type.as_deref(), Some("jacobi"));
    let expected_prefixes = vec![
        "pc_fieldsplit_0_".to_string(),
        "pc_fieldsplit_1_".to_string(),
    ];
    assert_eq!(
        opts.pc_fieldsplit_prefixes.as_ref(),
        Some(&expected_prefixes)
    );
    let first = opts.scoped_child("pc_fieldsplit_0_").unwrap();
    let second = opts.scoped_child("pc_fieldsplit_1_").unwrap();
    assert_eq!(first.pc_type.as_deref(), Some("ilu"));
    assert_eq!(first.ilu_level, Some(2));
    assert_eq!(first.amg_levels, None);
    assert_eq!(second.pc_type.as_deref(), Some("amg"));
    assert_eq!(second.amg_levels, Some(3));
    assert_eq!(second.ilu_level, None);
}

#[test]
fn test_ksp_pc_scoped_options_do_not_leak() {
    let args = vec![
        "-pc_type",
        "ksp",
        "-pc_ksp_ksp_type",
        "gmres",
        "-pc_ksp_ksp_rtol",
        "1e-4",
        "-pc_ksp_pc_type",
        "ilu",
        "-pc_ksp_pc_ilu_levels",
        "3",
    ];
    let opts = PcOptions::from_args(&args).unwrap();
    assert_eq!(opts.pc_type.as_deref(), Some("ksp"));
    assert_eq!(opts.ilu_level, None);
    let nested_ksp = opts.pc_ksp_ksp_options.as_ref().unwrap();
    assert_eq!(nested_ksp.ksp_type.as_deref(), Some("gmres"));
    assert_eq!(nested_ksp.rtol, Some(1e-4));
    let nested_pc = opts.pc_ksp_pc_options.as_ref().unwrap();
    assert_eq!(nested_pc.pc_type.as_deref(), Some("ilu"));
    assert_eq!(nested_pc.ilu_level, Some(3));
}

#[test]
fn test_solver_type_from_str() {
    use kryst::context::ksp_context::SolverType;
    use std::str::FromStr;

    assert_eq!(SolverType::from_str("cg").unwrap(), SolverType::Cg);
    assert_eq!(SolverType::from_str("CG").unwrap(), SolverType::Cg); // Case insensitive
    assert_eq!(SolverType::from_str("gmres").unwrap(), SolverType::Gmres);
    assert_eq!(
        SolverType::from_str("bicgstab").unwrap(),
        SolverType::BiCgStab
    );
    assert_eq!(SolverType::from_str("tfqmr").unwrap(), SolverType::Tfqmr);
    assert_eq!(
        SolverType::from_str("preonly").unwrap(),
        SolverType::Preonly
    );

    assert!(SolverType::from_str("invalid_solver").is_err());
}

#[test]
fn test_pc_type_from_str() {
    use kryst::context::pc_context::PcType;
    use std::str::FromStr;

    assert_eq!(PcType::from_str("jacobi").unwrap(), PcType::Jacobi);
    assert_eq!(PcType::from_str("JACOBI").unwrap(), PcType::Jacobi); // Case insensitive
    assert_eq!(PcType::from_str("ilu0").unwrap(), PcType::Ilu0);
    assert_eq!(PcType::from_str("none").unwrap(), PcType::None);

    // Test new direct solver types
    assert_eq!(PcType::from_str("lu").unwrap(), PcType::Lu);
    assert_eq!(PcType::from_str("LU").unwrap(), PcType::Lu); // Case insensitive
    assert_eq!(PcType::from_str("qr").unwrap(), PcType::Qr);
    assert_eq!(PcType::from_str("QR").unwrap(), PcType::Qr); // Case insensitive

    assert!(PcType::from_str("invalid_pc").is_err());
}

#[test]
fn test_preonly_configuration() {
    use kryst::context::ksp_context::{KspContext, SolverType};

    let mut ksp = KspContext::new();

    // PREONLY is now selectable and should succeed
    assert!(ksp.set_type(SolverType::Preonly).is_ok());
}

#[test]
fn test_preonly_options_integration() {
    use kryst::config::options::{KspOptions, PcOptions};
    use kryst::context::ksp_context::KspContext;

    let args = vec!["-ksp_type", "preonly", "-pc_type", "lu"];
    let ksp_opts = KspOptions::from_args(&args).unwrap();
    let pc_opts = PcOptions::from_args(&args).unwrap();

    let mut ksp = KspContext::new();
    // PREONLY with LU should configure successfully
    assert!(ksp.set_from_all_options(&ksp_opts, &pc_opts).is_ok());
}

#[test]
fn test_bicgstab_variant_options_parse() {
    let args = vec![
        "-ksp_type",
        "bicgstab",
        "-ksp_bicgstab_variant",
        "reliable",
        "-ksp_bicgstab_replace_every",
        "7",
    ];
    let opts = KspOptions::from_args(&args).unwrap();
    assert_eq!(opts.ksp_type.as_deref(), Some("bicgstab"));
    assert_eq!(opts.bicgstab_variant.as_deref(), Some("reliable"));
    assert_eq!(opts.bicgstab_replace_every, Some(7));
}
