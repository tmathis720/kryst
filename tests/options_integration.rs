#![cfg(not(feature = "complex"))]
//! Tests for the PETSc-style options parsing and integration.

use kryst::config::options::{KspOptions, PcOptions, PcSide, parse_all_options};
use kryst::context::ksp_context::KspContext;
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
