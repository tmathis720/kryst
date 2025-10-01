//! Example demonstrating comprehensive AMG and ILU command-line option parsing.
//!
//! This example shows how to parse and use all the AMG and ILU options
//! in the Kryst linear solver library.
//! 
//! to run:
//! ```sh
//! cargo run --example amg_options_demo -- -ksp_type cg -pc_type ilu -pc_ilu_type ilu0 -pc_ilu_reordering_type rcm
//! ```

use kryst::config::options::{KspOptions, PcOptions, parse_all_options};
use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Kryst Comprehensive Preconditioner Options Demonstration");
    println!("=========================================================");
    println!();

    // Example 1: Basic AMG setup
    let basic_amg_args = vec![
        "-ksp_type".to_string(),
        "cg".to_string(),
        "-pc_type".to_string(),
        "amg".to_string(),
        "-pc_amg_levels".to_string(),
        "10".to_string(),
        "-pc_amg_strength_threshold".to_string(),
        "0.5".to_string(),
    ];

    println!("Example 1: Basic AMG Configuration");
    println!("Command: -ksp_type cg -pc_type amg -pc_amg_levels 10 -pc_amg_strength_threshold 0.5");
    let (ksp_opts, pc_opts) = parse_all_options(&basic_amg_args)?;
    print_configuration(&ksp_opts, &pc_opts);
    println!();

    // Example 2: Comprehensive ILU setup
    let comprehensive_ilu_args = vec![
        "-ksp_type".to_string(),
        "gmres".to_string(),
        "-ksp_rtol".to_string(),
        "1e-8".to_string(),
        "-pc_type".to_string(),
        "ilu".to_string(),
        "-pc_ilu_type".to_string(),
        "ilut".to_string(),
        "-pc_ilu_level_of_fill".to_string(),
        "3".to_string(),
        "-pc_ilu_max_fill_per_row".to_string(),
        "50".to_string(),
        "-pc_ilu_reordering_type".to_string(),
        "rcm".to_string(),
        "-pc_ilu_triangular_solve".to_string(),
        "iterative".to_string(),
        "-pc_ilu_lower_jacobi_iters".to_string(),
        "2".to_string(),
        "-pc_ilu_upper_jacobi_iters".to_string(),
        "3".to_string(),
        "-pc_ilu_tolerance".to_string(),
        "1e-10".to_string(),
        "-pc_ilu_ieee_checks".to_string(),
        "true".to_string(),
        "-pc_ilu_pivot_monitoring".to_string(),
        "true".to_string(),
        "-pc_ilu_print_level".to_string(),
        "1".to_string(),
    ];

    println!("Example 2: Comprehensive ILU Configuration");
    println!("Command: -ksp_type gmres -ksp_rtol 1e-8 -pc_type ilu \\");
    println!("         -pc_ilu_type ilut -pc_ilu_level_of_fill 3 \\");
    println!("         -pc_ilu_max_fill_per_row 50 -pc_ilu_reordering_type rcm \\");
    println!("         -pc_ilu_triangular_solve iterative -pc_ilu_lower_jacobi_iters 2 \\");
    println!("         -pc_ilu_upper_jacobi_iters 3 -pc_ilu_tolerance 1e-10 \\");
    println!(
        "         -pc_ilu_ieee_checks true -pc_ilu_pivot_monitoring true -pc_ilu_print_level 1"
    );
    let (ksp_opts, pc_opts) = parse_all_options(&comprehensive_ilu_args)?;
    print_configuration(&ksp_opts, &pc_opts);
    println!();

    // Example 3: Advanced AMG setup
    let advanced_amg_args = vec![
        "-ksp_type".to_string(),
        "bicgstab".to_string(),
        "-ksp_rtol".to_string(),
        "1e-6".to_string(),
        "-pc_type".to_string(),
        "amg".to_string(),
        "-pc_amg_levels".to_string(),
        "15".to_string(),
        "-pc_amg_strength_threshold".to_string(),
        "0.25".to_string(),
        "-pc_amg_coarsen_type".to_string(),
        "hmis".to_string(),
        "-pc_amg_interp_type".to_string(),
        "classical".to_string(),
        "-pc_amg_relax_type".to_string(),
        "gs".to_string(),
        "-pc_amg_nu_pre".to_string(),
        "2".to_string(),
        "-pc_amg_nu_post".to_string(),
        "2".to_string(),
        "-pc_amg_print_level".to_string(),
        "1".to_string(),
    ];

    println!("Example 3: Advanced AMG Configuration");
    println!("Command: -ksp_type bicgstab -ksp_rtol 1e-6 -pc_type amg \\");
    println!("         -pc_amg_levels 15 -pc_amg_strength_threshold 0.25 \\");
    println!("         -pc_amg_coarsen_type hmis -pc_amg_interp_type classical \\");
    println!(
        "         -pc_amg_relax_type gs -pc_amg_nu_pre 2 -pc_amg_nu_post 2 -pc_amg_print_level 1"
    );
    let (ksp_opts, pc_opts) = parse_all_options(&advanced_amg_args)?;
    print_configuration(&ksp_opts, &pc_opts);
    println!();

    // Example 4: Parse from command line if arguments provided
    let args: Vec<String> = env::args().collect();
    if args.len() > 1 {
        println!("Example 4: Command-line Arguments");
        println!("Arguments: {:?}", &args[1..]);
        let (ksp_opts, pc_opts) = parse_all_options(&args[1..])?;
        print_configuration(&ksp_opts, &pc_opts);
    } else {
        println!("Example 4: No command-line arguments provided.");
        println!(
            "Try running with: cargo run --example amg_options_demo -- -ksp_type cg -pc_type ilu -pc_ilu_type ilu0 -pc_ilu_reordering_type rcm"
        );
    }

    println!();
    println!("Use -help to see all available options.");

    Ok(())
}

fn print_configuration(ksp_opts: &KspOptions, pc_opts: &PcOptions) {
    println!("Configuration:");

    // KSP options
    if let Some(ref solver) = ksp_opts.ksp_type {
        println!("  Solver: {}", solver);
    }
    if let Some(rtol) = ksp_opts.rtol {
        println!("  Relative tolerance: {:.2e}", rtol);
    }
    if let Some(atol) = ksp_opts.atol {
        println!("  Absolute tolerance: {:.2e}", atol);
    }
    if let Some(maxits) = ksp_opts.maxits {
        println!("  Maximum iterations: {}", maxits);
    }
    if let Some(restart) = ksp_opts.restart {
        println!("  GMRES restart: {}", restart);
    }

    // PC options
    if let Some(ref pc_type) = pc_opts.pc_type {
        println!("  Preconditioner: {}", pc_type);
    }

    // AMG-specific options
    if pc_opts.pc_type.as_deref() == Some("amg") {
        println!("  AMG Configuration:");
        if let Some(levels) = pc_opts.amg_levels {
            println!("    Levels: {}", levels);
        }
        if let Some(threshold) = pc_opts.amg_strength_threshold {
            println!("    Strength threshold: {}", threshold);
        }
        if let Some(nu_pre) = pc_opts.amg_nu_pre {
            println!("    Pre-smoothing iterations: {}", nu_pre);
        }
        if let Some(nu_post) = pc_opts.amg_nu_post {
            println!("    Post-smoothing iterations: {}", nu_post);
        }
        if let Some(ref coarsen_type) = pc_opts.amg_coarsen_type {
            println!("    Coarsening type: {}", coarsen_type);
        }
        if let Some(ref interp_type) = pc_opts.amg_interp_type {
            println!("    Interpolation type: {}", interp_type);
        }
        if let Some(ref relax_type) = pc_opts.amg_relax_type {
            println!("    Relaxation type: {}", relax_type);
        }
        if let Some(max_coarse) = pc_opts.amg_max_coarse_size {
            println!("    Maximum coarse size: {}", max_coarse);
        }
        if let Some(min_coarse) = pc_opts.amg_min_coarse_size {
            println!("    Minimum coarse size: {}", min_coarse);
        }
        if let Some(trunc_factor) = pc_opts.amg_truncation_factor {
            println!("    Truncation factor: {}", trunc_factor);
        }
        if let Some(print_level) = pc_opts.amg_print_level {
            println!("    Print level: {}", print_level);
        }
        if let Some(ieee_checks) = pc_opts.amg_ieee_checks {
            println!("    IEEE checks: {}", ieee_checks);
        }
        if let Some(optimize_workspace) = pc_opts.amg_optimize_workspace {
            println!("    Optimize workspace: {}", optimize_workspace);
        }
    }

    // ILU-specific options
    if pc_opts.pc_type.as_deref() == Some("ilu") {
        println!("  ILU Configuration:");
        if let Some(ref ilu_type) = pc_opts.ilu_type {
            println!("    ILU type: {}", ilu_type);
        }
        if let Some(level_of_fill) = pc_opts.ilu_level_of_fill {
            println!("    Level of fill: {}", level_of_fill);
        }
        if let Some(max_fill_per_row) = pc_opts.ilu_max_fill_per_row {
            println!("    Max fill per row: {}", max_fill_per_row);
        }
        if let Some(drop_tol) = pc_opts.ilut_drop_tol {
            println!("    Drop tolerance: {:.2e}", drop_tol);
        }
        if let Some(offdiag_drop_tol) = pc_opts.ilu_offdiag_drop_tolerance {
            println!("    Off-diagonal drop tolerance: {:.2e}", offdiag_drop_tol);
        }
        if let Some(schur_drop_tol) = pc_opts.ilu_schur_drop_tolerance {
            println!(
                "    Schur complement drop tolerance: {:.2e}",
                schur_drop_tol
            );
        }
        if let Some(ref reordering) = pc_opts.ilu_reordering_type {
            println!("    Reordering type: {}", reordering);
        }
        if let Some(ref tri_solve) = pc_opts.ilu_triangular_solve {
            println!("    Triangular solve: {}", tri_solve);
        }
        if let Some(lower_iters) = pc_opts.ilu_lower_jacobi_iters {
            println!("    Lower Jacobi iterations: {}", lower_iters);
        }
        if let Some(upper_iters) = pc_opts.ilu_upper_jacobi_iters {
            println!("    Upper Jacobi iterations: {}", upper_iters);
        }
        if let Some(tolerance) = pc_opts.ilu_tolerance {
            println!("    ILU tolerance: {:.2e}", tolerance);
        }
        if let Some(max_iter) = pc_opts.ilu_max_iterations {
            println!("    Max iterations: {}", max_iter);
        }
        if let Some(print_level) = pc_opts.ilu_print_level {
            println!("    Print level: {}", print_level);
        }
        if let Some(ieee_checks) = pc_opts.ilu_ieee_checks {
            println!("    IEEE checks: {}", ieee_checks);
        }
        if let Some(pivot_monitoring) = pc_opts.ilu_pivot_monitoring {
            println!("    Pivot monitoring: {}", pivot_monitoring);
        }
        if let Some(optimize_workspace) = pc_opts.ilu_optimize_workspace {
            println!("    Optimize workspace: {}", optimize_workspace);
        }
        if let Some(pivot_threshold) = pc_opts.ilu_pivot_threshold {
            println!("    Pivot threshold: {:.2e}", pivot_threshold);
        }
    }

    // Other PC options
    if let Some(ilu_level) = pc_opts.ilu_level {
        println!("  ILU levels (legacy): {}", ilu_level);
    }
    if let Some(degree) = pc_opts.chebyshev_degree {
        println!("  Chebyshev degree: {}", degree);
    }
    if let Some(ref reorder) = pc_opts.reorder {
        println!("  Matrix reordering: {}", reorder);
    }
    if let Some(ref scaling) = pc_opts.scaling {
        println!("  Matrix scaling: {}", scaling);
    }
}
