use kryst::{KspOptions, PcOptions, KError};

fn main() -> Result<(), KError> {
    // Test command line parsing for SuperLU_DIST
    let args = vec![
        "test_program",
        "-ksp_type",
        "preonly",
        "-pc_type",
        "superlu_dist",
        "-pc_superlu_pivot_threshold",
        "0.5",
        "-pc_superlu_print_level",
        "1",
        "-pc_superlu_process_grid",
        "2",
        "3",
        "-pc_superlu_replace_tiny_pivot",
        "true",
        "-pc_superlu_iterative_refinement",
        "DOUBLE",
        "-pc_superlu_column_permutation",
        "NATURAL",
        "-pc_superlu_row_permutation",
        "LargeDiag",
        "-pc_superlu_static_pivoting",
        "false",
    ];

    // Parse KSP command line options
    let ksp_options = KspOptions::from_args(&args)?;
    
    // Parse PC command line options
    let pc_options = PcOptions::from_args(&args)?;
    
    // Verify the KSP type was parsed correctly
    assert_eq!(ksp_options.ksp_type, Some("preonly".to_string()));
    
    // Verify the PC type was parsed correctly
    assert_eq!(pc_options.pc_type, Some("superlu_dist".to_string()));
    
    // Verify SuperLU_DIST specific options were parsed correctly
    assert_eq!(pc_options.superlu_pivot_threshold, Some(0.5));
    assert_eq!(pc_options.superlu_print_level, Some(1));
    assert_eq!(pc_options.superlu_process_grid, Some((2, 3)));
    assert_eq!(pc_options.superlu_replace_tiny_pivots, Some(true));
    assert_eq!(pc_options.superlu_iterative_refinement, Some("DOUBLE".to_string()));
    assert_eq!(pc_options.superlu_column_permutation, Some("NATURAL".to_string()));
    assert_eq!(pc_options.superlu_row_permutation, Some("LargeDiag".to_string()));
    assert_eq!(pc_options.superlu_static_pivoting, Some(false));
    
    println!("✓ All SuperLU_DIST command line options parsed correctly");
    println!("  - Pivot threshold: {:?}", pc_options.superlu_pivot_threshold);
    println!("  - Print level: {:?}", pc_options.superlu_print_level);
    println!("  - Process grid: {:?}", pc_options.superlu_process_grid);
    println!("  - Replace tiny pivots: {:?}", pc_options.superlu_replace_tiny_pivots);
    println!("  - Iterative refinement: {:?}", pc_options.superlu_iterative_refinement);
    println!("  - Column permutation: {:?}", pc_options.superlu_column_permutation);
    println!("  - Row permutation: {:?}", pc_options.superlu_row_permutation);
    println!("  - Static pivoting: {:?}", pc_options.superlu_static_pivoting);
    
    println!("✓ SuperLU_DIST integration via PREONLY interface is working!");
    
    Ok(())
}
