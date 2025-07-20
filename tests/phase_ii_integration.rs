use faer::Mat;
use kryst::{KspContext, PcOptions};

#[test]
fn test_phase_ii_rcm_reordering_integration() {
    // Create a simple sparse matrix with poor ordering
    let matrix = Mat::<f64>::from_fn(5, 5, |i, j| {
        match (i, j) {
            (0, 0) | (1, 1) | (2, 2) | (3, 3) | (4, 4) => 2.0, // diagonal
            (0, 4) | (4, 0) => -1.0, // far off-diagonal entries (poor ordering)
            (1, 3) | (3, 1) => -1.0,
            (2, 0) | (0, 2) => -0.5,
            _ => 0.0,
        }
    });

    // Configure PC with RCM reordering and diagonal scaling
    let mut pc_options = PcOptions::new();
    pc_options.reorder = Some("rcm".to_string());
    pc_options.scaling = Some("diagonal".to_string());

    let mut context = KspContext::new();
    context.set_from_options(&kryst::KspOptions::new()).unwrap();
    context.set_pc_options(pc_options);

    // Set up with the matrix and verify preprocessing works
    let n = matrix.nrows();
    context.setup(&matrix, n).unwrap();

    // Verify that preprocessing was applied
    assert!(context.get_preprocessing().is_some());
    let preprocessing = context.get_preprocessing().unwrap();
    
    // Should not be identity transformation
    assert!(!preprocessing.is_identity);
    // Should have permutation (not empty)
    assert!(!preprocessing.permutation.is_empty());
    // Should have scaling factors for diagonal scaling
    assert!(preprocessing.left_scaling.is_some());
    assert!(preprocessing.right_scaling.is_some());

    println!("Phase II integration test passed: RCM + diagonal scaling");
}

#[test]
fn test_phase_ii_cuthill_mckee_integration() {
    // Create another test matrix
    let matrix = Mat::<f64>::from_fn(4, 4, |i, j| {
        match (i, j) {
            (0, 0) | (1, 1) | (2, 2) | (3, 3) => 3.0, // diagonal
            (0, 3) | (3, 0) => -1.0, // connection that CM should improve
            (1, 2) | (2, 1) => -1.0,
            _ => 0.0,
        }
    });

    // Configure PC with Cuthill-McKee reordering and diagonal scaling (symmetric not implemented yet)
    let mut pc_options = PcOptions::new();
    pc_options.reorder = Some("cuthill_mckee".to_string());
    pc_options.scaling = Some("diagonal".to_string());

    let mut context = KspContext::new();
    context.set_from_options(&kryst::KspOptions::new()).unwrap();
    context.set_pc_options(pc_options);

    // Set up with the matrix and verify preprocessing works
    let n = matrix.nrows();
    context.setup(&matrix, n).unwrap();

    // Verify preprocessing was applied
    assert!(context.get_preprocessing().is_some());
    let preprocessing = context.get_preprocessing().unwrap();
    
    // Should not be identity transformation
    assert!(!preprocessing.is_identity);
    // Should have permutation and scaling
    assert!(!preprocessing.permutation.is_empty());
    assert!(preprocessing.left_scaling.is_some());
    assert!(preprocessing.right_scaling.is_some());

    println!("Phase II integration test passed: Cuthill-McKee + diagonal scaling");
}

#[test]
fn test_phase_ii_no_preprocessing_when_disabled() {
    let matrix = Mat::<f64>::from_fn(3, 3, |i, j| {
        if i == j { 1.0 } else { 0.0 }
    });

    // Default options should have no preprocessing
    let ksp_options = kryst::KspOptions::new();
    let pc_options = PcOptions::new();

    let mut context = KspContext::new();
    context.set_from_options(&ksp_options).unwrap();
    context.set_pc_options(pc_options);
    
    let n = matrix.nrows();
    context.setup(&matrix, n).unwrap();

    // Should have some preprocessing (identity) when setup is called
    assert!(context.get_preprocessing().is_some());
    let preprocessing = context.get_preprocessing().unwrap();
    
    // Should be identity transformation when no reordering/scaling specified
    assert!(preprocessing.is_identity);

    println!("Phase II integration test passed: No preprocessing when disabled");
}
