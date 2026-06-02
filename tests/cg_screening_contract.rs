#![cfg(feature = "backend-faer")]

use kryst::utils::matrix_market::read_matrix_market;
use kryst::utils::matrix_screening::{
    SYMMETRY_MAX_ASYMMETRY_RATE, assess_symmetry, cg_compatibility_screen,
};

#[test]
fn fidap005_remains_cg_eligible_with_soft_warnings_only() {
    let base = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join("mtx")
        .join("fidap005.mtx");
    let matrix = read_matrix_market(base.to_str().expect("path to string"))
        .expect("read fidap005")
        .to_csr_matrix()
        .expect("csr conversion");

    let screen = cg_compatibility_screen(&matrix, false, None, false);
    assert!(
        !screen.is_hard_reject,
        "fidap005 should not be hard-rejected; got {:?}",
        screen.hard_reject_reasons
    );
    assert!(
        !screen.warnings.is_empty(),
        "screen should preserve soft-warning visibility when metadata is missing"
    );
}

#[test]
fn sherman3_cannot_be_spd_like_when_sampled_nonsymmetry_is_high() {
    let base = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join("mtx")
        .join("sherman3.mtx");
    let matrix = read_matrix_market(base.to_str().expect("path to string"))
        .expect("read sherman3")
        .to_csr_matrix()
        .expect("csr conversion");

    let symmetry = assess_symmetry(&matrix, None, false);
    let spd_like_hint = symmetry.passes_threshold;
    let high_nonsymmetry = symmetry.symmetry_violation_rate > SYMMETRY_MAX_ASYMMETRY_RATE;

    assert!(
        !(spd_like_hint && high_nonsymmetry),
        "contradictory symmetry diagnostics: spd_like_hint={} with asymmetry={:.2}% (threshold {:.2}%)",
        spd_like_hint,
        100.0 * symmetry.symmetry_violation_rate,
        100.0 * SYMMETRY_MAX_ASYMMETRY_RATE
    );
}
