use kryst::utils::matrix_market::read_matrix_market;
use kryst::utils::matrix_screening::cg_compatibility_screen;

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
