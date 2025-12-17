#![cfg(not(feature = "complex"))]
#[test]
fn no_distributed_claims_without_mpi() {
    #[cfg(not(feature = "mpi"))]
    {
        let readme = include_str!("../README.md").to_lowercase();
        for line in readme.lines() {
            if line.contains("distributed") {
                assert!(
                    line.contains("mpi"),
                    "Line mentions distributed without stating `mpi`: {line}"
                );
            }
        }
    }
}
