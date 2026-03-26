#![cfg(not(feature = "complex"))]

use std::collections::BTreeSet;

// #[test]
// fn no_distributed_claims_without_mpi() {
//     #[cfg(not(feature = "mpi"))]
//     {
//         let readme = include_str!("../README.md").to_lowercase();
//         for line in readme.lines() {
//             if line.contains("distributed") {
//                 assert!(
//                     line.contains("mpi"),
//                     "Line mentions distributed without stating `mpi`: {line}"
//                 );
//             }
//         }
//     }
// }

#[test]
fn petsc_mapping_convergence_table_tracks_code_reasons() {
    let docs = include_str!("../docs/petsc_mapping.md");
    let convergence = include_str!("../src/utils/convergence.rs");

    let mut mapped_reasons = BTreeSet::new();
    for line in convergence.lines() {
        if let Some((_, rhs)) = line.split_once("=> \"") {
            if let Some((petsc, _)) = rhs.split_once('"') {
                if petsc.starts_with("KSP_") {
                    mapped_reasons.insert(petsc.to_string());
                }
            }
        }
    }

    for petsc_reason in &mapped_reasons {
        let row_token = format!("| `{petsc_reason}` |");
        assert!(
            docs.contains(&row_token),
            "docs/petsc_mapping.md convergence table is missing row for mapped reason {petsc_reason}"
        );
    }

    for reason in [
        "KSP_DIVERGED_BREAKDOWN",
        "KSP_DIVERGED_BREAKDOWN_BICG",
        "KSP_DIVERGED_NANORINF",
    ] {
        let unsupported_row = format!("| `{reason}` | — | Unsupported |");
        assert!(
            !docs.contains(&unsupported_row),
            "docs/petsc_mapping.md still marks implemented reason {reason} as unsupported"
        );
    }
}
