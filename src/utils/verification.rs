//! Reusable verification-state modeling for optional direct reference checks.

/// Normalized verification status for reporting direct reference checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerificationStatus {
    Yes,
    No,
    Skip,
    Unavailable,
}

impl VerificationStatus {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Yes => "yes",
            Self::No => "no",
            Self::Skip => "skip",
            Self::Unavailable => "unavailable",
        }
    }
}

/// Lightweight abstraction for direct-reference comparison payloads.
pub trait DirectReferenceLike {
    fn matches_verified_answer(&self) -> bool;
    fn policy_note(&self) -> &str;
}

/// Maps direct-reference comparison output and policy note into a normalized status.
pub fn verification_status_from_direct_reference<T: DirectReferenceLike>(
    comparison: Option<&T>,
) -> VerificationStatus {
    let Some(comparison) = comparison else {
        return VerificationStatus::Unavailable;
    };

    let note = comparison.policy_note().to_ascii_lowercase();
    let is_skip = note.starts_with("skip:")
        || note.contains("auto skip")
        || note.contains("forced off")
        || note.contains("policy skip");
    if is_skip {
        return VerificationStatus::Skip;
    }

    let is_unavailable = note.contains("feature is disabled")
        || note.contains("not enabled")
        || note.contains("unavailable")
        || note.contains("prevented")
        || note.contains("direct lu failed");
    if is_unavailable {
        return VerificationStatus::Unavailable;
    }

    if comparison.matches_verified_answer() {
        VerificationStatus::Yes
    } else {
        VerificationStatus::No
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DirectReferenceLike, VerificationStatus, verification_status_from_direct_reference,
    };

    struct MockCmp {
        matches: bool,
        note: &'static str,
    }

    impl DirectReferenceLike for MockCmp {
        fn matches_verified_answer(&self) -> bool {
            self.matches
        }

        fn policy_note(&self) -> &str {
            self.note
        }
    }

    #[test]
    fn maps_yes_no_skip_and_unavailable() {
        let yes = MockCmp {
            matches: true,
            note: "auto: density 1.0e-1 >= 1.0e-1 and size gate passed",
        };
        assert_eq!(
            verification_status_from_direct_reference(Some(&yes)),
            VerificationStatus::Yes
        );

        let no = MockCmp {
            matches: false,
            note: "env override: forced on",
        };
        assert_eq!(
            verification_status_from_direct_reference(Some(&no)),
            VerificationStatus::No
        );

        let skip = MockCmp {
            matches: false,
            note: "auto skip: density 2.1e-3 < 1.0e-1 (size gate passed)",
        };
        assert_eq!(
            verification_status_from_direct_reference(Some(&skip)),
            VerificationStatus::Skip
        );

        let unavailable = MockCmp {
            matches: false,
            note: "env override: forced on; direct LU failed (dense-direct feature is disabled)",
        };
        assert_eq!(
            verification_status_from_direct_reference(Some(&unavailable)),
            VerificationStatus::Unavailable
        );
    }
}
