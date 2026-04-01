use crate::config::options::KspOptions;

/// Benchmark/demo-oriented GMRES policy profile.
///
/// This helper centralizes robust defaults so demos and benchmarks do not carry
/// ad-hoc local toggles.
pub fn benchmark_demo_gmres_profile(difficult_nonsymmetric: bool) -> KspOptions {
    let mut opts = KspOptions {
        gmres_orthog: Some("mgs".to_string()),
        ..KspOptions::default()
    };

    if difficult_nonsymmetric {
        opts.gmres_reorth = Some("always".to_string());
        // Keep if-needed threshold on the stricter side for paths that may
        // still switch policies.
        opts.gmres_reorth_tol = Some(0.5);
    } else {
        opts.gmres_reorth = Some("ifneeded".to_string());
        opts.gmres_reorth_tol = Some(0.7);
    }

    opts
}

#[cfg(test)]
mod tests {
    use super::benchmark_demo_gmres_profile;

    #[test]
    fn demo_profile_defaults_to_mgs() {
        let opts = benchmark_demo_gmres_profile(false);
        assert_eq!(opts.gmres_orthog.as_deref(), Some("mgs"));
        assert_eq!(opts.gmres_reorth.as_deref(), Some("ifneeded"));
    }

    #[test]
    fn demo_profile_tightens_reorth_for_difficult_nonsymmetric() {
        let opts = benchmark_demo_gmres_profile(true);
        assert_eq!(opts.gmres_orthog.as_deref(), Some("mgs"));
        assert_eq!(opts.gmres_reorth.as_deref(), Some("always"));
        assert_eq!(opts.gmres_reorth_tol, Some(0.5));
    }
}
