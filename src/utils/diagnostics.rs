//! Structured diagnostics for KSP/PC configuration.

use crate::config::options::PcOptions;
use crate::context::ksp_context::KspContext;
use crate::context::pc_context::PcType;
use serde::Serialize;
use serde_json::Value;
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy)]
enum PcComplexSupport {
    NativeComplex,
    ProjectedComplex,
    NativeWithDegradedFallback,
}

impl PcComplexSupport {
    fn as_str(self) -> &'static str {
        match self {
            Self::NativeComplex => "native_complex",
            Self::ProjectedComplex => "projected_complex",
            Self::NativeWithDegradedFallback => "native_complex_with_degraded_fallback",
        }
    }
}

fn pc_complex_support(pc_type: Option<PcType>) -> PcComplexSupport {
    match pc_type {
        Some(PcType::Sor) => PcComplexSupport::NativeComplex,
        Some(PcType::Jacobi) | Some(PcType::Chebyshev) => PcComplexSupport::NativeComplex,
        Some(PcType::Ilu0) | Some(PcType::Ilu) | Some(PcType::Ilut) => {
            PcComplexSupport::NativeComplex
        }
        Some(PcType::ApproxInverse) => PcComplexSupport::NativeComplex,
        _ => PcComplexSupport::ProjectedComplex,
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct PcDiagnostics {
    pub pc_type: Option<String>,
    pub config: BTreeMap<String, Value>,
    pub complex_support: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub setup_mode: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fallback_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub residual_reduction_per_time: Option<f64>,
    /// Nested KSP diagnostics when `pc_type = Ksp`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nested_ksp: Option<Box<KspDiagnostics>>,
}

impl PcDiagnostics {
    pub fn from_options(pc_type: Option<PcType>, opts: Option<&PcOptions>) -> Self {
        let mut config = BTreeMap::new();
        let nested_ksp = opts.and_then(|opts| match pc_type {
            Some(PcType::Ksp) => build_nested_ksp_diagnostics(opts).map(Box::new),
            _ => None,
        });
        if let Some(opts) = opts {
            insert_opt(&mut config, "ilu_level", opts.ilu_level);
            insert_opt(&mut config, "ilut_drop_tol", opts.ilut_drop_tol);
            insert_opt(&mut config, "ilut_max_fill", opts.ilut_max_fill);
            insert_opt(&mut config, "ilut_perm_tol", opts.ilut_perm_tol);
            insert_opt(&mut config, "ilutp_max_fill", opts.ilutp_max_fill);
            insert_opt(&mut config, "ilutp_drop_tol", opts.ilutp_drop_tol);
            insert_opt(&mut config, "ilutp_perm_tol", opts.ilutp_perm_tol);
            insert_opt(&mut config, "asm_overlap", opts.asm_overlap);
            insert_opt(&mut config, "asm_mode", opts.asm_mode.clone());
            insert_opt(
                &mut config,
                "pc_dist_local_apply",
                opts.pc_dist_local_apply.clone(),
            );
            insert_opt(&mut config, "pc_dist_route", opts.pc_dist_route.clone());
            insert_opt(&mut config, "asm_weighting", opts.asm_weighting.clone());
            insert_opt(&mut config, "chebyshev_degree", opts.chebyshev_degree);
            insert_opt(
                &mut config,
                "chebyshev_lambda_min",
                opts.chebyshev_lambda_min,
            );
            insert_opt(
                &mut config,
                "chebyshev_lambda_max",
                opts.chebyshev_lambda_max,
            );
            insert_opt(&mut config, "amg_levels", opts.amg_levels);
            insert_opt(&mut config, "amg_cycle_type", opts.amg_cycle_type.clone());
            insert_opt(
                &mut config,
                "amg_coarse_solver",
                opts.amg_coarse_solver.clone(),
            );
            insert_opt(&mut config, "pc_chain", opts.pc_chain.clone());
            insert_opt(
                &mut config,
                "pc_fieldsplit_type",
                opts.pc_fieldsplit_type.clone(),
            );
            insert_opt(
                &mut config,
                "pc_fieldsplit_schur_fact_type",
                opts.pc_fieldsplit_schur_fact_type.clone(),
            );
            insert_opt(
                &mut config,
                "pc_fieldsplit_schur_precondition",
                opts.pc_fieldsplit_schur_precondition.clone(),
            );
            insert_opt(
                &mut config,
                "pc_bddc_coarse_ksp_type",
                opts.pc_bddc_coarse_ksp_type.clone(),
            );
            insert_opt(
                &mut config,
                "pc_bddc_coarse_pc_type",
                opts.pc_bddc_coarse_pc_type.clone(),
            );
            insert_opt(
                &mut config,
                "pc_bddc_use_vertices",
                opts.pc_bddc_use_vertices,
            );
            insert_opt(
                &mut config,
                "pc_bddc_constraint_selection",
                opts.pc_bddc_constraint_selection.clone(),
            );
            insert_opt(&mut config, "pc_bddc_scaling", opts.pc_bddc_scaling.clone());
        }

        let complex_support = pc_complex_support(pc_type).as_str().to_string();

        Self {
            pc_type: pc_type.map(|pct| format!("{pct:?}")),
            config,
            complex_support,
            setup_mode: None,
            fallback_reason: None,
            residual_reduction_per_time: None,
            nested_ksp,
        }
    }

    pub fn to_json_pretty(&self) -> String {
        serde_json::to_string_pretty(self)
            .unwrap_or_else(|_| "{\"pc_type\":null,\"config\":{}}".to_string())
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct PcViewDiagnostics {
    pub pc: Option<Box<PcDiagnostics>>,
    pub pc_chain: Option<Vec<PcDiagnostics>>,
}

impl PcViewDiagnostics {
    pub fn to_json_pretty(&self) -> String {
        serde_json::to_string_pretty(self)
            .unwrap_or_else(|_| "{\"pc\":null,\"pc_chain\":null}".to_string())
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct KspDiagnostics {
    pub solver_type: Option<String>,
    pub solver_config: BTreeMap<String, Value>,
    pub pc: Option<Box<PcDiagnostics>>,
    pub pc_chain: Option<Vec<PcDiagnostics>>,
    pub setup_called: bool,
    pub bound_comm_id: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_converged_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_converged_reason_petsc: Option<String>,
    pub reason_counters_breakdown: usize,
    pub reason_counters_nan: usize,
    pub reason_counters_inf: usize,
    pub reason_counters_pc_setup: usize,
    pub reason_counters_pc_apply: usize,
}

impl KspDiagnostics {
    pub fn pc_view(&self) -> PcViewDiagnostics {
        PcViewDiagnostics {
            pc: self.pc.clone(),
            pc_chain: self.pc_chain.clone(),
        }
    }

    pub fn to_json_pretty(&self) -> String {
        serde_json::to_string_pretty(self)
            .unwrap_or_else(|_| "{\"solver_type\":null,\"solver_config\":{}}".to_string())
    }
}

fn insert_opt<T: Serialize>(map: &mut BTreeMap<String, Value>, key: &str, val: Option<T>) {
    if let Some(value) = val {
        if let Ok(value) = serde_json::to_value(value) {
            map.insert(key.to_string(), value);
        }
    }
}

fn build_nested_ksp_diagnostics(opts: &PcOptions) -> Option<KspDiagnostics> {
    let mut ksp_opts = opts.pc_ksp_ksp_options.clone().unwrap_or_default();
    if let Some(v) = opts.pc_ksp_ksp_type.clone() {
        ksp_opts.ksp_type = Some(v);
    }
    if let Some(v) = opts.pc_ksp_maxits {
        ksp_opts.maxits = Some(v);
    }
    if let Some(v) = opts.pc_ksp_rtol {
        ksp_opts.rtol = Some(v);
    }
    if let Some(v) = opts.pc_ksp_pc_side.clone() {
        ksp_opts.pc_side = Some(v);
    }
    if ksp_opts.ksp_type.is_none() {
        ksp_opts.ksp_type = Some("richardson".to_string());
    }
    ksp_opts.ksp_view = Some(false);

    let mut pc_opts = opts
        .pc_ksp_pc_options
        .as_ref()
        .map(|b| b.as_ref().clone())
        .unwrap_or_default();
    if let Some(v) = opts.pc_ksp_pc_type.clone() {
        pc_opts.pc_type = Some(v);
    }
    if pc_opts.pc_type.is_none() {
        pc_opts.pc_type = Some("jacobi".to_string());
    }
    pc_opts.pc_view = Some(false);

    let mut ksp = KspContext::new();
    ksp.set_from_all_options(&ksp_opts, &pc_opts).ok()?;
    Some(ksp.view())
}

#[cfg(test)]
mod tests {
    use crate::config::options::{KspOptions, PcOptions};
    use crate::context::ksp_context::KspContext;

    #[test]
    fn diagnostics_include_solver_pc_and_key_fields() {
        let mut ksp_opts = KspOptions::default();
        ksp_opts.ksp_type = Some("gmres".into());
        ksp_opts.rtol = Some(1e-7);
        ksp_opts.maxits = Some(222);

        let mut pc_opts = PcOptions::default();
        pc_opts.pc_type = Some("ilu0".into());
        pc_opts.ilu_level = Some(2);

        let mut ksp = KspContext::new();
        ksp.set_from_all_options(&ksp_opts, &pc_opts).unwrap();

        let view = ksp.view();
        let json = view.to_json_pretty();

        assert!(json.contains("\"solver_type\""));
        assert!(json.contains("Gmres"));
        assert!(json.contains("\"pc_type\""));
        assert!(json.contains("Ilu0"));
        assert!(json.contains("\"rtol\""));
        assert!(json.contains("\"maxits\""));
        assert!(json.contains("\"ilu_level\""));
    }
}
