//! Generic fallback ladder primitives and attempt/result reporting helpers.

use crate::utils::convergence::{AcceptanceStatus, ConvergedReason};

#[derive(Clone, Debug)]
pub struct FallbackStep {
    pub solver: String,
    pub pc: String,
    pub rung: usize,
    pub note: String,
}

#[derive(Clone, Debug)]
pub struct AttemptRecord {
    pub rung_id: usize,
    pub rung_label: String,
    pub solver: String,
    pub preconditioner: String,
    pub preprocessing_profile: String,
    pub iterations: usize,
    pub true_abs_residual: f64,
    pub true_rel_residual: f64,
    pub solver_reported_residual: f64,
    pub monitor_residual: f64,
    pub ls_residual: f64,
    pub gmres_classical_retry: bool,
    pub solver_reported_status: String,
    pub acceptance_status: String,
    pub acceptance_reason: String,
    pub elapsed_seconds: f64,
    pub accepted: bool,
}

#[derive(Clone, Debug)]
pub struct TruthReference<TComparison> {
    pub selected_as_winner: bool,
    pub reference_solve_executed: bool,
    pub elapsed_seconds: Option<f64>,
    pub true_abs_residual: Option<f64>,
    pub true_rel_residual: Option<f64>,
    pub comparison: Option<TComparison>,
    pub note: String,
}

#[derive(Clone, Debug)]
pub struct SolverTestResult<TComparison> {
    pub primary_method: String,
    pub chosen_method: String,
    pub solver_reason: String,
    pub outcome_code: String,
    pub diagnostics: String,
    pub converged: bool,
    pub attempts: Vec<AttemptRecord>,
    pub truth_reference: Option<TruthReference<TComparison>>,
}

impl<TComparison> SolverTestResult<TComparison> {
    pub fn baseline_attempt(&self) -> Option<&AttemptRecord> {
        self.attempts.iter().find(|attempt| attempt.rung_id == 0)
    }

    pub fn best_accepted_iterative(&self) -> Option<&AttemptRecord> {
        self.attempts
            .iter()
            .filter(|attempt| attempt.accepted)
            .min_by(|a, b| {
                a.iterations
                    .cmp(&b.iterations)
                    .then_with(|| a.true_rel_residual.total_cmp(&b.true_rel_residual))
            })
    }

    pub fn best_attempted_iterative(&self) -> Option<&AttemptRecord> {
        self.attempts.iter().min_by(|a, b| {
            a.true_rel_residual
                .total_cmp(&b.true_rel_residual)
                .then_with(|| a.true_abs_residual.total_cmp(&b.true_abs_residual))
                .then_with(|| a.elapsed_seconds.total_cmp(&b.elapsed_seconds))
                .then_with(|| a.iterations.cmp(&b.iterations))
        })
    }

    pub fn best_verified_attempt(&self) -> Option<&AttemptRecord> {
        if self
            .truth_reference
            .as_ref()
            .is_some_and(|truth| truth.selected_as_winner)
        {
            return None;
        }
        self.best_accepted_iterative()
    }

    pub fn policy_rung_fidelity(&self) -> &'static str {
        let baseline_ok = self
            .attempts
            .iter()
            .any(|attempt| attempt.rung_id == 0 && attempt.accepted);
        let rescue_ok = self
            .attempts
            .iter()
            .any(|attempt| attempt.rung_id > 0 && attempt.accepted);
        match (baseline_ok, rescue_ok) {
            (true, _) => "baseline_ok",
            (false, true) => "rescue_only",
            (false, false) => "none_ok",
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct AcceptanceContract {
    pub rtol: f64,
    pub atol: f64,
    pub slack: f64,
}

pub fn classify_failure(msg: &str) -> &'static str {
    let m = msg.to_ascii_lowercase();
    if m.contains("contract") || m.contains("cg rejected") || m.contains("wrong method") {
        "CONTRACT_MISMATCH"
    } else if m.contains("breakdown")
        || m.contains("nan")
        || m.contains("inf")
        || m.contains("zero pivot")
        || m.contains("singular")
    {
        "BREAKDOWN"
    } else if m.contains("stagnat")
        || m.contains("max iter")
        || m.contains("did not converge")
        || m.contains("no convergence")
    {
        "STAGNATED"
    } else {
        "FAILED"
    }
}

pub fn classify_acceptance(
    reason: ConvergedReason,
    true_abs_res: f64,
    true_rel_res: f64,
    contract: AcceptanceContract,
) -> AcceptanceStatus {
    let rtol_ok = true_rel_res.is_finite() && true_rel_res <= contract.rtol * contract.slack;
    let atol_ok = true_abs_res.is_finite() && true_abs_res <= contract.atol * contract.slack;
    let meets_any_contract = rtol_ok || atol_ok;

    match reason {
        ConvergedReason::ConvergedRtol => {
            if rtol_ok {
                AcceptanceStatus::Ok
            } else if meets_any_contract {
                AcceptanceStatus::OkWithWarning
            } else {
                AcceptanceStatus::ContractMismatch
            }
        }
        ConvergedReason::ConvergedAtol => {
            if atol_ok {
                AcceptanceStatus::Ok
            } else if meets_any_contract {
                AcceptanceStatus::OkWithWarning
            } else {
                AcceptanceStatus::ContractMismatch
            }
        }
        ConvergedReason::DivergedBreakdown
        | ConvergedReason::DivergedArnoldiRankLoss
        | ConvergedReason::DivergedBreakdownBiCG
        | ConvergedReason::DivergedReducedSystemSingular
        | ConvergedReason::DivergedNan
        | ConvergedReason::DivergedInf
        | ConvergedReason::DivergedIndefiniteMatrix
        | ConvergedReason::DivergedIndefinitePC => {
            if meets_any_contract {
                AcceptanceStatus::OkWithWarning
            } else {
                AcceptanceStatus::Breakdown
            }
        }
        ConvergedReason::ConvergedTrustRegion | ConvergedReason::ConvergedHappyBreakdown => {
            if meets_any_contract {
                AcceptanceStatus::OkWithWarning
            } else {
                AcceptanceStatus::ContractMismatch
            }
        }
        ConvergedReason::DivergedDtol
        | ConvergedReason::DivergedMaxIts
        | ConvergedReason::StoppedByMonitor => {
            if meets_any_contract {
                AcceptanceStatus::OkWithWarning
            } else {
                AcceptanceStatus::Stagnated
            }
        }
        ConvergedReason::DivergedPcSetupFailed
        | ConvergedReason::DivergedPcFailed
        | ConvergedReason::Continued => {
            if meets_any_contract {
                AcceptanceStatus::OkWithWarning
            } else {
                AcceptanceStatus::Failed
            }
        }
    }
}

pub fn solver_reason_code(
    reason: ConvergedReason,
    acceptance_status: AcceptanceStatus,
) -> &'static str {
    match acceptance_status {
        AcceptanceStatus::Ok | AcceptanceStatus::OkWithWarning => match reason {
            ConvergedReason::ConvergedRtol => "RTOL_OK",
            ConvergedReason::ConvergedAtol => "ATOL_OK",
            ConvergedReason::ConvergedTrustRegion => "TRUST_REGION_OK",
            ConvergedReason::ConvergedHappyBreakdown => "HAPPY_BREAKDOWN_OK",
            ConvergedReason::DivergedDtol
            | ConvergedReason::DivergedMaxIts
            | ConvergedReason::StoppedByMonitor => "CONTRACT_OK_WARN",
            _ => "CONTRACT_OK_WARN",
        },
        AcceptanceStatus::ContractMismatch => "CONTRACT_MISMATCH",
        AcceptanceStatus::Breakdown => "BREAKDOWN",
        AcceptanceStatus::Stagnated => "STAGNATED",
        AcceptanceStatus::Failed => "FAILED",
    }
}

pub fn render_attempt_chain(attempts: &[AttemptRecord]) -> String {
    attempts
        .iter()
        .map(|a| {
            format!(
                "{}:{}+{}:{}(iters={}, rel={:.2e}, time={:.2} ms)",
                a.rung_label,
                a.solver,
                a.preconditioner,
                a.acceptance_status,
                a.iterations,
                a.true_rel_residual,
                a.elapsed_seconds * 1_000.0
            )
        })
        .collect::<Vec<_>>()
        .join(" | ")
}

pub fn execute_fallback_ladder<T, E, F>(
    steps: &[FallbackStep],
    mut execute_step: F,
) -> Vec<(usize, Result<T, E>)>
where
    F: FnMut(&FallbackStep) -> Result<T, E>,
{
    let mut outcomes = Vec::with_capacity(steps.len());
    for step in steps {
        outcomes.push((step.rung, execute_step(step)));
    }
    outcomes
}
