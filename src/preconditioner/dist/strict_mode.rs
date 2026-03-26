use crate::error::KError;
use crate::matrix::DistCsrOp;
use crate::matrix::op::LinOp;
use crate::parallel::Comm;
use crate::preconditioner::asm::{AsmBlockSolver, AsmInnerPc};

use super::{DistLocalApplyMode, DistPcBuilder, GlobalPcKind};

fn strict_mode_error(global_pc: GlobalPcKind, detail_key: &str, detail: &str) -> KError {
    KError::InvalidInput(format!(
        "err_key=pc_dist_strict_mode_rejected pc_dist_local_apply=strict pc_global={global_pc:?} detail_key={detail_key} detail={detail}"
    ))
}

fn asm_local_solver_supports_native(block_solver: AsmBlockSolver, inner_pc: AsmInnerPc) -> bool {
    match (block_solver, inner_pc) {
        (AsmBlockSolver::Csr, AsmInnerPc::Jacobi)
        | (AsmBlockSolver::LuDense, AsmInnerPc::Jacobi)
        | (AsmBlockSolver::Csr, AsmInnerPc::Ilu0)
        | (AsmBlockSolver::Csr, AsmInnerPc::Ilut { .. })
        | (AsmBlockSolver::LuDense, AsmInnerPc::Ilutp { .. }) => true,
        (AsmBlockSolver::LuDense, AsmInnerPc::Ilu0)
        | (AsmBlockSolver::LuDense, AsmInnerPc::Ilut { .. })
        | (AsmBlockSolver::Csr, AsmInnerPc::Ilutp { .. }) => false,
    }
}

fn validate_asm_like_strict_mode(
    dist_op: &DistCsrOp,
    global_pc: GlobalPcKind,
    overlap: usize,
    block_solver: AsmBlockSolver,
    inner_pc: AsmInnerPc,
    local_apply_mode: DistLocalApplyMode,
) -> Result<(), KError> {
    if !local_apply_mode.requires_native() {
        return Ok(());
    }
    if overlap == 0 {
        return Err(strict_mode_error(
            global_pc,
            "overlap_mode",
            "strict distributed local-apply requires overlap>0 for ASM/RAS native kernels",
        ));
    }
    if !asm_local_solver_supports_native(block_solver, inner_pc) {
        return Err(strict_mode_error(
            global_pc,
            "local_solver_support",
            "strict distributed local-apply requires a supported ASM/RAS block_solver + inner_pc pair",
        ));
    }
    if dist_op.comm().size() <= 1 {
        return Err(strict_mode_error(
            global_pc,
            "communication_plan_constraints",
            "strict distributed local-apply requires MPI communicator size>1 for ASM/RAS communication plans",
        ));
    }
    Ok(())
}

pub fn validate_dist_builder_strict_mode(
    dist_op: &DistCsrOp,
    builder: &DistPcBuilder,
) -> Result<(), KError> {
    match builder {
        DistPcBuilder::BlockJacobi { opts } => {
            if !opts.local_apply_mode.requires_native() {
                return Ok(());
            }
            if !opts.local_pc.build_capabilities().native_local_apply {
                Err(strict_mode_error(
                    GlobalPcKind::BlockJacobi,
                    "unsupported_local_pc",
                    "block-jacobi strict mode requires native local kernel; local pc supports wrapped_local only",
                ))
            } else {
                Ok(())
            }
        }
        DistPcBuilder::Asm {
            overlap,
            block_solver,
            inner_pc,
            local_apply_mode,
            ..
        } => validate_asm_like_strict_mode(
            dist_op,
            GlobalPcKind::Asm,
            *overlap,
            *block_solver,
            *inner_pc,
            *local_apply_mode,
        ),
        DistPcBuilder::Ras {
            overlap,
            block_solver,
            inner_pc,
            local_apply_mode,
            ..
        } => validate_asm_like_strict_mode(
            dist_op,
            GlobalPcKind::Ras,
            *overlap,
            *block_solver,
            *inner_pc,
            *local_apply_mode,
        ),
    }
}
