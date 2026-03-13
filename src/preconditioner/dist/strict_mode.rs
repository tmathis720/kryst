use crate::error::KError;

use super::{DistLocalApplyMode, DistPcBuilder, GlobalPcKind, LocalPcKind};

fn strict_mode_error(global_pc: GlobalPcKind, detail_key: &str, detail: &str) -> KError {
    KError::InvalidInput(format!(
        "err_key=pc_dist_strict_mode_rejected pc_dist_local_apply=strict pc_global={global_pc:?} detail_key={detail_key} detail={detail}"
    ))
}

pub fn validate_dist_builder_strict_mode(builder: &DistPcBuilder) -> Result<(), KError> {
    match builder {
        DistPcBuilder::BlockJacobi { opts } => {
            if !opts.local_apply_mode.requires_native() {
                return Ok(());
            }
            match opts.local_pc {
                LocalPcKind::Fsai | LocalPcKind::Spai => Err(strict_mode_error(
                    GlobalPcKind::BlockJacobi,
                    "unsupported_local_pc",
                    "block-jacobi strict mode requires native local kernel; local pc supports wrapped_local only",
                )),
                _ => Ok(()),
            }
        }
        DistPcBuilder::Asm {
            local_apply_mode, ..
        }
        | DistPcBuilder::Ras {
            local_apply_mode, ..
        } => {
            if matches!(local_apply_mode, DistLocalApplyMode::NativeStrict) {
                let global = match builder {
                    DistPcBuilder::Asm { .. } => GlobalPcKind::Asm,
                    DistPcBuilder::Ras { .. } => GlobalPcKind::Ras,
                    DistPcBuilder::BlockJacobi { .. } => unreachable!(),
                };
                return Err(strict_mode_error(
                    global,
                    "unsupported_global_pc",
                    "strict distributed local-apply mode is currently unsupported for ASM/RAS builders",
                ));
            }
            Ok(())
        }
    }
}
