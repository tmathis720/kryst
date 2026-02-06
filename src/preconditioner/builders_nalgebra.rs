#![cfg(feature = "backend-nalgebra")]

use crate::context::pc_context::PcConfig;
use crate::error::KError;
use crate::preconditioner::Preconditioner;

pub fn try_build(cfg: &PcConfig) -> Result<Option<Box<dyn Preconditioner>>, KError> {
    match cfg {
        PcConfig::Lu => Ok(Some(Box::new(
            crate::preconditioner::nalgebra_direct::NalgebraLuPc::new(),
        ))),
        PcConfig::Qr => Ok(Some(Box::new(
            crate::preconditioner::nalgebra_direct::NalgebraQrPc::new(),
        ))),
        PcConfig::FieldSplit { .. }
        | PcConfig::Shell { .. }
        | PcConfig::Ksp { .. }
        | PcConfig::Mg { .. }
        | PcConfig::Bddc { .. } => Ok(None),
        _ => Ok(None),
    }
}
