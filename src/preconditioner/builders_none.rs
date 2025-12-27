use crate::context::pc_context::PcConfig;
use crate::context::pc_context::NoOpPreconditioner;
use crate::error::KError;
use crate::preconditioner::Preconditioner;

pub fn try_build(cfg: &PcConfig) -> Result<Option<Box<dyn Preconditioner>>, KError> {
    match cfg {
        PcConfig::None => Ok(Some(Box::new(NoOpPreconditioner))),
        _ => Ok(None),
    }
}
