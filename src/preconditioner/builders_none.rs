use crate::context::pc_context::NoOpPreconditioner;
use crate::context::pc_context::PcConfig;
use crate::error::KError;
use crate::preconditioner::Preconditioner;

pub fn try_build(cfg: &PcConfig) -> Result<Option<Box<dyn Preconditioner>>, KError> {
    match cfg {
        PcConfig::None => Ok(Some(Box::new(NoOpPreconditioner))),
        PcConfig::FieldSplit {
            block_sizes,
            child_pc_type,
        } => {
            let opts = crate::config::options::PcOptions::default();
            let pc = crate::preconditioner::fieldsplit::FieldSplitPc::new(
                block_sizes.clone(),
                child_pc_type.clone(),
                opts,
            )?;
            Ok(Some(Box::new(pc)))
        }
        PcConfig::Shell { name } => Ok(Some(Box::new(crate::preconditioner::shell::ShellPc::new(
            name.clone(),
        )))),
        PcConfig::Ksp {
            inner_ksp_type: _,
            inner_pc_type,
            maxits,
            rtol,
        } => {
            let mut opts = crate::config::options::PcOptions::default();
            opts.pc_type = inner_pc_type.clone();
            let pc = crate::preconditioner::ksp_pc::KspAsPc::new(
                inner_pc_type.clone(),
                *maxits,
                *rtol,
                opts,
            )?;
            Ok(Some(Box::new(pc)))
        }
        PcConfig::Mg { levels, cycle_type } => Ok(Some(Box::new(
            crate::preconditioner::mg::MgPc::new(*levels, cycle_type.clone()),
        ))),
        PcConfig::Bddc { .. } => Err(KError::Unsupported(
            "BDDC is not yet implemented; use ASM/FieldSplit for now",
        )),
        _ => Ok(None),
    }
}
