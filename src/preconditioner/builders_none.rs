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
            options,
        } => {
            let pc = crate::preconditioner::fieldsplit::FieldSplitPc::new(
                block_sizes.clone(),
                child_pc_type.clone(),
                options.clone(),
            )?;
            Ok(Some(Box::new(pc)))
        }
        PcConfig::Shell { name } => Ok(Some(Box::new(crate::preconditioner::shell::ShellPc::new(
            name.clone(),
        )))),
        PcConfig::Ksp {
            inner_ksp_type,
            inner_pc_type,
            maxits,
            rtol,
        } => {
            let mut opts = crate::config::options::PcOptions::default();
            opts.pc_type = inner_pc_type.clone();
            let pc = crate::preconditioner::ksp_pc::KspAsPc::new(
                inner_pc_type.clone(),
                inner_ksp_type.clone(),
                *maxits,
                *rtol,
                opts,
            )?;
            Ok(Some(Box::new(pc)))
        }
        PcConfig::Mg {
            levels,
            cycle_type,
            smoother,
            smoother_steps,
        } => Ok(Some(Box::new(crate::preconditioner::mg::MgPc::new(
            *levels,
            cycle_type.clone(),
            smoother.clone(),
            *smoother_steps,
        )))),
        PcConfig::Bddc { .. } => Err(KError::Unsupported(
            "BDDC is not yet implemented; use ASM/FieldSplit for now",
        )),
        PcConfig::Gamg { .. } => Err(KError::Unsupported(
            "GAMG is not yet implemented; use pc_type=amg for now",
        )),
        _ => Ok(None),
    }
}
