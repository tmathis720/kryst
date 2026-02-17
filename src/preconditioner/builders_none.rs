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
        PcConfig::Shell {
            name,
            apply_transpose,
            apply_symmetric,
            setup,
            destroy,
            context,
        } => Ok(Some(Box::new(crate::preconditioner::shell::ShellPc::new(
            name.clone(),
            apply_transpose.clone(),
            apply_symmetric.clone(),
            setup.clone(),
            destroy.clone(),
            context.clone(),
        )))),
        PcConfig::Ksp {
            inner_ksp_type,
            inner_pc_type,
            maxits,
            rtol,
            ksp_options,
            pc_options,
        } => {
            let mut opts = pc_options.clone().unwrap_or_default();
            if opts.pc_type.is_none() {
                opts.pc_type = inner_pc_type.clone();
            }
            let pc = crate::preconditioner::ksp_pc::KspAsPc::new(
                inner_pc_type.clone(),
                inner_ksp_type.clone(),
                *maxits,
                *rtol,
                ksp_options.clone(),
                opts,
            )?;
            Ok(Some(Box::new(pc)))
        }
        PcConfig::Mg {
            levels,
            cycle_type,
            smoother,
            smoother_steps,
            coarsen_type,
            interpolation_type,
            restriction_type,
            coarse_pc_type,
            coarse_ksp_type,
            coarse_ksp_maxits,
            coarse_ksp_rtol,
            level_policies,
        } => {
            let mut mg = crate::preconditioner::mg::MgPc::new(
                *levels,
                cycle_type.clone().map(|v| v.to_lowercase()),
                smoother.clone().map(|v| v.to_lowercase()),
                *smoother_steps,
                coarsen_type.clone().map(|v| v.to_lowercase()),
                interpolation_type.clone().map(|v| v.to_lowercase()),
                restriction_type.clone().map(|v| v.to_lowercase()),
                coarse_pc_type.clone().map(|v| v.to_lowercase()),
                coarse_ksp_type.clone().map(|v| v.to_lowercase()),
                *coarse_ksp_maxits,
                *coarse_ksp_rtol,
            );
            mg.set_level_policies(level_policies.clone());
            Ok(Some(Box::new(mg)))
        }
        PcConfig::Bddc {
            coarse_ksp_type,
            coarse_pc_type,
            use_vertices,
        } => Ok(Some(crate::preconditioner::builders::build_bddc(
            crate::preconditioner::bddc::BddcConfig {
                coarse_ksp_type: coarse_ksp_type.clone(),
                coarse_pc_type: coarse_pc_type.clone(),
                use_vertices: *use_vertices,
            },
        )?)),
        PcConfig::Gamg { .. } => Ok(None),
        _ => Ok(None),
    }
}
