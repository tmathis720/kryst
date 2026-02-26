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
            apply_conjugate_transpose,
            apply_symmetric,
            apply_symmetric_left,
            apply_symmetric_right,
            setup,
            destroy,
            context,
        } => Ok(Some(Box::new(crate::preconditioner::shell::ShellPc::new(
            name.clone(),
            apply_transpose.clone(),
            apply_conjugate_transpose.clone(),
            apply_symmetric.clone(),
            apply_symmetric_left.clone(),
            apply_symmetric_right.clone(),
            setup.clone(),
            destroy.clone(),
            context.clone(),
        )))),
        PcConfig::Ksp {
            ksp_options,
            pc_options,
        } => {
            let pc = crate::preconditioner::ksp_pc::KspAsPc::new(
                ksp_options.clone(),
                pc_options.clone(),
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
