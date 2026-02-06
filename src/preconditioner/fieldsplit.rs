use crate::algebra::scalar::{KrystScalar, S};
use crate::config::options::PcOptions;
use crate::context::pc_context::{PcFactory, PcType};
use crate::error::KError;
use crate::matrix::op::LinOp;
use crate::preconditioner::{PcSide, Preconditioner};
use std::str::FromStr;

pub struct FieldSplitPc {
    block_sizes: Vec<usize>,
    children: Vec<Box<dyn Preconditioner>>,
}

impl FieldSplitPc {
    pub fn new(
        block_sizes: Vec<usize>,
        child_pc_type: Option<String>,
        opts: PcOptions,
    ) -> Result<Self, KError> {
        let child_type = child_pc_type
            .as_deref()
            .map(PcType::from_str)
            .transpose()?
            .unwrap_or(PcType::Jacobi);
        let mut children = Vec::with_capacity(block_sizes.len());
        let prefixes = opts.pc_fieldsplit_prefixes.clone().unwrap_or_default();
        for (i, _) in block_sizes.iter().enumerate() {
            let mut child_opts = opts.clone();
            if let Some(prefix) = prefixes.get(i)
                && let Some(scoped) = opts.scoped_child(prefix)
            {
                child_opts.overlay_from(scoped.clone());
            }
            children.push(PcFactory::create_preconditioner(
                child_type,
                Some(&child_opts),
            )?);
        }
        Ok(Self {
            block_sizes,
            children,
        })
    }
}

impl Preconditioner for FieldSplitPc {
    fn setup(&mut self, a: &dyn LinOp<S = S>) -> Result<(), KError> {
        let n = a.dims().0;
        if self.block_sizes.iter().sum::<usize>() != n {
            return Err(KError::InvalidInput(format!(
                "pc_fieldsplit_block_sizes must sum to matrix size ({n})"
            )));
        }
        for child in &mut self.children {
            child.setup(a)?;
        }
        Ok(())
    }

    fn apply(&self, side: PcSide, x: &[S], y: &mut [S]) -> Result<(), KError> {
        if x.len() != y.len() {
            return Err(KError::InvalidInput(
                "fieldsplit input/output length mismatch".into(),
            ));
        }
        y.fill(S::zero());
        let n = x.len();
        let mut off = 0usize;
        for (blk, child) in self.block_sizes.iter().zip(self.children.iter()) {
            let end = off + *blk;
            let mut xin = vec![S::zero(); n];
            xin[off..end].copy_from_slice(&x[off..end]);
            let mut zout = vec![S::zero(); n];
            child.apply(side, &xin, &mut zout)?;
            y[off..end].copy_from_slice(&zout[off..end]);
            off = end;
        }
        Ok(())
    }
}
