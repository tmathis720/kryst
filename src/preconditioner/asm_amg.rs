use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

#[allow(unused_imports)]
use crate::algebra::blas::{dot_conj, nrm2};
#[allow(unused_imports)]
use crate::algebra::prelude::*;
use crate::error::KError;
use crate::matrix::op::LinOp;
use crate::preconditioner::amg::{AMGConfig, CycleType};
use crate::preconditioner::asm::{Asm, AsmConfig};
use crate::preconditioner::{PcCaps, PcDistributedSupport, PcSide, Preconditioner};

#[cfg(feature = "complex")]
use crate::algebra::bridge::BridgeScratch;
#[cfg(feature = "complex")]
use crate::preconditioner::pc_bridge::{apply_pc_mut_s, apply_pc_s};

use super::amg::AMG;

/// Two-level combination strategy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TwoLevelMode {
    /// Apply coarse correction additively after local ASM smoothing.
    AdditiveCoarse,
    /// Apply coarse grid first and sweep ASM on the remaining residual.
    MultiplicativeCoarse,
}

/// Configuration for the hybrid ASM + AMG preconditioner.
#[derive(Clone, Debug)]
pub struct TwoLevelConfig {
    pub mode: TwoLevelMode,
    pub coarse_every: usize,
    pub amg_cfg: AMGConfig,
    pub coarse_cycle: CycleType,
}

impl Default for TwoLevelConfig {
    fn default() -> Self {
        Self {
            mode: TwoLevelMode::AdditiveCoarse,
            coarse_every: 1,
            amg_cfg: AMGConfig::default(),
            coarse_cycle: CycleType::V,
        }
    }
}

/// Builder for [`AsmAmg`].
#[derive(Clone, Debug)]
pub struct AsmAmgBuilder {
    asm_cfg: AsmConfig,
    two_cfg: TwoLevelConfig,
}

impl AsmAmgBuilder {
    pub fn new() -> Self {
        Self {
            asm_cfg: AsmConfig::default(),
            two_cfg: TwoLevelConfig::default(),
        }
    }

    pub fn asm_config(mut self, cfg: AsmConfig) -> Self {
        self.asm_cfg = cfg;
        self
    }

    pub fn two_level_config(mut self, cfg: TwoLevelConfig) -> Self {
        self.two_cfg = cfg;
        self
    }

    pub fn build(self) -> AsmAmg {
        AsmAmg::with_configs(self.asm_cfg, self.two_cfg)
    }
}

impl Default for AsmAmgBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Two-level hybrid Additive Schwarz + AMG coarse correction.
pub struct AsmAmg {
    asm: Asm,
    amg: super::amg::AMG,
    cfg: TwoLevelConfig,
    tmp_local: Mutex<Vec<R>>,
    tmp_residual: Mutex<Vec<R>>,
    tmp_coarse: Mutex<Vec<R>>,
    apply_count: AtomicUsize,
}

impl AsmAmg {
    pub fn builder() -> AsmAmgBuilder {
        AsmAmgBuilder::new()
    }

    pub fn with_configs(asm_cfg: AsmConfig, mut two_cfg: TwoLevelConfig) -> Self {
        two_cfg.amg_cfg.cycle_type = two_cfg.coarse_cycle;
        let amg = AMG::with_config(two_cfg.amg_cfg.clone());
        let asm = Asm::with_config(asm_cfg);
        Self {
            asm,
            amg,
            cfg: two_cfg,
            tmp_local: Mutex::new(Vec::new()),
            tmp_residual: Mutex::new(Vec::new()),
            tmp_coarse: Mutex::new(Vec::new()),
            apply_count: AtomicUsize::new(0),
        }
    }

    fn ensure_workspace(&self, n: usize) {
        {
            let mut buf = self.tmp_local.lock().unwrap();
            if buf.len() != n {
                buf.resize(n, R::zero());
            }
        }
        {
            let mut buf = self.tmp_residual.lock().unwrap();
            if buf.len() != n {
                buf.resize(n, R::zero());
            }
        }
        {
            let mut buf = self.tmp_coarse.lock().unwrap();
            if buf.len() != n {
                buf.resize(n, R::zero());
            }
        }
    }
}

#[cfg(not(feature = "complex"))]
impl Preconditioner for AsmAmg {
    fn setup(&mut self, op: &dyn LinOp<S = f64>) -> Result<(), KError> {
        self.asm.setup(op)?;
        self.amg.setup(op)?;
        let n = self
            .asm
            .dimension()
            .ok_or_else(|| KError::InvalidInput("ASM setup failed".into()))?;
        self.ensure_workspace(n);
        self.apply_count.store(0, Ordering::Relaxed);
        Ok(())
    }

    fn required_format(&self) -> crate::matrix::format::OpFormat {
        crate::matrix::format::OpFormat::Csr
    }

    fn apply(&self, side: PcSide, rhs: &[f64], out: &mut [f64]) -> Result<(), KError> {
        if rhs.len() != out.len() {
            return Err(KError::InvalidInput("AsmAmg apply length mismatch".into()));
        }
        let n = rhs.len();
        self.ensure_workspace(n);
        let mut q_local = self.tmp_local.lock().unwrap();
        let mut residual = self.tmp_residual.lock().unwrap();
        let mut q_coarse = self.tmp_coarse.lock().unwrap();
        let coarse_every = self.cfg.coarse_every.max(1);
        let iter = self.apply_count.fetch_add(1, Ordering::Relaxed) + 1;

        if coarse_every > 1 && iter % coarse_every != 0 {
            self.asm.apply(PcSide::Left, rhs, &mut q_local)?;
            out.copy_from_slice(&q_local);
            return Ok(());
        }

        let a = self
            .asm
            .matrix()
            .ok_or_else(|| KError::InvalidInput("ASM matrix unavailable".into()))?;

        match self.cfg.mode {
            TwoLevelMode::AdditiveCoarse => {
                self.asm.apply(PcSide::Left, rhs, &mut q_local)?;
                a.spmv_scaled(1.0, &q_local, 0.0, &mut residual)?;
                for i in 0..n {
                    residual[i] = rhs[i] - residual[i];
                    q_coarse[i] = R::zero();
                }
                self.amg.apply(side, &residual, &mut q_coarse)?;
                for i in 0..n {
                    out[i] = q_local[i] + q_coarse[i];
                }
            }
            TwoLevelMode::MultiplicativeCoarse => {
                for qi in q_coarse.iter_mut() {
                    *qi = R::zero();
                }
                self.amg.apply(side, rhs, &mut q_coarse)?;
                a.spmv_scaled(1.0, &q_coarse, 0.0, &mut residual)?;
                for i in 0..n {
                    residual[i] = rhs[i] - residual[i];
                }
                self.asm.apply(PcSide::Left, &residual, &mut q_local)?;
                for i in 0..n {
                    out[i] = q_local[i] + q_coarse[i];
                }
            }
        }
        Ok(())
    }

    fn supports_numeric_update(&self) -> bool {
        true
    }

    fn update_numeric(&mut self, op: &dyn LinOp<S = f64>) -> Result<(), KError> {
        self.asm.update_numeric(op)?;
        self.amg.update_numeric(op)
    }

    fn update_symbolic(&mut self, op: &dyn LinOp<S = f64>) -> Result<(), KError> {
        self.asm.update_symbolic(op)?;
        self.amg.update_symbolic(op)
    }

    fn capabilities(&self) -> PcCaps {
        PcCaps::default()
    }

    fn distributed_support(&self) -> PcDistributedSupport {
        PcDistributedSupport::Distributed
    }
}

#[cfg(feature = "complex")]
impl Preconditioner for AsmAmg {
    fn setup(&mut self, _op: &dyn LinOp<S = S>) -> Result<(), KError> {
        Err(KError::Unsupported(
            "AsmAmg does not support complex scalars yet".into(),
        ))
    }

    fn apply(&self, _side: PcSide, _rhs: &[S], _out: &mut [S]) -> Result<(), KError> {
        Err(KError::Unsupported(
            "AsmAmg does not support complex scalars yet".into(),
        ))
    }

    fn distributed_support(&self) -> PcDistributedSupport {
        PcDistributedSupport::Distributed
    }
}

#[cfg(feature = "complex")]
impl crate::ops::kpc::KPreconditioner for AsmAmg {
    type Scalar = S;

    #[inline]
    fn dims(&self) -> (usize, usize) {
        self.asm
            .dimension()
            .map(|n| (n, n))
            .unwrap_or_else(|| crate::ops::kpc::KPreconditioner::dims(&self.amg))
    }

    fn apply_s(
        &self,
        side: PcSide,
        x: &[S],
        y: &mut [S],
        scratch: &mut BridgeScratch,
    ) -> Result<(), KError> {
        apply_pc_s(self, side, x, y, scratch)
    }

    fn apply_mut_s(
        &mut self,
        side: PcSide,
        x: &[S],
        y: &mut [S],
        scratch: &mut BridgeScratch,
    ) -> Result<(), KError> {
        apply_pc_mut_s(self, side, x, y, scratch)
    }
}
