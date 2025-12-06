#[derive(Clone, Default, Debug)]
pub struct PcStats {
    pub name: &'static str,
    pub n: usize,
    pub build_ms: f64,
    pub nnz_pc: usize,
    pub fill_ratio: f64,
    pub applies: u64,
}

pub trait PcIntrospect {
    fn stats(&self) -> PcStats;
    fn enable_logging(&mut self, _on: bool) {}
}

#[cfg(feature = "logging")]
#[macro_export]
macro_rules! pc_log {
    ($stats:expr) => {
        log::info!(
            "PC {}: n={}, build_ms={:.3}, nnz_pc={}, fill={:.3}",
            $stats.name,
            $stats.n,
            $stats.build_ms,
            $stats.nnz_pc,
            $stats.fill_ratio
        );
    };
}

#[derive(Debug, Clone)]
pub struct ParIluIterSample {
    pub iter: u32,
    pub residual: f64,
}

#[derive(Debug, Clone)]
pub struct ParIluHistory {
    pub used: u32,
    pub buf: Vec<ParIluIterSample>,
}

impl ParIluHistory {
    pub fn with_capacity(cap: usize) -> Self {
        let buf = Vec::with_capacity(cap);
        Self { used: 0, buf }
    }

    #[inline]
    pub fn push(&mut self, s: ParIluIterSample) {
        self.buf.push(s);
        self.used += 1;
    }

    pub fn as_slice(&self) -> &[ParIluIterSample] {
        &self.buf[..self.used as usize]
    }
}
