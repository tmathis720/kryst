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
            $stats.name, $stats.n, $stats.build_ms, $stats.nnz_pc, $stats.fill_ratio
        );
    };
}

