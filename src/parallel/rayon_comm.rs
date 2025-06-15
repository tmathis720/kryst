// rayon-based parallel communication

use rayon::prelude::*;

pub struct RayonComm;

impl RayonComm {
    pub fn new() -> Self {
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_cpus::get())
            .build_global()
            .ok();
        RayonComm
    }
}

impl super::Comm for RayonComm {
    type Vec = Vec<f64>;
    fn rank(&self) -> usize { 0 }
    fn size(&self) -> usize { num_cpus::get() }
    fn barrier(&self) { rayon::scope(|_| {}); }
    fn scatter<T: Clone>(&self, global: &[T], out: &mut [T], root: usize) {
        let n = out.len();
        let start = root * n;
        out.clone_from_slice(&global[start..start + n]);
    }
    fn gather<T: Clone>(&self, local: &[T], out: &mut Vec<T>, _root: usize) {
        out.clear();
        out.extend_from_slice(local);
    }
    fn all_reduce(&self, x: f64) -> f64 {
        x // No-op for shared memory
    }
    fn parallel_mat_vec(&self, a: &faer::Mat<f64>, x: &[f64], y: &mut [f64]) {
        assert_eq!(a.ncols(), x.len());
        assert_eq!(a.nrows(), y.len());
        y.par_iter_mut().enumerate().for_each(|(i, yi)| {
            *yi = (0..a.ncols()).map(|j| a[(i, j)] * x[j]).sum();
        });
    }
}
