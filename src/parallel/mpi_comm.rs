// MPI-based parallel communication (mpi)

#[cfg(feature = "mpi")]
use mpi::traits::*;
#[cfg(feature = "mpi")]
use mpi::topology::SimpleCommunicator;

#[cfg(feature = "mpi")]
pub struct MpiComm {
    pub world: SimpleCommunicator,
    pub rank: usize,
    pub size: usize,
}

#[cfg(feature = "mpi")]
impl MpiComm {
    pub fn new() -> Self {
        let universe = mpi::initialize().unwrap();
        let world    = universe.world();
        let rank     = world.rank() as usize;
        let size     = world.size() as usize;
        MpiComm { world, rank, size }
    }
}

#[cfg(feature = "mpi")]
impl super::Comm for MpiComm {
    type Vec = Vec<f64>;

    fn rank(&self) -> usize { self.rank }
    fn size(&self) -> usize { self.size }
    fn barrier(&self) { self.world.barrier(); }

    fn scatter<T: Clone + mpi::datatype::Equivalence>(
        &self,
        global: &[T],
        out: &mut [T],
        root: usize,
    ) {
        self.world
            .process_at_rank(root as i32)
            .scatter_into_root(global, out);
    }

    fn gather<T: Clone + mpi::datatype::Equivalence>(
        &self,
        local: &[T],
        out: &mut Vec<T>,
        root: usize,
    ) {
        let mut recvbuf = if self.rank == root {
            vec![local[0].clone(); local.len() * self.size]
        } else {
            Vec::new()
        };
        self.world
            .process_at_rank(root as i32)
            .gather_into_root(local, &mut recvbuf);
        if self.rank == root {
            *out = recvbuf;
        }
    }

    fn all_reduce(&self, x: f64) -> f64 {
        use mpi::collective::SystemOperation;
        let mut y = x;
        self.world.all_reduce_into(&x, &mut y, &SystemOperation::sum());
        y
    }

    fn parallel_mat_vec(&self, a: &faer::Mat<f64>, x: &[f64], y: &mut [f64]) {
        // For now, just serial mat-vec. TODO: partition by rank for distributed mat-vec.
        assert_eq!(a.ncols(), x.len());
        assert_eq!(a.nrows(), y.len());
        for i in 0..a.nrows() {
            y[i] = 0.0;
            for j in 0..a.ncols() {
                y[i] += a[(i, j)] * x[j];
            }
        }
    }
}
