#[cfg(feature = "mpi")]
use mpi::datatype::Equivalence;

pub trait Comm {
    type Vec;
    fn rank(&self) -> usize;
    fn size(&self) -> usize;
    fn barrier(&self);
    #[cfg(feature = "mpi")]
    fn scatter<T: Clone + Equivalence>(&self, global: &[T], out: &mut [T], root: usize);
    #[cfg(not(feature = "mpi"))]
    fn scatter<T: Clone>(&self, global: &[T], out: &mut [T], root: usize);
    #[cfg(feature = "mpi")]
    fn gather<T: Clone + Equivalence>(&self, local: &[T], out: &mut Vec<T>, root: usize);
    #[cfg(not(feature = "mpi"))]
    fn gather<T: Clone>(&self, local: &[T], out: &mut Vec<T>, root: usize);
    // New collective operations
    fn all_reduce(&self, x: f64) -> f64;
    fn dot(&self, a: &[f64], b: &[f64]) -> f64 {
        let local = a.iter().zip(b).map(|(&x, &y)| x * y).sum::<f64>();
        self.all_reduce(local)
    }
    // Parallel/distributed matrix-vector product
    fn parallel_mat_vec(&self, a: &faer::Mat<f64>, x: &[f64], y: &mut [f64]) {
        // Default: serial mat-vec
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

#[cfg(feature="mpi")]
pub mod mpi_comm;
#[cfg(feature="mpi")]
pub use mpi_comm::MpiComm;

#[cfg(feature="rayon")]
pub mod rayon_comm;
#[cfg(feature="rayon")]
pub use rayon_comm::RayonComm;

pub enum UniverseComm {
    #[cfg(feature="mpi")]
    Mpi(MpiComm),
    #[cfg(feature="rayon")]
    Rayon(RayonComm),
    #[cfg(not(any(feature="mpi", feature="rayon")))]
    Serial,
}

impl Comm for UniverseComm {
    type Vec = Vec<f64>; // Default, can be made generic
    fn rank(&self) -> usize {
        match self {
            #[cfg(feature="mpi")]
            UniverseComm::Mpi(comm) => comm.rank(),
            #[cfg(feature="rayon")]
            UniverseComm::Rayon(comm) => comm.rank(),
            #[cfg(not(any(feature="mpi", feature="rayon")))]
            UniverseComm::Serial => 0,
        }
    }
    fn size(&self) -> usize {
        match self {
            #[cfg(feature="mpi")]
            UniverseComm::Mpi(comm) => comm.size(),
            #[cfg(feature="rayon")]
            UniverseComm::Rayon(comm) => comm.size(),
            #[cfg(not(any(feature="mpi", feature="rayon")))]
            UniverseComm::Serial => 1,
        }
    }
    fn barrier(&self) {
        match self {
            #[cfg(feature="mpi")]
            UniverseComm::Mpi(comm) => comm.barrier(),
            #[cfg(feature="rayon")]
            UniverseComm::Rayon(comm) => comm.barrier(),
            #[cfg(not(any(feature="mpi", feature="rayon")))]
            UniverseComm::Serial => {},
        }
    }
    #[cfg(feature = "mpi")]
    fn scatter<T: Clone + Equivalence>(&self, global: &[T], out: &mut [T], root: usize) {
        match self {
            UniverseComm::Mpi(comm) => comm.scatter(global, out, root),
            _ => unreachable!(),
        }
    }
    #[cfg(not(feature = "mpi"))]
    fn scatter<T: Clone>(&self, global: &[T], out: &mut [T], root: usize) {
        match self {
            #[cfg(feature="rayon")]
            UniverseComm::Rayon(comm) => comm.scatter(global, out, root),
            #[cfg(not(feature="rayon"))]
            UniverseComm::Serial => out.copy_from_slice(global),
            _ => unreachable!(),
        }
    }
    #[cfg(feature = "mpi")]
    fn gather<T: Clone + Equivalence>(&self, local: &[T], out: &mut Vec<T>, root: usize) {
        match self {
            UniverseComm::Mpi(comm) => comm.gather(local, out, root),
            _ => unreachable!(),
        }
    }
    #[cfg(not(feature = "mpi"))]
    fn gather<T: Clone>(&self, local: &[T], out: &mut Vec<T>, root: usize) {
        match self {
            #[cfg(feature="rayon")]
            UniverseComm::Rayon(comm) => comm.gather(local, out, root),
            #[cfg(not(feature="rayon"))]
            UniverseComm::Serial => {
                out.clear();
                out.extend_from_slice(local);
            },
            _ => unreachable!(),
        }
    }
    fn all_reduce(&self, x: f64) -> f64 {
        match self {
            #[cfg(feature="mpi")]
            UniverseComm::Mpi(comm) => comm.all_reduce(x),
            #[cfg(feature="rayon")]
            UniverseComm::Rayon(comm) => comm.all_reduce(x),
            #[cfg(not(any(feature="mpi", feature="rayon")))]
            UniverseComm::Serial => x,
        }
    }
}

#[cfg(all(not(feature = "mpi")))]
pub trait Equivalence {}
