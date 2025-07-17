#[cfg(feature = "mpi")]
use mpi::datatype::Equivalence;
#[cfg(feature = "mpi")]
use mpi::topology::{Communicator, Color};

/// Abstract communicator for reductions & splits
pub trait Comm: Send + Sync + 'static {
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
    
    /// All‐reduce a scalar (sum) across ranks
    fn all_reduce_f64(&self, local: f64) -> f64;
    
    /// Split this communicator into sub‐colors
    fn split(&self, color: i32, key: i32) -> UniverseComm;
    
    /// Legacy all_reduce method for backward compatibility
    fn all_reduce(&self, x: f64) -> f64 {
        self.all_reduce_f64(x)
    }
    
    fn dot(&self, a: &[f64], b: &[f64]) -> f64 {
        let local = a.iter().zip(b).map(|(&x, &y)| x * y).sum::<f64>();
        self.all_reduce_f64(local)
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

/// Default no‐MPI/no‐parallel communicator for serial execution
#[derive(Clone)]
pub struct NoComm;

impl Comm for NoComm {
    type Vec = Vec<f64>;
    
    fn rank(&self) -> usize { 0 }
    fn size(&self) -> usize { 1 }
    fn barrier(&self) {}
    
    #[cfg(feature = "mpi")]
    fn scatter<T: Clone + Equivalence>(&self, global: &[T], out: &mut [T], _root: usize) {
        // For no-comm case, just copy from first elements
        for (dst, src) in out.iter_mut().zip(global.iter()) {
            *dst = src.clone();
        }
    }
    #[cfg(not(feature = "mpi"))]
    fn scatter<T: Clone>(&self, global: &[T], out: &mut [T], _root: usize) {
        // For no-comm case, just copy from first elements
        for (dst, src) in out.iter_mut().zip(global.iter()) {
            *dst = src.clone();
        }
    }
    
    #[cfg(feature = "mpi")]
    fn gather<T: Clone + Equivalence>(&self, local: &[T], out: &mut Vec<T>, _root: usize) {
        out.clear();
        out.extend_from_slice(local);
    }
    #[cfg(not(feature = "mpi"))]
    fn gather<T: Clone>(&self, local: &[T], out: &mut Vec<T>, _root: usize) {
        out.clear();
        out.extend_from_slice(local);
    }
    
    fn all_reduce_f64(&self, local: f64) -> f64 { local }
    
    fn split(&self, _color: i32, _key: i32) -> UniverseComm { 
        UniverseComm::NoComm(NoComm)
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
    NoComm(NoComm),
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
            UniverseComm::NoComm(comm) => comm.rank(),
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
            UniverseComm::NoComm(comm) => comm.size(),
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
            UniverseComm::NoComm(comm) => comm.barrier(),
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
            UniverseComm::NoComm(comm) => comm.scatter(global, out, root),
            UniverseComm::Mpi(comm) => comm.scatter(global, out, root),
            _ => unreachable!(),
        }
    }
    #[cfg(not(feature = "mpi"))]
    fn scatter<T: Clone>(&self, global: &[T], out: &mut [T], root: usize) {
        match self {
            UniverseComm::NoComm(comm) => comm.scatter(global, out, root),
            #[cfg(feature="rayon")]
            UniverseComm::Rayon(comm) => comm.scatter(global, out, root),
            #[cfg(not(feature="rayon"))]
            UniverseComm::Serial => {
                for (dst, src) in out.iter_mut().zip(global.iter()) {
                    *dst = src.clone();
                }
            },
        }
    }
    #[cfg(feature = "mpi")]
    fn gather<T: Clone + Equivalence>(&self, local: &[T], out: &mut Vec<T>, root: usize) {
        match self {
            UniverseComm::NoComm(comm) => comm.gather(local, out, root),
            UniverseComm::Mpi(comm) => comm.gather(local, out, root),
            _ => unreachable!(),
        }
    }
    #[cfg(not(feature = "mpi"))]
    fn gather<T: Clone>(&self, local: &[T], out: &mut Vec<T>, _root: usize) {
        match self {
            UniverseComm::NoComm(comm) => comm.gather(local, out, _root),
            #[cfg(feature="rayon")]
            UniverseComm::Rayon(comm) => comm.gather(local, out, _root),
            #[cfg(not(feature="rayon"))]
            UniverseComm::Serial => {
                out.clear();
                out.extend_from_slice(local);
            },
        }
    }
    fn all_reduce(&self, x: f64) -> f64 {
        match self {
            UniverseComm::NoComm(comm) => comm.all_reduce(x),
            #[cfg(feature="mpi")]
            UniverseComm::Mpi(comm) => comm.all_reduce(x),
            #[cfg(feature="rayon")]
            UniverseComm::Rayon(comm) => comm.all_reduce(x),
            #[cfg(not(any(feature="mpi", feature="rayon")))]
            UniverseComm::Serial => x,
        }
    }
    
    fn all_reduce_f64(&self, local: f64) -> f64 {
        match self {
            UniverseComm::NoComm(comm) => comm.all_reduce_f64(local),
            #[cfg(feature="mpi")]
            UniverseComm::Mpi(comm) => comm.all_reduce_f64(local),
            #[cfg(feature="rayon")]
            UniverseComm::Rayon(comm) => comm.all_reduce_f64(local),
            #[cfg(not(any(feature="mpi", feature="rayon")))]
            UniverseComm::Serial => local,
        }
    }
    
    fn split(&self, color: i32, key: i32) -> UniverseComm {
        match self {
            UniverseComm::NoComm(comm) => comm.split(color, key),
            #[cfg(feature="mpi")]
            UniverseComm::Mpi(comm) => {
                // Split the MPI communicator and return a new UniverseComm
                let sub = comm.world.split_by_color_with_key(Color::with_value(color), key).expect("Failed to split communicator");
                let sub_rank = sub.rank() as usize;
                let sub_size = sub.size() as usize;
                let new_comm = MpiComm { 
                    world: sub, 
                    rank: sub_rank, 
                    size: sub_size 
                };
                UniverseComm::Mpi(new_comm)
            },
            #[cfg(feature="rayon")]
            UniverseComm::Rayon(_comm) => UniverseComm::Rayon(RayonComm::new()),
            #[cfg(not(any(feature="mpi", feature="rayon")))]
            UniverseComm::Serial => UniverseComm::Serial,
        }
    }
}

#[cfg(all(not(feature = "mpi")))]
pub trait Equivalence {}

pub enum ReduceOp {
    Sum,
    // Add more as needed
}
