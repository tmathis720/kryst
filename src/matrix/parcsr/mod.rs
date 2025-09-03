pub type Global = u64;
pub type Local = usize;

pub mod builder;
pub mod halo;
pub mod mat;

pub use halo::HaloPlan;
pub use mat::ParCsrMatrix;
