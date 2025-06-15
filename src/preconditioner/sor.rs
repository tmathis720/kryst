use std::marker::PhantomData;
use std::fmt;
use bitflags::bitflags;
use crate::preconditioner::Preconditioner;
use crate::core::traits::{MatVec, Indexing};
use crate::error::KError;
use num_traits::Float;

bitflags! {
    #[derive(Copy, Clone, Debug)]
    pub struct MatSorType: u32 {
        const ZERO_INITIAL_GUESS       = 0b000_00001;
        const APPLY_LOWER              = 0b000_00010; // forward Gauss–Seidel
        const APPLY_UPPER              = 0b000_00100; // backward
        const SYMMETRIC_SWEEP          = Self::APPLY_LOWER.bits() | Self::APPLY_UPPER.bits();
        const LOCAL_FORWARD_SWEEP      = 0b000_01000;
        const LOCAL_BACKWARD_SWEEP     = 0b000_10000;
        const LOCAL_SYMMETRIC_SWEEP    = Self::LOCAL_FORWARD_SWEEP.bits() | Self::LOCAL_BACKWARD_SWEEP.bits();
        const EISENSTAT                = 0b0010_0000;
    }
}

pub struct Sor<M, V, T> {
    pub its:      usize,
    pub lits:     usize,
    pub sym:      MatSorType,
    pub omega:    T,
    pub fshift:   T,
    pub inv_diag: Vec<T>,
    pub a:        Option<M>,
    _phantom:     PhantomData<V>,
}

impl<M, V, T> Sor<M, V, T>
where
    T: Float,
{
    pub fn new(omega: T, its: usize, lits: usize, sym: MatSorType, fshift: T) -> Self {
        Self { its, lits, sym, omega, fshift, inv_diag: Vec::new(), a: None, _phantom: PhantomData }
    }
    pub fn set_omega(&mut self, omega: T) { self.omega = omega; }
    pub fn omega(&self) -> T { self.omega }
    pub fn set_its(&mut self, its: usize) { self.its = its; }
    pub fn its(&self) -> usize { self.its }
    pub fn set_lits(&mut self, lits: usize) { self.lits = lits; }
    pub fn lits(&self) -> usize { self.lits }
    pub fn set_sym(&mut self, sym: MatSorType) { self.sym = sym; }
    pub fn sym(&self) -> MatSorType { self.sym }
    pub fn set_fshift(&mut self, fshift: T) { self.fshift = fshift; }
    pub fn fshift(&self) -> T { self.fshift }
}

impl<M, V, T> fmt::Display for Sor<M, V, T>
where
    T: Float + fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SOR(omega={}, its={}, lits={}, sym={:?}, fshift={})",
            self.omega, self.its, self.lits, self.sym, self.fshift)
    }
}

impl<M, V, T> Preconditioner<M, V> for Sor<M, V, T>
where
    M: MatVec<V> + Indexing + Clone + std::ops::Index<(usize, usize), Output = T>,
    V: AsRef<[T]> + AsMut<[T]> + From<Vec<T>>,
    T: Float + Copy,
{
    fn setup(&mut self, a: &M) -> Result<(), KError> {
        self.a = Some(a.clone());
        let n = a.nrows();
        self.inv_diag.resize(n, T::zero());
        for i in 0..n {
            let aii = a[(i, i)] + self.fshift;
            if aii == T::zero() {
                return Err(KError::ZeroPivot(i));
            }
            self.inv_diag[i] = T::one() / aii;
        }
        Ok(())
    }

    fn apply(&self, x: &V, y: &mut V) -> Result<(), KError> {
        let a = self.a.as_ref().expect("SOR not setup");
        let n = x.as_ref().len();
        y.as_mut().fill(T::zero());
        for _ in 0..self.its {
            // FORWARD sweep
            if self.sym.intersects(MatSorType::APPLY_LOWER) {
                for i in 0..n {
                    let mut sigma = T::zero();
                    for j in 0..i {
                        sigma = sigma + a[(i, j)] * y.as_ref()[j];
                    }
                    if !self.sym.contains(MatSorType::EISENSTAT) {
                        for j in (i+1)..n {
                            sigma = sigma + a[(i, j)] * x.as_ref()[j];
                        }
                    }
                    let xi = x.as_ref()[i];
                    let yi = (xi - sigma) * self.inv_diag[i];
                    y.as_mut()[i] = (T::one()-self.omega)*xi + self.omega*yi;
                }
            }
            // BACKWARD sweep
            if self.sym.intersects(MatSorType::APPLY_UPPER) {
                for ii in (0..n).rev() {
                    let mut sigma = T::zero();
                    for j in (ii+1)..n {
                        sigma = sigma + a[(ii, j)] * y.as_ref()[j];
                    }
                    if !self.sym.contains(MatSorType::EISENSTAT) {
                        for j in 0..ii {
                            sigma = sigma + a[(ii, j)] * y.as_ref()[j];
                        }
                    }
                    let xi = x.as_ref()[ii];
                    let yi = (xi - sigma) * self.inv_diag[ii];
                    y.as_mut()[ii] = (T::one()-self.omega)*xi + self.omega*yi;
                }
            }
        }
        Ok(())
    }
}
