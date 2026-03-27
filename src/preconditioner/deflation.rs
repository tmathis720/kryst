use std::sync::Mutex;

#[cfg(feature = "complex")]
use crate::algebra::bridge::BridgeScratch;
#[allow(unused_imports)]
use crate::algebra::prelude::*;
use crate::error::KError;
#[cfg(all(feature = "mpi", not(feature = "complex")))]
use crate::matrix::DistCsrOp;
use crate::matrix::convert::csr_from_linop;
use crate::matrix::op::LinOp;
use crate::matrix::sparse::CsrMatrix;
use crate::matrix::spmv::csr_spmm_dense;
#[cfg(feature = "complex")]
use crate::ops::kpc::KPreconditioner;
use crate::parallel::{Comm, UniverseComm};
use crate::preconditioner::amg::AMG;
#[cfg(feature = "complex")]
use crate::preconditioner::bridge::{
    apply_pc_mut_s as bridge_apply_pc_mut_s, apply_pc_s as bridge_apply_pc_s,
};
use crate::preconditioner::{OpFormat, PcCaps, PcDistributedSupport, PcSide, Preconditioner};
use faer::{Mat, MatRef};

/// Dense coarse space extracted from an AMG hierarchy or provided externally.
#[derive(Clone, Debug)]
pub struct AmgCoarseSpace {
    /// Column-major storage of the coarse basis vectors.
    pub z: Mat<f64>,
    /// Row range owned locally for ParCSR. Serial uses the full matrix.
    pub local_range: Option<(usize, usize)>,
}

/// Source for constructing the coarse space.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ZSource {
    /// Use identity columns on the coarsest grid, prolongated to the finest.
    CoarsestIdentity { cap_k: Option<usize> },
    /// Reuse near-nullspace vectors already defined on the finest grid.
    NearNullspace,
    /// Provided explicitly by the caller (no extraction performed).
    External,
}

/// Configuration knobs for deflation setup.
#[derive(Clone, Debug)]
pub struct DeflationOptions {
    pub z_source: ZSource,
    pub cond_cap: Option<f64>,
    pub augment_initial_guess: bool,
}

impl Default for DeflationOptions {
    fn default() -> Self {
        Self {
            z_source: ZSource::CoarsestIdentity { cap_k: None },
            cond_cap: None,
            augment_initial_guess: false,
        }
    }
}

#[derive(Clone, Debug)]
struct CholeskyFactor {
    n: usize,
    data: Vec<f64>,
}

impl CholeskyFactor {
    fn decompose(mat: &MatRef<'_, f64>) -> Result<Self, ()> {
        let n = mat.nrows();
        if n != mat.ncols() {
            return Err(());
        }
        let mut data = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..=i {
                let mut sum = mat[(i, j)];
                for k in 0..j {
                    sum -= data[i * n + k] * data[j * n + k];
                }
                if i == j {
                    if sum <= 0.0 {
                        return Err(());
                    }
                    data[i * n + j] = sum.sqrt();
                } else {
                    let piv = data[j * n + j];
                    if piv == 0.0 {
                        return Err(());
                    }
                    data[i * n + j] = sum / piv;
                }
            }
        }
        Ok(Self { n, data })
    }

    fn solve_in_place(&self, rhs: &mut [f64]) {
        debug_assert_eq!(rhs.len(), self.n);
        // Forward solve L y = rhs.
        for i in 0..self.n {
            let mut acc = rhs[i];
            for k in 0..i {
                acc -= self.data[i * self.n + k] * rhs[k];
            }
            rhs[i] = acc / self.data[i * self.n + i];
        }
        // Backward solve Lᵀ x = y.
        for i in (0..self.n).rev() {
            let mut acc = rhs[i];
            for k in (i + 1)..self.n {
                acc -= self.data[k * self.n + i] * rhs[k];
            }
            rhs[i] = acc / self.data[i * self.n + i];
        }
    }

    #[cfg(feature = "complex")]
    fn solve_in_place_complex(&self, rhs: &mut [S]) {
        debug_assert_eq!(rhs.len(), self.n);
        for i in 0..self.n {
            let mut acc = rhs[i];
            for k in 0..i {
                acc -= S::from_real(self.data[i * self.n + k]) * rhs[k];
            }
            rhs[i] = acc / S::from_real(self.data[i * self.n + i]);
        }
        for i in (0..self.n).rev() {
            let mut acc = rhs[i];
            for k in (i + 1)..self.n {
                acc -= S::from_real(self.data[k * self.n + i]) * rhs[k];
            }
            rhs[i] = acc / S::from_real(self.data[i * self.n + i]);
        }
    }
}

#[derive(Clone, Debug)]
struct LuFactor {
    n: usize,
    lu: Vec<f64>,
    piv: Vec<usize>,
}

impl LuFactor {
    fn decompose(mat: &MatRef<'_, f64>) -> Result<Self, KError> {
        let n = mat.nrows();
        if n != mat.ncols() {
            return Err(KError::InvalidInput("LU requires square matrix".into()));
        }
        let mut lu = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                lu[i * n + j] = mat[(i, j)];
            }
        }
        let mut piv: Vec<usize> = (0..n).collect();
        for k in 0..n {
            let mut piv_row = k;
            let mut max_val = lu[k * n + k].abs();
            for i in (k + 1)..n {
                let val = lu[i * n + k].abs();
                if val > max_val {
                    max_val = val;
                    piv_row = i;
                }
            }
            if max_val == 0.0 {
                return Err(KError::ZeroPivot(k));
            }
            if piv_row != k {
                for j in 0..n {
                    lu.swap(k * n + j, piv_row * n + j);
                }
                piv.swap(k, piv_row);
            }
            let pivot = lu[k * n + k];
            for i in (k + 1)..n {
                lu[i * n + k] /= pivot;
                let mult = lu[i * n + k];
                for j in (k + 1)..n {
                    lu[i * n + j] -= mult * lu[k * n + j];
                }
            }
        }
        Ok(Self { n, lu, piv })
    }

    fn solve_in_place(&self, rhs: &mut [f64]) {
        debug_assert_eq!(rhs.len(), self.n);
        // Apply row permutations to rhs.
        let mut permuted = vec![0.0; self.n];
        for (i, &p) in self.piv.iter().enumerate() {
            permuted[i] = rhs[p];
        }
        // Forward solve L y = P b (unit diagonal).
        for i in 0..self.n {
            let mut acc = permuted[i];
            for k in 0..i {
                acc -= self.lu[i * self.n + k] * permuted[k];
            }
            permuted[i] = acc;
        }
        // Backward solve U x = y.
        for i in (0..self.n).rev() {
            let mut acc = permuted[i];
            for k in (i + 1)..self.n {
                acc -= self.lu[i * self.n + k] * permuted[k];
            }
            permuted[i] = acc / self.lu[i * self.n + i];
        }
        rhs.copy_from_slice(&permuted);
    }

    #[cfg(feature = "complex")]
    fn solve_in_place_complex(&self, rhs: &mut [S]) {
        debug_assert_eq!(rhs.len(), self.n);
        let mut permuted = vec![S::zero(); self.n];
        for (i, &p) in self.piv.iter().enumerate() {
            permuted[i] = rhs[p];
        }
        for i in 0..self.n {
            let mut acc = permuted[i];
            for k in 0..i {
                acc -= S::from_real(self.lu[i * self.n + k]) * permuted[k];
            }
            permuted[i] = acc;
        }
        for i in (0..self.n).rev() {
            let mut acc = permuted[i];
            for k in (i + 1)..self.n {
                acc -= S::from_real(self.lu[i * self.n + k]) * permuted[k];
            }
            permuted[i] = acc / S::from_real(self.lu[i * self.n + i]);
        }
        rhs.copy_from_slice(&permuted);
    }
}

#[derive(Clone, Debug)]
enum EFactor {
    Chol(CholeskyFactor),
    Lu(LuFactor),
}

#[derive(Default)]
struct DeflationWorkspace {
    coarse: Vec<R>,
    coarse_sol: Vec<R>,
    fine_tmp: Vec<R>,
    base_out: Vec<R>,
}

impl DeflationWorkspace {
    fn ensure(&mut self, n: usize, k: usize) {
        if self.coarse.len() != k {
            self.coarse.resize(k, R::default());
        }
        if self.coarse_sol.len() != k {
            self.coarse_sol.resize(k, R::default());
        }
        if self.fine_tmp.len() != n {
            self.fine_tmp.resize(n, R::default());
        }
        if self.base_out.len() != n {
            self.base_out.resize(n, R::default());
        }
    }
}

pub struct DeflationPC<PB> {
    base: PB,
    a: CsrMatrix<f64>,
    z: Mat<f64>,
    az: Mat<f64>,
    e: Mat<f64>,
    e_factor: EFactor,
    local_range: Option<(usize, usize)>,
    dist_comm: Option<UniverseComm>,
    augment_initial_guess: bool,
    work: Mutex<DeflationWorkspace>,
}

fn gram_schmidt(z: &Mat<f64>, tol: f64) -> Mat<f64> {
    let (n, k) = z.shape();
    let mut kept: Vec<Vec<f64>> = Vec::with_capacity(k);
    for col in 0..k {
        let mut v = vec![0.0; n];
        for i in 0..n {
            v[i] = z[(i, col)];
        }
        for q in &kept {
            let mut dot = 0.0;
            for i in 0..n {
                dot += q[i] * v[i];
            }
            for i in 0..n {
                v[i] -= dot * q[i];
            }
        }
        let mut norm2 = 0.0;
        for &vi in &v {
            norm2 += vi * vi;
        }
        if norm2.sqrt() > tol {
            let scale = 1.0 / norm2.sqrt();
            for vi in &mut v {
                *vi *= scale;
            }
            kept.push(v);
        }
    }
    let cols = kept.len();
    let mut out = Mat::<f64>::zeros(n, cols);
    for (j, col) in kept.iter().enumerate() {
        for i in 0..n {
            out[(i, j)] = col[i];
        }
    }
    out
}

fn reduce_e_if_needed(e: &mut Mat<f64>, comm: Option<&UniverseComm>) {
    let Some(comm) = comm else {
        return;
    };
    if comm.size() <= 1 {
        return;
    }
    let mut flat = vec![0.0; e.nrows() * e.ncols()];
    for i in 0..e.nrows() {
        for j in 0..e.ncols() {
            flat[i * e.ncols() + j] = e[(i, j)];
        }
    }
    comm.allreduce_sum_slice(&mut flat);
    for i in 0..e.nrows() {
        for j in 0..e.ncols() {
            e[(i, j)] = flat[i * e.ncols() + j];
        }
    }
}

fn build_e(z: &Mat<f64>, az: &Mat<f64>) -> Mat<f64> {
    let k = z.ncols();
    let mut e = Mat::<f64>::zeros(k, k);
    for i in 0..k {
        for j in 0..k {
            let mut acc = 0.0;
            for r in 0..z.nrows() {
                acc += z[(r, i)] * az[(r, j)];
            }
            e[(i, j)] = acc;
        }
    }
    e
}

impl<PB> DeflationPC<PB>
where
    PB: Preconditioner,
{
    pub fn new(
        base: PB,
        a: &CsrMatrix<f64>,
        coarse: AmgCoarseSpace,
        opts: &DeflationOptions,
    ) -> Result<Self, KError> {
        let n = a.nrows();
        if coarse.z.nrows() != n {
            return Err(KError::InvalidInput(
                "coarse space dimension mismatch".into(),
            ));
        }
        if let Some((start, end)) = coarse.local_range
            && end.saturating_sub(start) != n
        {
            return Err(KError::InvalidInput(
                "coarse space local_range must match local matrix rows".into(),
            ));
        }
        let tol = opts.cond_cap.map(|cap| (1.0 / cap).abs()).unwrap_or(1e-12);
        let z = gram_schmidt(&coarse.z, tol.max(1e-12));
        if z.ncols() == 0 {
            return Err(KError::InvalidInput("coarse space lost all columns".into()));
        }
        let k = z.ncols();
        let mut az = Mat::<f64>::zeros(n, k);
        csr_spmm_dense(a, z.as_ref(), az.as_mut())?;
        let mut e = build_e(&z, &az);
        reduce_e_if_needed(&mut e, None);
        let e_factor = match CholeskyFactor::decompose(&e.as_ref()) {
            Ok(ch) => EFactor::Chol(ch),
            Err(_) => {
                let lu = LuFactor::decompose(&e.as_ref())?;
                EFactor::Lu(lu)
            }
        };
        Ok(Self {
            base,
            a: a.clone(),
            z,
            az,
            e,
            e_factor,
            local_range: coarse.local_range,
            dist_comm: None,
            augment_initial_guess: opts.augment_initial_guess,
            work: Mutex::new(DeflationWorkspace::default()),
        })
    }

    pub fn coarse_dim(&self) -> usize {
        self.z.ncols()
    }

    pub fn fine_dim(&self) -> usize {
        self.z.nrows()
    }

    fn solve_e(&self, rhs: &mut [f64]) {
        match &self.e_factor {
            EFactor::Chol(ch) => ch.solve_in_place(rhs),
            EFactor::Lu(lu) => lu.solve_in_place(rhs),
        }
    }

    #[cfg(feature = "complex")]
    fn solve_e_complex(&self, rhs: &mut [S]) {
        match &self.e_factor {
            EFactor::Chol(ch) => ch.solve_in_place_complex(rhs),
            EFactor::Lu(lu) => lu.solve_in_place_complex(rhs),
        }
    }

    pub fn coarse_initial_guess(&self, b: &[f64], x0: &mut [f64]) -> Result<(), KError> {
        let n = self.z.nrows();
        if b.len() != n || x0.len() != n {
            return Err(KError::InvalidInput(
                "coarse_initial_guess: dimension mismatch".into(),
            ));
        }
        let k = self.z.ncols();
        let mut tmp = vec![0.0; k];
        for j in 0..k {
            let mut acc = 0.0;
            for i in 0..n {
                acc += self.z[(i, j)] * b[i];
            }
            tmp[j] = acc;
        }
        self.solve_e(&mut tmp);
        for i in 0..n {
            let mut acc = 0.0;
            for j in 0..k {
                acc += self.z[(i, j)] * tmp[j];
            }
            x0[i] = acc;
        }
        Ok(())
    }

    pub fn augment_initial_guess(&self) -> bool {
        self.augment_initial_guess
    }

    fn refresh_e(&mut self) -> Result<(), KError> {
        self.e = build_e(&self.z, &self.az);
        reduce_e_if_needed(&mut self.e, self.dist_comm.as_ref());
        self.e_factor = match CholeskyFactor::decompose(&self.e.as_ref()) {
            Ok(ch) => EFactor::Chol(ch),
            Err(_) => {
                let lu = LuFactor::decompose(&self.e.as_ref())?;
                EFactor::Lu(lu)
            }
        };
        Ok(())
    }
}

#[cfg(not(feature = "complex"))]
impl<PB> Preconditioner for DeflationPC<PB>
where
    PB: Preconditioner,
{
    fn dims(&self) -> (usize, usize) {
        let n = self.fine_dim();
        (n, n)
    }

    fn setup(&mut self, op: &dyn LinOp<S = f64>) -> Result<(), KError> {
        #[cfg(all(feature = "mpi", not(feature = "complex")))]
        {
            if let Some(dist) = op.as_any().downcast_ref::<DistCsrOp>() {
                self.dist_comm = Some(dist.comm());
            }
        }
        if self.dist_comm.is_none() {
            let comm = op.comm();
            if comm.size() > 1 {
                self.dist_comm = Some(comm);
            }
        }
        self.refresh_e()?;
        self.base.setup(op)
    }

    fn apply(&self, side: PcSide, r: &[f64], y: &mut [f64]) -> Result<(), KError> {
        let n = self.z.nrows();
        if r.len() != n || y.len() != n {
            return Err(KError::InvalidInput(
                "DeflationPC: dimension mismatch".into(),
            ));
        }
        let k = self.z.ncols();
        let mut work = self.work.lock().unwrap();
        work.ensure(n, k);
        let DeflationWorkspace {
            coarse,
            coarse_sol,
            fine_tmp,
            base_out,
        } = &mut *work;

        // t = Zᵀ r
        for j in 0..k {
            let mut acc = 0.0;
            for i in 0..n {
                acc += self.z[(i, j)] * r[i];
            }
            coarse[j] = acc;
            coarse_sol[j] = acc;
        }
        if let Some(comm) = self.dist_comm.as_ref()
            && self.local_range.is_some()
            && comm.size() > 1
        {
            comm.allreduce_sum_slice(coarse_sol);
        }
        self.solve_e(coarse_sol);

        // y := Z * coarse_sol (coarse correction)
        for i in 0..n {
            let mut acc = 0.0;
            for j in 0..k {
                acc += self.z[(i, j)] * coarse_sol[j];
            }
            y[i] = acc;
        }

        // fine_tmp = r - A Z y_c (use cached AZ)
        for i in 0..n {
            let mut acc = 0.0;
            for j in 0..k {
                acc += self.az[(i, j)] * coarse_sol[j];
            }
            fine_tmp[i] = r[i] - acc;
        }

        // base_out = M^{-1} fine_tmp
        base_out.fill(R::default());
        self.base.apply(side, fine_tmp, base_out)?;

        for i in 0..n {
            y[i] += base_out[i];
        }

        Ok(())
    }

    fn distributed_support(&self) -> PcDistributedSupport {
        if self.local_range.is_some() {
            PcDistributedSupport::Distributed
        } else {
            self.base.distributed_support()
        }
    }

    fn supports_numeric_update(&self) -> bool {
        self.base.supports_numeric_update()
    }

    fn update_numeric(&mut self, op: &dyn LinOp<S = f64>) -> Result<(), KError> {
        self.base.update_numeric(op)?;
        let updated = csr_from_linop(op, 0.0)?;
        let updated = updated.as_ref();
        if self.a.row_ptr() != updated.row_ptr() || self.a.col_idx() != updated.col_idx() {
            return Err(KError::InvalidInput(
                "deflation numeric update requires unchanged sparsity; call update_symbolic instead"
                    .into(),
            ));
        }
        self.a.values_mut().copy_from_slice(updated.values());
        csr_spmm_dense(&self.a, self.z.as_ref(), self.az.as_mut())?;
        self.refresh_e()
    }

    fn update_symbolic(&mut self, op: &dyn LinOp<S = f64>) -> Result<(), KError> {
        self.base.update_symbolic(op)?;
        let new_a = csr_from_linop(op, 0.0)?;
        self.a = (*new_a).clone();
        csr_spmm_dense(&self.a, self.z.as_ref(), self.az.as_mut())?;
        self.refresh_e()
    }

    fn required_format(&self) -> OpFormat {
        self.base.required_format()
    }

    fn preferred_drop_tol_for_format(&self) -> Option<f64> {
        self.base.preferred_drop_tol_for_format()
    }

    fn capabilities(&self) -> PcCaps {
        let mut caps = self.base.capabilities();
        if matches!(self.e_factor, EFactor::Chol(_)) && caps.is_spd {
            caps.is_spd = true;
        }
        caps
    }
}

#[cfg(feature = "complex")]
impl<PB> Preconditioner for DeflationPC<PB>
where
    PB: Preconditioner,
{
    fn dims(&self) -> (usize, usize) {
        let n = self.fine_dim();
        (n, n)
    }

    fn setup(&mut self, op: &dyn LinOp<S = S>) -> Result<(), KError> {
        self.base.setup(op)
    }

    fn apply(&self, side: PcSide, r: &[S], y: &mut [S]) -> Result<(), KError> {
        let n = self.z.nrows();
        if r.len() != n || y.len() != n {
            return Err(KError::InvalidInput(
                "DeflationPC: dimension mismatch".into(),
            ));
        }
        let k = self.z.ncols();
        let mut work = self.work.lock().unwrap();
        work.ensure(n, k);
        let DeflationWorkspace { base_out, .. } = &mut *work;
        let mut coarse = vec![S::zero(); k];
        let mut coarse_sol = vec![S::zero(); k];
        let mut fine_tmp = vec![S::zero(); n];

        for j in 0..k {
            let mut acc = S::zero();
            for i in 0..n {
                acc += S::from_real(self.z[(i, j)]) * r[i];
            }
            coarse[j] = acc;
            coarse_sol[j] = acc;
        }
        self.solve_e_complex(&mut coarse_sol);

        for i in 0..n {
            let mut acc = S::zero();
            for j in 0..k {
                acc += S::from_real(self.z[(i, j)]) * coarse_sol[j];
            }
            y[i] = acc;
        }

        for i in 0..n {
            let mut acc = S::zero();
            for j in 0..k {
                acc += S::from_real(self.az[(i, j)]) * coarse_sol[j];
            }
            fine_tmp[i] = r[i] - acc;
        }

        let mut base_y = vec![S::zero(); n];
        self.base.apply(side, &fine_tmp, &mut base_y)?;
        for i in 0..n {
            y[i] += base_y[i];
        }

        let _ = base_out;
        Ok(())
    }
}

pub fn with_amg_deflation<PB: Preconditioner>(
    base: PB,
    a: &CsrMatrix<f64>,
    amg: &AMG,
    opts: &DeflationOptions,
) -> Result<DeflationPC<PB>, KError> {
    let coarse = if matches!(opts.z_source, ZSource::External) {
        return Err(KError::InvalidInput(
            "ZSource::External requires explicit coarse space".into(),
        ));
    } else {
        amg.extract_coarse_space(opts)?
    };
    DeflationPC::new(base, a, coarse, opts)
}

#[cfg(feature = "complex")]
impl<PB> KPreconditioner for DeflationPC<PB>
where
    PB: Preconditioner,
{
    type Scalar = S;

    #[inline]
    fn dims(&self) -> (usize, usize) {
        <Self as Preconditioner>::dims(self)
    }

    fn apply_s(
        &self,
        side: PcSide,
        x: &[S],
        y: &mut [S],
        scratch: &mut BridgeScratch,
    ) -> Result<(), KError> {
        bridge_apply_pc_s(self, side, x, y, scratch)
    }

    fn apply_mut_s(
        &mut self,
        side: PcSide,
        x: &[S],
        y: &mut [S],
        scratch: &mut BridgeScratch,
    ) -> Result<(), KError> {
        bridge_apply_pc_mut_s(self, side, x, y, scratch)
    }
}

#[cfg(all(test, feature = "complex"))]
mod tests {
    use super::*;
    use crate::algebra::bridge::BridgeScratch;
    use crate::matrix::sparse::CsrMatrix;
    use crate::ops::kpc::KPreconditioner;
    use crate::preconditioner::{DeflationOptions, Jacobi, PcSide, ZSource};
    use faer::Mat;

    #[test]
    fn deflation_complex_path_preserves_imaginary_component() {
        let n = 3;
        let a = CsrMatrix::from_csr(n, n, vec![0, 1, 2, 3], vec![0, 1, 2], vec![2.0, 3.0, 4.0]);
        let mut z = Mat::<f64>::zeros(n, 1);
        for i in 0..n {
            z[(i, 0)] = 1.0;
        }
        let coarse = AmgCoarseSpace {
            z,
            local_range: None,
        };
        let opts = DeflationOptions {
            z_source: ZSource::External,
            cond_cap: None,
            augment_initial_guess: false,
        };
        let base = Jacobi::new();
        let pc = DeflationPC::new(base, &a, coarse, &opts).expect("deflation construction");

        let rhs = vec![S::from_parts(1.0, 0.75); n];
        let mut out = vec![S::zero(); n];
        pc.apply(PcSide::Left, &rhs, &mut out).unwrap();
        assert!(out.iter().any(|v| v.imag().abs() > 1e-12));
    }

    #[test]
    fn apply_s_runs_for_complex_rhs() {
        let n = 4;
        let row_ptr = vec![0, 1, 2, 3, 4];
        let col_idx = vec![0, 1, 2, 3];
        let values = vec![4.0, 5.0, 6.0, 7.0];
        let a = CsrMatrix::from_csr(n, n, row_ptr, col_idx, values);

        let mut z = Mat::<f64>::zeros(n, 1);
        for i in 0..n {
            z[(i, 0)] = 1.0;
        }
        let coarse = AmgCoarseSpace {
            z,
            local_range: None,
        };
        let opts = DeflationOptions {
            z_source: ZSource::External,
            cond_cap: None,
            augment_initial_guess: false,
        };

        let base = Jacobi::new();
        let pc = DeflationPC::new(base, &a, coarse, &opts).expect("deflation construction");
        let rhs = vec![S::from_parts(1.0, -0.25); n];
        let mut out = vec![S::zero(); n];
        let mut scratch = BridgeScratch::default();
        pc.apply_s(PcSide::Left, &rhs, &mut out, &mut scratch)
            .unwrap();
        assert!(out.iter().all(|v| v.is_finite()));
    }
}
