use std::sync::Mutex;

use crate::error::KError;
use crate::matrix::convert::csr_from_linop;
use crate::matrix::op::LinOp;
use crate::matrix::sparse::CsrMatrix;
use crate::matrix::spmv::csr_spmm_dense;
use crate::preconditioner::amg::AMG;
use crate::preconditioner::{FormatHint, PcCaps, PcSide, Preconditioner};
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
}

#[derive(Clone, Debug)]
enum EFactor {
    Chol(CholeskyFactor),
    Lu(LuFactor),
}

#[derive(Default)]
struct DeflationWorkspace {
    coarse: Vec<f64>,
    coarse_sol: Vec<f64>,
    fine_tmp: Vec<f64>,
    base_out: Vec<f64>,
}

impl DeflationWorkspace {
    fn ensure(&mut self, n: usize, k: usize) {
        if self.coarse.len() != k {
            self.coarse.resize(k, 0.0);
        }
        if self.coarse_sol.len() != k {
            self.coarse_sol.resize(k, 0.0);
        }
        if self.fine_tmp.len() != n {
            self.fine_tmp.resize(n, 0.0);
        }
        if self.base_out.len() != n {
            self.base_out.resize(n, 0.0);
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
        let tol = opts.cond_cap.map(|cap| (1.0 / cap).abs()).unwrap_or(1e-12);
        let z = gram_schmidt(&coarse.z, tol.max(1e-12));
        if z.ncols() == 0 {
            return Err(KError::InvalidInput("coarse space lost all columns".into()));
        }
        let k = z.ncols();
        let mut az = Mat::<f64>::zeros(n, k);
        csr_spmm_dense(a, z.as_ref(), az.as_mut())?;
        let e = build_e(&z, &az);
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

impl<PB> Preconditioner for DeflationPC<PB>
where
    PB: Preconditioner,
{
    fn setup(&mut self, op: &dyn LinOp<S = f64>) -> Result<(), KError> {
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
        base_out.fill(0.0);
        self.base.apply(side, fine_tmp, base_out)?;

        for i in 0..n {
            y[i] += base_out[i];
        }

        Ok(())
    }

    fn supports_numeric_update(&self) -> bool {
        self.base.supports_numeric_update()
    }

    fn update_numeric(&mut self, op: &dyn LinOp<S = f64>) -> Result<(), KError> {
        self.base.update_numeric(op)?;
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

    fn required_format(&self) -> FormatHint {
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
