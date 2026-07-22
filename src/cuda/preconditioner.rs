use super::triangular::{CudaTriangularSolve, Triangle};
use super::{CudaCsrOp, CudaLinOp, CudaOperation, CudaRuntime, CudaVector};
use crate::algebra::prelude::*;
use crate::error::KError;
use crate::matrix::sparse::CsrMatrix;
use crate::preconditioner::PcCaps;
use std::any::Any;
use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};

pub trait CudaPreconditioner: Send + Sync + Any {
    fn dims(&self) -> (usize, usize);
    fn apply(&self, x: &CudaVector, y: &mut CudaVector) -> Result<(), KError>;
    fn prepare(&self) -> Result<(), KError> {
        Ok(())
    }
    fn capabilities(&self) -> PcCaps {
        PcCaps::default()
    }
    fn device_ordinal(&self) -> usize;
    fn as_any(&self) -> &dyn Any;
}

#[derive(Debug)]
pub struct CudaNone {
    runtime: Arc<CudaRuntime>,
    n: usize,
}

impl CudaNone {
    pub fn new(runtime: Arc<CudaRuntime>, n: usize) -> Self {
        Self { runtime, n }
    }
}

impl CudaPreconditioner for CudaNone {
    fn dims(&self) -> (usize, usize) {
        (self.n, self.n)
    }

    fn apply(&self, x: &CudaVector, y: &mut CudaVector) -> Result<(), KError> {
        x.ensure_compatible(y)?;
        if x.len() != self.n {
            return Err(KError::InvalidInput(format!(
                "CUDA identity preconditioner expected length {}, got {}",
                self.n,
                x.len()
            )));
        }
        self.runtime.copy(x.buffer(), y.buffer_mut())
    }

    fn capabilities(&self) -> PcCaps {
        PcCaps {
            supports_transpose: true,
            supports_conj_trans: true,
            is_spd: true,
            side_restriction: None,
        }
    }

    fn device_ordinal(&self) -> usize {
        self.runtime.device_ordinal()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

pub struct CudaJacobi {
    inverse_diagonal: CudaCsrOp,
    tiny_diagonal_threshold: R,
}

impl std::fmt::Debug for CudaJacobi {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaJacobi")
            .field("dims", &self.inverse_diagonal.dims())
            .field("device_ordinal", &self.device_ordinal())
            .field("tiny_diagonal_threshold", &self.tiny_diagonal_threshold)
            .finish()
    }
}

impl CudaJacobi {
    pub fn from_csr(matrix: &CudaCsrOp) -> Result<Self, KError> {
        Self::from_csr_with_threshold(matrix, 1e-14)
    }

    pub fn from_csr_with_threshold(
        matrix: &CudaCsrOp,
        tiny_diagonal_threshold: R,
    ) -> Result<Self, KError> {
        let (rows, cols) = matrix.dims();
        if rows != cols {
            return Err(KError::InvalidInput(format!(
                "CUDA Jacobi requires a square operator, got {rows}x{cols}"
            )));
        }
        if !tiny_diagonal_threshold.is_finite() || tiny_diagonal_threshold < 0.0 {
            return Err(KError::InvalidInput(
                "CUDA Jacobi tiny diagonal threshold must be finite and non-negative".into(),
            ));
        }
        let inverse = inverted_diagonal(matrix, tiny_diagonal_threshold)?;
        let row_offsets: Vec<usize> = (0..=rows).collect();
        let columns: Vec<usize> = (0..rows).collect();
        let diagonal = CsrMatrix::from_csr(rows, rows, row_offsets, columns, inverse);
        let inverse_diagonal = CudaCsrOp::from_host(matrix.runtime().clone(), &diagonal)?;
        Ok(Self {
            inverse_diagonal,
            tiny_diagonal_threshold,
        })
    }

    /// Refresh the inverse-diagonal values without rebuilding or re-uploading
    /// its CSR structure.
    pub fn update_from_csr(&self, matrix: &CudaCsrOp) -> Result<(), KError> {
        if matrix.dims() != self.dims() {
            return Err(KError::InvalidInput(format!(
                "CUDA Jacobi numeric update dimension mismatch: {:?} vs {:?}",
                matrix.dims(),
                self.dims()
            )));
        }
        let inverse = inverted_diagonal(matrix, self.tiny_diagonal_threshold)?;
        self.inverse_diagonal.update_values(&inverse)
    }
}

fn inverted_diagonal(matrix: &CudaCsrOp, threshold: R) -> Result<Vec<S>, KError> {
    let diagonal = matrix.diagonal_host()?;
    let mut inverse = Vec::with_capacity(diagonal.len());
    for (row, value) in diagonal.into_iter().enumerate() {
        if !value.is_finite() {
            return Err(KError::InvalidInput(format!(
                "CUDA Jacobi diagonal at row {row} is not finite"
            )));
        }
        if value.abs() <= threshold {
            return Err(KError::ZeroPivot(row));
        }
        inverse.push(value.inv());
    }
    Ok(inverse)
}

impl CudaPreconditioner for CudaJacobi {
    fn dims(&self) -> (usize, usize) {
        self.inverse_diagonal.dims()
    }

    fn apply(&self, x: &CudaVector, y: &mut CudaVector) -> Result<(), KError> {
        self.inverse_diagonal
            .apply(CudaOperation::NonTranspose, x, y)
    }

    fn prepare(&self) -> Result<(), KError> {
        self.inverse_diagonal.prepare()
    }

    fn capabilities(&self) -> PcCaps {
        PcCaps {
            supports_transpose: true,
            supports_conj_trans: true,
            is_spd: true,
            side_restriction: None,
        }
    }

    fn device_ordinal(&self) -> usize {
        self.inverse_diagonal.device_ordinal()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[derive(Clone, Debug)]
pub struct CudaAmgOptions {
    pub max_levels: usize,
    pub coarse_size: usize,
    pub pre_smoothing_steps: usize,
    pub post_smoothing_steps: usize,
    pub coarse_iterations: usize,
    pub jacobi_omega: R,
}

impl Default for CudaAmgOptions {
    fn default() -> Self {
        Self {
            max_levels: 10,
            coarse_size: 32,
            pre_smoothing_steps: 2,
            post_smoothing_steps: 2,
            coarse_iterations: 16,
            jacobi_omega: 2.0 / 3.0,
        }
    }
}

struct CudaAmgLevel {
    operator: CudaCsrOp,
    smoother: CudaJacobi,
    prolongation: Option<CudaCsrOp>,
    restriction: Option<CudaCsrOp>,
}

struct CudaAmgLevelWorkspace {
    rhs: CudaVector,
    solution: CudaVector,
    residual: CudaVector,
    correction: CudaVector,
    temp: CudaVector,
}

/// Device-resident unsmoothed-aggregation AMG hierarchy.
///
/// Pairwise aggregates and Galerkin coarse operators are built on the host at
/// setup. All level operators, transfer operators, smoothers, and V-cycle
/// workspace are then retained on the selected CUDA device.
pub struct CudaAmg {
    levels: Vec<CudaAmgLevel>,
    workspace: Mutex<Vec<CudaAmgLevelWorkspace>>,
    runtime: Arc<CudaRuntime>,
    options: CudaAmgOptions,
    n: usize,
}

impl std::fmt::Debug for CudaAmg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaAmg")
            .field("dims", &(self.n, self.n))
            .field("levels", &self.levels.len())
            .field("options", &self.options)
            .finish_non_exhaustive()
    }
}

impl CudaAmg {
    pub fn from_csr(matrix: &CudaCsrOp) -> Result<Self, KError> {
        Self::from_csr_with_options(matrix, CudaAmgOptions::default())
    }

    pub fn from_csr_with_options(
        matrix: &CudaCsrOp,
        options: CudaAmgOptions,
    ) -> Result<Self, KError> {
        validate_amg_options(&options)?;
        let (n, cols) = matrix.dims();
        if n != cols {
            return Err(KError::InvalidInput(format!(
                "CUDA AMG requires a square operator, got {n}x{cols}"
            )));
        }
        let (rows, columns, values) = matrix.host_csr_parts()?;
        let mut host_operators = vec![CsrMatrix::from_csr(n, n, rows, columns, values)];
        let mut host_transfers = Vec::new();
        while host_operators.len() < options.max_levels
            && host_operators.last().unwrap().nrows() > options.coarse_size
        {
            let fine = host_operators.last().unwrap();
            let (coarse, prolongation, restriction) = aggregate_level(fine)?;
            if coarse.nrows() >= fine.nrows() {
                break;
            }
            host_transfers.push((prolongation, restriction));
            host_operators.push(coarse);
        }

        let runtime = matrix.runtime().clone();
        let mut levels = Vec::with_capacity(host_operators.len());
        for (index, host_operator) in host_operators.iter().enumerate() {
            let operator = CudaCsrOp::from_host(runtime.clone(), host_operator)?;
            let smoother = CudaJacobi::from_csr(&operator)?;
            let (prolongation, restriction) = if index < host_transfers.len() {
                (
                    Some(CudaCsrOp::from_host(
                        runtime.clone(),
                        &host_transfers[index].0,
                    )?),
                    Some(CudaCsrOp::from_host(
                        runtime.clone(),
                        &host_transfers[index].1,
                    )?),
                )
            } else {
                (None, None)
            };
            levels.push(CudaAmgLevel {
                operator,
                smoother,
                prolongation,
                restriction,
            });
        }

        let mut workspace = Vec::with_capacity(levels.len());
        for level in &levels {
            let size = level.operator.dims().0;
            workspace.push(CudaAmgLevelWorkspace {
                rhs: CudaVector::zeros(runtime.clone(), size)?,
                solution: CudaVector::zeros(runtime.clone(), size)?,
                residual: CudaVector::zeros(runtime.clone(), size)?,
                correction: CudaVector::zeros(runtime.clone(), size)?,
                temp: CudaVector::zeros(runtime.clone(), size)?,
            });
        }
        Ok(Self {
            levels,
            workspace: Mutex::new(workspace),
            runtime,
            options,
            n,
        })
    }

    pub fn level_count(&self) -> usize {
        self.levels.len()
    }

    fn smooth(
        &self,
        level: usize,
        workspace: &mut CudaAmgLevelWorkspace,
        steps: usize,
    ) -> Result<(), KError> {
        let hierarchy_level = &self.levels[level];
        for _ in 0..steps {
            hierarchy_level.operator.apply(
                CudaOperation::NonTranspose,
                &workspace.solution,
                &mut workspace.temp,
            )?;
            self.runtime
                .copy(workspace.rhs.buffer(), workspace.residual.buffer_mut())?;
            self.runtime.axpby(
                -S::one(),
                workspace.temp.buffer(),
                S::one(),
                workspace.residual.buffer_mut(),
            )?;
            hierarchy_level
                .smoother
                .apply(&workspace.residual, &mut workspace.correction)?;
            self.runtime.axpy(
                S::from_real(self.options.jacobi_omega),
                workspace.correction.buffer(),
                workspace.solution.buffer_mut(),
            )?;
        }
        Ok(())
    }

    fn v_cycle(&self, level: usize, workspace: &mut [CudaAmgLevelWorkspace]) -> Result<(), KError> {
        if level + 1 == self.levels.len() {
            return self.smooth(level, &mut workspace[level], self.options.coarse_iterations);
        }
        self.smooth(
            level,
            &mut workspace[level],
            self.options.pre_smoothing_steps,
        )?;
        {
            let (fine_levels, coarse_levels) = workspace.split_at_mut(level + 1);
            let fine = &mut fine_levels[level];
            let coarse = &mut coarse_levels[0];
            self.levels[level].operator.apply(
                CudaOperation::NonTranspose,
                &fine.solution,
                &mut fine.temp,
            )?;
            self.runtime
                .copy(fine.rhs.buffer(), fine.residual.buffer_mut())?;
            self.runtime.axpby(
                -S::one(),
                fine.temp.buffer(),
                S::one(),
                fine.residual.buffer_mut(),
            )?;
            self.levels[level].restriction.as_ref().unwrap().apply(
                CudaOperation::NonTranspose,
                &fine.residual,
                &mut coarse.rhs,
            )?;
            coarse.solution.fill_zero()?;
        }
        self.v_cycle(level + 1, workspace)?;
        {
            let (fine_levels, coarse_levels) = workspace.split_at_mut(level + 1);
            let fine = &mut fine_levels[level];
            let coarse = &coarse_levels[0];
            self.levels[level].prolongation.as_ref().unwrap().apply(
                CudaOperation::NonTranspose,
                &coarse.solution,
                &mut fine.correction,
            )?;
            self.runtime.axpy(
                S::one(),
                fine.correction.buffer(),
                fine.solution.buffer_mut(),
            )?;
        }
        self.smooth(
            level,
            &mut workspace[level],
            self.options.post_smoothing_steps,
        )
    }
}

impl CudaPreconditioner for CudaAmg {
    fn dims(&self) -> (usize, usize) {
        (self.n, self.n)
    }

    fn apply(&self, x: &CudaVector, y: &mut CudaVector) -> Result<(), KError> {
        x.ensure_compatible(y)?;
        if x.len() != self.n {
            return Err(KError::InvalidInput(format!(
                "CUDA AMG expected length {}, got {}",
                self.n,
                x.len()
            )));
        }
        let mut workspace = self
            .workspace
            .lock()
            .map_err(|_| KError::SolveError("CUDA AMG workspace mutex was poisoned".into()))?;
        self.runtime
            .copy(x.buffer(), workspace[0].rhs.buffer_mut())?;
        workspace[0].solution.fill_zero()?;
        self.v_cycle(0, &mut workspace)?;
        self.runtime
            .copy(workspace[0].solution.buffer(), y.buffer_mut())
    }

    fn prepare(&self) -> Result<(), KError> {
        for level in &self.levels {
            level.operator.prepare()?;
            if let Some(prolongation) = &level.prolongation {
                prolongation.prepare()?;
            }
            if let Some(restriction) = &level.restriction {
                restriction.prepare()?;
            }
        }
        Ok(())
    }

    fn capabilities(&self) -> PcCaps {
        PcCaps {
            supports_transpose: false,
            supports_conj_trans: false,
            is_spd: true,
            side_restriction: None,
        }
    }

    fn device_ordinal(&self) -> usize {
        self.runtime.device_ordinal()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

fn validate_amg_options(options: &CudaAmgOptions) -> Result<(), KError> {
    if options.max_levels == 0 || options.coarse_size == 0 || options.coarse_iterations == 0 {
        return Err(KError::InvalidInput(
            "CUDA AMG level, coarse-size, and coarse-iteration limits must be positive".into(),
        ));
    }
    if !options.jacobi_omega.is_finite() || options.jacobi_omega <= 0.0 {
        return Err(KError::InvalidInput(
            "CUDA AMG Jacobi damping must be finite and positive".into(),
        ));
    }
    Ok(())
}

fn aggregate_level(
    fine: &CsrMatrix<S>,
) -> Result<(CsrMatrix<S>, CsrMatrix<S>, CsrMatrix<S>), KError> {
    if fine.nrows() != fine.ncols() {
        return Err(KError::InvalidInput(
            "CUDA AMG aggregation requires square levels".into(),
        ));
    }
    let n = fine.nrows();
    let coarse_n = n.div_ceil(2);
    let prolongation = CsrMatrix::from_csr(
        n,
        coarse_n,
        (0..=n).collect(),
        (0..n).map(|row| row / 2).collect(),
        vec![S::one(); n],
    );
    let mut restriction_rows = Vec::with_capacity(coarse_n + 1);
    let mut restriction_columns = Vec::with_capacity(n);
    restriction_rows.push(0);
    for aggregate in 0..coarse_n {
        let start = 2 * aggregate;
        restriction_columns.push(start);
        if start + 1 < n {
            restriction_columns.push(start + 1);
        }
        restriction_rows.push(restriction_columns.len());
    }
    let restriction = CsrMatrix::from_csr(
        coarse_n,
        n,
        restriction_rows,
        restriction_columns,
        vec![S::one(); n],
    );

    let mut coarse_rows_maps = vec![BTreeMap::<usize, S>::new(); coarse_n];
    for row in 0..n {
        let coarse_row = row / 2;
        for entry in fine.row_ptr()[row]..fine.row_ptr()[row + 1] {
            let coarse_column = fine.col_idx()[entry] / 2;
            let slot = coarse_rows_maps[coarse_row]
                .entry(coarse_column)
                .or_insert(S::zero());
            *slot += fine.values()[entry];
        }
    }
    let mut rows = Vec::with_capacity(coarse_n + 1);
    let mut columns = Vec::new();
    let mut values = Vec::new();
    rows.push(0);
    for row in coarse_rows_maps {
        for (column, value) in row {
            columns.push(column);
            values.push(value);
        }
        rows.push(columns.len());
    }
    Ok((
        CsrMatrix::from_csr(coarse_n, coarse_n, rows, columns, values),
        prolongation,
        restriction,
    ))
}

/// Dense diagonal-block inverse stored as a device CSR operator. Factorization
/// is performed during setup on the host; every apply is device resident.
pub struct CudaBlockJacobi {
    inverse_blocks: CudaCsrOp,
    block_size: usize,
    tiny_pivot_threshold: R,
}

impl std::fmt::Debug for CudaBlockJacobi {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaBlockJacobi")
            .field("dims", &self.inverse_blocks.dims())
            .field("block_size", &self.block_size)
            .field("tiny_pivot_threshold", &self.tiny_pivot_threshold)
            .finish()
    }
}

impl CudaBlockJacobi {
    pub fn from_csr(matrix: &CudaCsrOp, block_size: usize) -> Result<Self, KError> {
        Self::from_csr_with_threshold(matrix, block_size, 1e-14)
    }

    pub fn from_csr_with_threshold(
        matrix: &CudaCsrOp,
        block_size: usize,
        tiny_pivot_threshold: R,
    ) -> Result<Self, KError> {
        validate_block_jacobi_options(matrix, block_size, tiny_pivot_threshold)?;
        let (rows, columns, values) = inverted_blocks(matrix, block_size, tiny_pivot_threshold)?;
        let n = matrix.dims().0;
        let inverse_blocks =
            CudaCsrOp::from_csr_parts(matrix.runtime().clone(), n, n, &rows, &columns, &values)?;
        Ok(Self {
            inverse_blocks,
            block_size,
            tiny_pivot_threshold,
        })
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Re-factor numeric blocks and update the existing device CSR values.
    pub fn update_from_csr(&self, matrix: &CudaCsrOp) -> Result<(), KError> {
        validate_block_jacobi_options(matrix, self.block_size, self.tiny_pivot_threshold)?;
        let (_, _, values) = inverted_blocks(matrix, self.block_size, self.tiny_pivot_threshold)?;
        self.inverse_blocks.update_values(&values)
    }
}

impl CudaPreconditioner for CudaBlockJacobi {
    fn dims(&self) -> (usize, usize) {
        self.inverse_blocks.dims()
    }

    fn apply(&self, x: &CudaVector, y: &mut CudaVector) -> Result<(), KError> {
        self.inverse_blocks.apply(CudaOperation::NonTranspose, x, y)
    }

    fn prepare(&self) -> Result<(), KError> {
        self.inverse_blocks.prepare()
    }

    fn capabilities(&self) -> PcCaps {
        PcCaps {
            supports_transpose: true,
            supports_conj_trans: true,
            is_spd: true,
            side_restriction: None,
        }
    }

    fn device_ordinal(&self) -> usize {
        self.inverse_blocks.device_ordinal()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

struct CudaChebyshevScratch {
    v0: CudaVector,
    v1: CudaVector,
    v2: CudaVector,
}

/// Device-resident Chebyshev polynomial preconditioner. Its recurrence and
/// normalization match the host `ChebyshevPc`; all work vectors are allocated
/// at construction and reused by every application.
pub struct CudaChebyshev {
    operator: Arc<dyn CudaLinOp>,
    degree: usize,
    lambda_min: R,
    lambda_max: R,
    runtime: Arc<CudaRuntime>,
    n: usize,
    scratch: Mutex<CudaChebyshevScratch>,
}

impl std::fmt::Debug for CudaChebyshev {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaChebyshev")
            .field("dims", &(self.n, self.n))
            .field("degree", &self.degree)
            .field("lambda_min", &self.lambda_min)
            .field("lambda_max", &self.lambda_max)
            .finish_non_exhaustive()
    }
}

impl CudaChebyshev {
    pub fn new(
        runtime: Arc<CudaRuntime>,
        operator: Arc<dyn CudaLinOp>,
        degree: usize,
        lambda_min: R,
        lambda_max: R,
    ) -> Result<Self, KError> {
        let (rows, cols) = operator.dims();
        if rows != cols {
            return Err(KError::InvalidInput(format!(
                "CUDA Chebyshev preconditioner requires a square operator, got {rows}x{cols}"
            )));
        }
        if degree == 0
            || !lambda_min.is_finite()
            || !lambda_max.is_finite()
            || lambda_min < 0.0
            || lambda_max <= lambda_min
        {
            return Err(KError::InvalidInput(
                "CUDA Chebyshev requires degree >= 1 and finite bounds 0 <= lambda_min < lambda_max"
                    .into(),
            ));
        }
        if operator.device_ordinal() != runtime.device_ordinal() {
            return Err(KError::InvalidInput(
                "CUDA Chebyshev operator and runtime use different devices".into(),
            ));
        }
        let scratch = CudaChebyshevScratch {
            v0: CudaVector::zeros(runtime.clone(), rows)?,
            v1: CudaVector::zeros(runtime.clone(), rows)?,
            v2: CudaVector::zeros(runtime.clone(), rows)?,
        };
        Ok(Self {
            operator,
            degree,
            lambda_min,
            lambda_max,
            runtime,
            n: rows,
            scratch: Mutex::new(scratch),
        })
    }

    pub fn degree(&self) -> usize {
        self.degree
    }

    pub fn spectral_bounds(&self) -> (R, R) {
        (self.lambda_min, self.lambda_max)
    }
}

impl CudaPreconditioner for CudaChebyshev {
    fn dims(&self) -> (usize, usize) {
        (self.n, self.n)
    }

    fn apply(&self, x: &CudaVector, y: &mut CudaVector) -> Result<(), KError> {
        x.ensure_compatible(y)?;
        if x.len() != self.n {
            return Err(KError::InvalidInput(format!(
                "CUDA Chebyshev expected length {}, got {}",
                self.n,
                x.len()
            )));
        }
        let c = (self.lambda_max + self.lambda_min) / 2.0;
        let d = (self.lambda_max - self.lambda_min) / 2.0;
        if d.abs() < R::EPSILON {
            return self.runtime.copy(x.buffer(), y.buffer_mut());
        }
        let normalization = chebyshev_t(self.degree, -c / d);
        if !normalization.is_finite() || normalization.abs() <= R::EPSILON {
            return Err(KError::InvalidInput(
                "CUDA Chebyshev normalization is zero or non-finite for the selected bounds".into(),
            ));
        }
        let tau = S::from_real(1.0 / normalization);
        let mut scratch = self.scratch.lock().map_err(|_| {
            KError::SolveError("CUDA Chebyshev workspace mutex was poisoned".into())
        })?;
        let CudaChebyshevScratch { v0, v1, v2 } = &mut *scratch;

        self.operator.apply(CudaOperation::NonTranspose, x, v1)?;
        self.runtime
            .axpby(S::from_real(-c), x.buffer(), S::one(), v1.buffer_mut())?;
        self.runtime.scale(S::from_real(1.0 / d), v1.buffer_mut())?;
        if self.degree == 1 {
            self.runtime.copy(v1.buffer(), y.buffer_mut())?;
            return self.runtime.scale(tau, y.buffer_mut());
        }

        self.runtime.copy(x.buffer(), v0.buffer_mut())?;
        for _ in 2..=self.degree {
            self.operator.apply(CudaOperation::NonTranspose, v1, v2)?;
            self.runtime
                .axpby(S::from_real(-c), v1.buffer(), S::one(), v2.buffer_mut())?;
            self.runtime.scale(S::from_real(2.0 / d), v2.buffer_mut())?;
            self.runtime.axpy(-S::one(), v0.buffer(), v2.buffer_mut())?;
            std::mem::swap(v0, v1);
            std::mem::swap(v1, v2);
        }
        self.runtime.copy(v1.buffer(), y.buffer_mut())?;
        self.runtime.scale(tau, y.buffer_mut())
    }

    fn prepare(&self) -> Result<(), KError> {
        self.operator.prepare()
    }

    fn capabilities(&self) -> PcCaps {
        PcCaps {
            supports_transpose: false,
            supports_conj_trans: false,
            is_spd: false,
            side_restriction: None,
        }
    }

    fn device_ordinal(&self) -> usize {
        self.runtime.device_ordinal()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

struct Ilu0Factors {
    lower_rows: Vec<usize>,
    lower_columns: Vec<usize>,
    lower_values: Vec<S>,
    upper_rows: Vec<usize>,
    upper_columns: Vec<usize>,
    upper_values: Vec<S>,
}

/// ILU(0) factorization computed on the host and applied on the device through
/// two analyzed cuSPARSE SpSV triangular solves.
pub struct CudaIlu0 {
    lower: CudaTriangularSolve,
    upper: CudaTriangularSolve,
    scratch: Mutex<CudaVector>,
    runtime: Arc<CudaRuntime>,
    n: usize,
    pivot_threshold: R,
}

impl std::fmt::Debug for CudaIlu0 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaIlu0")
            .field("dims", &(self.n, self.n))
            .field("pivot_threshold", &self.pivot_threshold)
            .finish_non_exhaustive()
    }
}

impl CudaIlu0 {
    pub fn from_csr(matrix: &CudaCsrOp) -> Result<Self, KError> {
        Self::from_csr_with_threshold(matrix, 1e-14)
    }

    pub fn from_csr_with_threshold(matrix: &CudaCsrOp, pivot_threshold: R) -> Result<Self, KError> {
        if !pivot_threshold.is_finite() || pivot_threshold < 0.0 {
            return Err(KError::InvalidInput(
                "CUDA ILU(0) pivot threshold must be finite and non-negative".into(),
            ));
        }
        let n = matrix.dims().0;
        let factors = ilu0_factors(matrix, pivot_threshold)?;
        let runtime = matrix.runtime().clone();
        let lower = CudaTriangularSolve::new(
            runtime.clone(),
            n,
            &factors.lower_rows,
            &factors.lower_columns,
            &factors.lower_values,
            Triangle::LowerUnit,
        )?;
        let upper = CudaTriangularSolve::new(
            runtime.clone(),
            n,
            &factors.upper_rows,
            &factors.upper_columns,
            &factors.upper_values,
            Triangle::UpperNonUnit,
        )?;
        Ok(Self {
            lower,
            upper,
            scratch: Mutex::new(CudaVector::zeros(runtime.clone(), n)?),
            runtime,
            n,
            pivot_threshold,
        })
    }

    /// Re-factor values on the host and update the existing device factor
    /// allocations. The matrix sparsity pattern must be unchanged.
    pub fn update_from_csr(&self, matrix: &CudaCsrOp) -> Result<(), KError> {
        if matrix.dims() != (self.n, self.n) {
            return Err(KError::InvalidInput(format!(
                "CUDA ILU(0) numeric update dimension mismatch: {:?} vs ({}, {})",
                matrix.dims(),
                self.n,
                self.n
            )));
        }
        let factors = ilu0_factors(matrix, self.pivot_threshold)?;
        self.lower.update_values(&factors.lower_values)?;
        self.upper.update_values(&factors.upper_values)
    }
}

impl CudaPreconditioner for CudaIlu0 {
    fn dims(&self) -> (usize, usize) {
        (self.n, self.n)
    }

    fn apply(&self, x: &CudaVector, y: &mut CudaVector) -> Result<(), KError> {
        let mut scratch = self
            .scratch
            .lock()
            .map_err(|_| KError::SolveError("CUDA ILU(0) workspace mutex was poisoned".into()))?;
        self.lower.solve(x, &mut scratch)?;
        self.upper.solve(&scratch, y)
    }

    fn capabilities(&self) -> PcCaps {
        PcCaps {
            supports_transpose: false,
            supports_conj_trans: false,
            is_spd: false,
            side_restriction: None,
        }
    }

    fn device_ordinal(&self) -> usize {
        self.runtime.device_ordinal()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

fn ilu0_factors(matrix: &CudaCsrOp, threshold: R) -> Result<Ilu0Factors, KError> {
    let (n, cols) = matrix.dims();
    if n != cols {
        return Err(KError::InvalidInput(format!(
            "CUDA ILU(0) requires a square operator, got {n}x{cols}"
        )));
    }
    let (rows, columns, values) = matrix.host_csr_parts()?;
    ilu0_factors_from_parts(n, &rows, &columns, &values, threshold)
}

fn ilu0_factors_from_parts(
    n: usize,
    rows: &[usize],
    columns: &[usize],
    values: &[S],
    threshold: R,
) -> Result<Ilu0Factors, KError> {
    let mut lu = values.to_vec();
    let mut positions = Vec::with_capacity(n);
    let mut diagonals = Vec::with_capacity(n);
    for row in 0..n {
        let mut map = HashMap::with_capacity(rows[row + 1] - rows[row]);
        let mut diagonal = None;
        for entry in rows[row]..rows[row + 1] {
            let column = columns[entry];
            if map.insert(column, entry).is_some() {
                return Err(KError::InvalidInput(format!(
                    "CUDA ILU(0) does not support duplicate column {column} in row {row}"
                )));
            }
            if column == row {
                diagonal = Some(entry);
            }
        }
        positions.push(map);
        diagonals.push(diagonal.ok_or_else(|| {
            KError::InvalidInput(format!("CUDA ILU(0) matrix is missing diagonal row {row}"))
        })?);
    }

    for row in 0..n {
        let mut lower_entries: Vec<(usize, usize)> = positions[row]
            .iter()
            .filter_map(|(&column, &entry)| (column < row).then_some((column, entry)))
            .collect();
        lower_entries.sort_unstable_by_key(|&(column, _)| column);
        for (pivot_row, entry) in lower_entries {
            let pivot = lu[diagonals[pivot_row]];
            if !pivot.is_finite() || pivot.abs() <= threshold {
                return Err(KError::ZeroPivot(pivot_row));
            }
            let multiplier = lu[entry] / pivot;
            lu[entry] = multiplier;
            for upper_entry in rows[pivot_row]..rows[pivot_row + 1] {
                let column = columns[upper_entry];
                if column <= pivot_row {
                    continue;
                }
                if let Some(&target) = positions[row].get(&column) {
                    lu[target] = lu[target] - multiplier * lu[upper_entry];
                }
            }
        }
        let pivot = lu[diagonals[row]];
        if !pivot.is_finite() || pivot.abs() <= threshold {
            return Err(KError::ZeroPivot(row));
        }
    }

    let mut lower_rows = Vec::with_capacity(n + 1);
    let mut lower_columns = Vec::new();
    let mut lower_values = Vec::new();
    let mut upper_rows = Vec::with_capacity(n + 1);
    let mut upper_columns = Vec::new();
    let mut upper_values = Vec::new();
    lower_rows.push(0);
    upper_rows.push(0);
    for row in 0..n {
        let mut entries: Vec<(usize, S)> = (rows[row]..rows[row + 1])
            .map(|entry| (columns[entry], lu[entry]))
            .collect();
        entries.sort_unstable_by_key(|&(column, _)| column);
        for &(column, value) in &entries {
            if column < row {
                lower_columns.push(column);
                lower_values.push(value);
            }
        }
        lower_columns.push(row);
        lower_values.push(S::one());
        lower_rows.push(lower_columns.len());
        for (column, value) in entries {
            if column >= row {
                upper_columns.push(column);
                upper_values.push(value);
            }
        }
        upper_rows.push(upper_columns.len());
    }
    Ok(Ilu0Factors {
        lower_rows,
        lower_columns,
        lower_values,
        upper_rows,
        upper_columns,
        upper_values,
    })
}

fn chebyshev_t(degree: usize, x: R) -> R {
    match degree {
        0 => 1.0,
        1 => x,
        _ => {
            let mut t0 = 1.0;
            let mut t1 = x;
            for _ in 2..=degree {
                let t2 = 2.0 * x * t1 - t0;
                t0 = t1;
                t1 = t2;
            }
            t1
        }
    }
}

fn validate_block_jacobi_options(
    matrix: &CudaCsrOp,
    block_size: usize,
    threshold: R,
) -> Result<(), KError> {
    let (rows, cols) = matrix.dims();
    if rows != cols {
        return Err(KError::InvalidInput(format!(
            "CUDA block Jacobi requires a square operator, got {rows}x{cols}"
        )));
    }
    if block_size == 0 {
        return Err(KError::InvalidInput(
            "CUDA block Jacobi block size must be nonzero".into(),
        ));
    }
    if !threshold.is_finite() || threshold < 0.0 {
        return Err(KError::InvalidInput(
            "CUDA block Jacobi pivot threshold must be finite and non-negative".into(),
        ));
    }
    Ok(())
}

fn inverted_blocks(
    matrix: &CudaCsrOp,
    block_size: usize,
    threshold: R,
) -> Result<(Vec<usize>, Vec<usize>, Vec<S>), KError> {
    let n = matrix.dims().0;
    let (source_rows, source_columns, source_values) = matrix.host_csr_parts()?;
    let mut rows = Vec::with_capacity(n + 1);
    let mut columns = Vec::new();
    let mut values = Vec::new();
    rows.push(0);
    for block_start in (0..n).step_by(block_size) {
        let block_end = (block_start + block_size).min(n);
        let width = block_end - block_start;
        let mut dense = vec![S::zero(); width * width];
        for global_row in block_start..block_end {
            let local_row = global_row - block_start;
            for entry in source_rows[global_row]..source_rows[global_row + 1] {
                let global_column = source_columns[entry];
                if (block_start..block_end).contains(&global_column) {
                    dense[local_row * width + global_column - block_start] = source_values[entry];
                }
            }
        }
        let inverse = invert_dense_block(&mut dense, width, block_start, threshold)?;
        for local_row in 0..width {
            for local_column in 0..width {
                columns.push(block_start + local_column);
                values.push(inverse[local_row * width + local_column]);
            }
            rows.push(columns.len());
        }
    }
    Ok((rows, columns, values))
}

fn invert_dense_block(
    matrix: &mut [S],
    width: usize,
    global_start: usize,
    threshold: R,
) -> Result<Vec<S>, KError> {
    let mut inverse = vec![S::zero(); width * width];
    for row in 0..width {
        inverse[row * width + row] = S::one();
    }
    for column in 0..width {
        let mut pivot_row = column;
        let mut pivot_abs = matrix[column * width + column].abs();
        for row in (column + 1)..width {
            let candidate = matrix[row * width + column].abs();
            if candidate > pivot_abs {
                pivot_abs = candidate;
                pivot_row = row;
            }
        }
        if !pivot_abs.is_finite() || pivot_abs <= threshold {
            return Err(KError::ZeroPivot(global_start + column));
        }
        if pivot_row != column {
            for entry in 0..width {
                matrix.swap(column * width + entry, pivot_row * width + entry);
                inverse.swap(column * width + entry, pivot_row * width + entry);
            }
        }
        let pivot_inverse = matrix[column * width + column].inv();
        for entry in 0..width {
            matrix[column * width + entry] = matrix[column * width + entry] * pivot_inverse;
            inverse[column * width + entry] = inverse[column * width + entry] * pivot_inverse;
        }
        for row in 0..width {
            if row == column {
                continue;
            }
            let factor = matrix[row * width + column];
            for entry in 0..width {
                matrix[row * width + entry] =
                    matrix[row * width + entry] - factor * matrix[column * width + entry];
                inverse[row * width + entry] =
                    inverse[row * width + entry] - factor * inverse[column * width + entry];
            }
        }
    }
    if inverse.iter().any(|value| !value.is_finite()) {
        return Err(KError::InvalidInput(format!(
            "CUDA block Jacobi inverse for block at row {global_start} is not finite"
        )));
    }
    Ok(inverse)
}

#[cfg(test)]
mod block_tests {
    use super::*;

    #[test]
    fn dense_block_inverse_handles_complex_compatible_arithmetic() {
        let mut block = vec![
            S::from_real(4.0),
            S::from_real(1.0),
            S::from_real(2.0),
            S::from_real(3.0),
        ];
        let inverse = invert_dense_block(&mut block, 2, 0, 1e-14).unwrap();
        let expected = [0.3, -0.1, -0.2, 0.4];
        for (actual, expected) in inverse.iter().zip(expected) {
            assert!((*actual - S::from_real(expected)).abs() < 1e-12);
        }
    }

    #[test]
    fn dense_block_inverse_reports_singular_pivot() {
        let mut block = vec![S::one(), S::one(), S::one(), S::one()];
        assert!(matches!(
            invert_dense_block(&mut block, 2, 7, 1e-14),
            Err(KError::ZeroPivot(8))
        ));
    }

    #[test]
    fn ilu0_factorization_matches_tridiagonal_lu() {
        let factors = ilu0_factors_from_parts(
            3,
            &[0, 2, 5, 7],
            &[0, 1, 0, 1, 2, 1, 2],
            &[
                S::from_real(4.0),
                S::one(),
                S::from_real(2.0),
                S::from_real(3.0),
                S::one(),
                S::one(),
                S::from_real(2.0),
            ],
            1e-14,
        )
        .unwrap();
        assert_eq!(factors.lower_rows, [0, 1, 3, 5]);
        assert_eq!(factors.lower_columns, [0, 0, 1, 1, 2]);
        assert_eq!(factors.upper_rows, [0, 2, 4, 5]);
        assert_eq!(factors.upper_columns, [0, 1, 1, 2, 2]);
        let expected_lower = [1.0, 0.5, 1.0, 0.4, 1.0];
        let expected_upper = [4.0, 1.0, 2.5, 1.0, 1.6];
        for (actual, expected) in factors.lower_values.iter().zip(expected_lower) {
            assert!((*actual - S::from_real(expected)).abs() < 1e-12);
        }
        for (actual, expected) in factors.upper_values.iter().zip(expected_upper) {
            assert!((*actual - S::from_real(expected)).abs() < 1e-12);
        }
    }

    #[test]
    fn pairwise_aggregation_builds_galerkin_operator_and_transfers() {
        let fine = CsrMatrix::from_csr(
            4,
            4,
            vec![0, 2, 5, 8, 10],
            vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3],
            vec![
                S::from_real(2.0),
                S::from_real(-1.0),
                S::from_real(-1.0),
                S::from_real(2.0),
                S::from_real(-1.0),
                S::from_real(-1.0),
                S::from_real(2.0),
                S::from_real(-1.0),
                S::from_real(-1.0),
                S::from_real(2.0),
            ],
        );
        let (coarse, prolongation, restriction) = aggregate_level(&fine).unwrap();
        assert_eq!((coarse.nrows(), coarse.ncols()), (2, 2));
        assert_eq!(coarse.row_ptr(), [0, 2, 4]);
        assert_eq!(coarse.col_idx(), [0, 1, 0, 1]);
        let expected = [2.0, -1.0, -1.0, 2.0];
        for (actual, expected) in coarse.values().iter().zip(expected) {
            assert!((*actual - S::from_real(expected)).abs() < 1e-12);
        }
        assert_eq!((prolongation.nrows(), prolongation.ncols()), (4, 2));
        assert_eq!((restriction.nrows(), restriction.ncols()), (2, 4));
    }
}
