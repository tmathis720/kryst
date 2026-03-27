//! Matrix conditioning helpers for preconditioner setup.

use crate::error::KError;
use crate::matrix::dense_api::{DenseMatMut, DenseMatRef};
use crate::matrix::sparse::CsrMatrix;
use std::fmt;
use std::str::FromStr;

const DEFAULT_TINY_DIAG: f64 = 1e-12;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScaleDirection {
    Row,
    Col,
    Both,
}

impl ScaleDirection {
    pub const fn allowed() -> &'static [&'static str] {
        &["row", "col", "both"]
    }
}

impl FromStr for ScaleDirection {
    type Err = KError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "row" => Ok(ScaleDirection::Row),
            "col" | "column" => Ok(ScaleDirection::Col),
            "both" => Ok(ScaleDirection::Both),
            other => Err(KError::InvalidInput(format!(
                "Invalid pc_scale: '{other}'. Allowed: {}",
                Self::allowed().join(", ")
            ))),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScaleNorm {
    One,
    Inf,
}

impl ScaleNorm {
    pub const fn allowed() -> &'static [&'static str] {
        &["1", "one", "inf", "infty", "infinity"]
    }
}

impl FromStr for ScaleNorm {
    type Err = KError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "1" | "one" => Ok(ScaleNorm::One),
            "inf" | "infty" | "infinity" => Ok(ScaleNorm::Inf),
            other => Err(KError::InvalidInput(format!(
                "Invalid pc_scale_norm: '{other}'. Allowed: {}",
                Self::allowed().join(", ")
            ))),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ConditioningOptions {
    pub fix_diag: bool,
    pub shift_diag: Option<f64>,
    pub diag_inject_tau: Option<f64>,
    pub scale: Option<ScaleDirection>,
    pub scale_norm: ScaleNorm,
    pub tiny_threshold: f64,
}

impl Default for ConditioningOptions {
    fn default() -> Self {
        Self {
            fix_diag: false,
            shift_diag: None,
            diag_inject_tau: None,
            scale: None,
            scale_norm: ScaleNorm::One,
            tiny_threshold: DEFAULT_TINY_DIAG,
        }
    }
}

impl ConditioningOptions {
    pub fn is_active(&self) -> bool {
        self.fix_diag
            || self.shift_diag.is_some()
            || self.diag_inject_tau.is_some()
            || self.scale.is_some()
    }

    pub fn validate(&self) -> Result<(), KError> {
        if let Some(v) = self.shift_diag
            && !v.is_finite() {
                return Err(KError::InvalidInput("pc_shift_diag must be finite".into()));
            }
        if let Some(v) = self.diag_inject_tau
            && !v.is_finite() {
                return Err(KError::InvalidInput(
                    "pc_diag_inject_tau must be finite".into(),
                ));
            }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct NormStats {
    pub min: f64,
    pub median: f64,
    pub max: f64,
}

impl fmt::Display for NormStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "min={:.3e} med={:.3e} max={:.3e}",
            self.min, self.median, self.max
        )
    }
}

#[derive(Clone, Debug)]
pub struct ConditioningStats {
    pub nrows: usize,
    pub ncols: usize,
    pub diag_min_abs: f64,
    pub diag_median_abs: f64,
    pub diag_tiny_count: usize,
    pub diag_missing_count: usize,
    pub row_norm_1: NormStats,
    pub row_norm_inf: NormStats,
    pub col_norm_1: NormStats,
    pub col_norm_inf: NormStats,
    pub symmetry_estimate: Option<f64>,
    pub zero_rows: Vec<usize>,
}

fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[mid - 1] + values[mid]) * 0.5
    } else {
        values[mid]
    }
}

fn norm_stats(values: &[f64]) -> NormStats {
    if values.is_empty() {
        return NormStats {
            min: 0.0,
            median: 0.0,
            max: 0.0,
        };
    }
    let mut buf = values.to_vec();
    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let med = median(&mut buf);
    NormStats {
        min,
        median: med,
        max,
    }
}

fn symmetry_estimate_dense<M: DenseMatRef<f64>>(matrix: &M) -> Option<f64> {
    if matrix.nrows() != matrix.ncols() {
        return None;
    }
    let n = matrix.nrows();
    let mut sum_abs = 0.0;
    let mut sum_diff = 0.0;
    for i in 0..n {
        for j in 0..n {
            let a = matrix.get(i, j);
            let b = matrix.get(j, i);
            sum_abs += a.abs();
            sum_diff += (a - b).abs();
        }
    }
    if sum_abs == 0.0 {
        Some(1.0)
    } else {
        Some((1.0 - (sum_diff / sum_abs)).max(0.0))
    }
}

fn csr_get_value(
    row_ptr: &[usize],
    col_idx: &[usize],
    values: &[f64],
    row: usize,
    col: usize,
) -> f64 {
    let start = row_ptr[row];
    let end = row_ptr[row + 1];
    if let Ok(pos) = col_idx[start..end].binary_search(&col) {
        values[start + pos]
    } else {
        0.0
    }
}

fn symmetry_estimate_csr(a: &CsrMatrix<f64>) -> Option<f64> {
    if a.nrows() != a.ncols() {
        return None;
    }
    let n = a.nrows();
    let row_ptr = a.row_ptr();
    let col_idx = a.col_idx();
    let values = a.values();
    let mut sum_abs = 0.0;
    let mut sum_diff = 0.0;
    for i in 0..n {
        for p in row_ptr[i]..row_ptr[i + 1] {
            let j = col_idx[p];
            let aij = values[p];
            let aji = csr_get_value(row_ptr, col_idx, values, j, i);
            sum_abs += aij.abs();
            sum_diff += (aij - aji).abs();
        }
    }
    if sum_abs == 0.0 {
        Some(1.0)
    } else {
        Some((1.0 - (sum_diff / sum_abs)).max(0.0))
    }
}

pub fn analyze_dense<M: DenseMatRef<f64>>(matrix: &M, tiny_threshold: f64) -> ConditioningStats {
    let nrows = matrix.nrows();
    let ncols = matrix.ncols();
    let diag_len = nrows.min(ncols);
    let mut diag_vals = Vec::with_capacity(diag_len);
    let mut diag_tiny = 0;
    let mut diag_missing = 0;

    for i in 0..diag_len {
        let val = matrix.get(i, i).abs();
        if val == 0.0 {
            diag_missing += 1;
        } else if val <= tiny_threshold {
            diag_tiny += 1;
        }
        diag_vals.push(val);
    }

    let diag_min_abs = if diag_vals.is_empty() {
        0.0
    } else {
        diag_vals.iter().copied().fold(f64::INFINITY, f64::min)
    };
    let diag_median_abs = median(&mut diag_vals);

    let mut row_norm_1 = vec![0.0; nrows];
    let mut row_norm_inf = vec![0.0; nrows];
    let mut col_norm_1 = vec![0.0; ncols];
    let mut col_norm_inf = vec![0.0; ncols];

    for i in 0..nrows {
        for j in 0..ncols {
            let val = matrix.get(i, j).abs();
            row_norm_1[i] += val;
            if val > row_norm_inf[i] {
                row_norm_inf[i] = val;
            }
            col_norm_1[j] += val;
            if val > col_norm_inf[j] {
                col_norm_inf[j] = val;
            }
        }
    }

    let zero_rows: Vec<usize> = row_norm_1
        .iter()
        .enumerate()
        .filter_map(|(i, v)| if *v == 0.0 { Some(i) } else { None })
        .collect();

    ConditioningStats {
        nrows,
        ncols,
        diag_min_abs,
        diag_median_abs,
        diag_tiny_count: diag_tiny,
        diag_missing_count: diag_missing,
        row_norm_1: norm_stats(&row_norm_1),
        row_norm_inf: norm_stats(&row_norm_inf),
        col_norm_1: norm_stats(&col_norm_1),
        col_norm_inf: norm_stats(&col_norm_inf),
        symmetry_estimate: symmetry_estimate_dense(matrix),
        zero_rows,
    }
}

pub fn analyze_csr(a: &CsrMatrix<f64>, tiny_threshold: f64) -> ConditioningStats {
    let nrows = a.nrows();
    let ncols = a.ncols();
    let diag_len = nrows.min(ncols);
    let row_ptr = a.row_ptr();
    let col_idx = a.col_idx();
    let values = a.values();
    let mut diag_vals = Vec::with_capacity(diag_len);
    let mut diag_tiny = 0;
    let mut diag_missing = 0;

    for i in 0..diag_len {
        let val = csr_get_value(row_ptr, col_idx, values, i, i).abs();
        if val == 0.0 {
            diag_missing += 1;
        } else if val <= tiny_threshold {
            diag_tiny += 1;
        }
        diag_vals.push(val);
    }

    let diag_min_abs = if diag_vals.is_empty() {
        0.0
    } else {
        diag_vals.iter().copied().fold(f64::INFINITY, f64::min)
    };
    let diag_median_abs = median(&mut diag_vals);

    let mut row_norm_1 = vec![0.0; nrows];
    let mut row_norm_inf = vec![0.0; nrows];
    let mut col_norm_1 = vec![0.0; ncols];
    let mut col_norm_inf = vec![0.0; ncols];

    for i in 0..nrows {
        for p in row_ptr[i]..row_ptr[i + 1] {
            let j = col_idx[p];
            let val = values[p].abs();
            row_norm_1[i] += val;
            if val > row_norm_inf[i] {
                row_norm_inf[i] = val;
            }
            col_norm_1[j] += val;
            if val > col_norm_inf[j] {
                col_norm_inf[j] = val;
            }
        }
    }

    let zero_rows: Vec<usize> = row_norm_1
        .iter()
        .enumerate()
        .filter_map(|(i, v)| if *v == 0.0 { Some(i) } else { None })
        .collect();

    ConditioningStats {
        nrows,
        ncols,
        diag_min_abs,
        diag_median_abs,
        diag_tiny_count: diag_tiny,
        diag_missing_count: diag_missing,
        row_norm_1: norm_stats(&row_norm_1),
        row_norm_inf: norm_stats(&row_norm_inf),
        col_norm_1: norm_stats(&col_norm_1),
        col_norm_inf: norm_stats(&col_norm_inf),
        symmetry_estimate: symmetry_estimate_csr(a),
        zero_rows,
    }
}

#[cfg(feature = "logging")]
fn log_stats(label: &str, stage: &str, stats: &ConditioningStats) {
    if !log::log_enabled!(log::Level::Info) {
        return;
    }
    log::info!(
        "Conditioning {label} ({stage}): diag |min|={:.3e} med={:.3e} tiny={} missing={}",
        stats.diag_min_abs,
        stats.diag_median_abs,
        stats.diag_tiny_count,
        stats.diag_missing_count
    );
    log::info!(
        "Conditioning {label} ({stage}): row 1-norm {row1}, row inf-norm {rowinf}",
        row1 = stats.row_norm_1,
        rowinf = stats.row_norm_inf
    );
    log::info!(
        "Conditioning {label} ({stage}): col 1-norm {col1}, col inf-norm {colinf}",
        col1 = stats.col_norm_1,
        colinf = stats.col_norm_inf
    );
    if let Some(sym) = stats.symmetry_estimate {
        log::info!("Conditioning {label} ({stage}): symmetry≈{sym:.3} (1=symmetric)",);
    }
    if !stats.zero_rows.is_empty() {
        let preview: Vec<String> = stats
            .zero_rows
            .iter()
            .take(8)
            .map(|i| i.to_string())
            .collect();
        log::info!(
            "Conditioning {label} ({stage}): zero rows={} [{}{}]",
            stats.zero_rows.len(),
            preview.join(", "),
            if stats.zero_rows.len() > 8 {
                ", ..."
            } else {
                ""
            }
        );
    }
}

#[cfg(not(feature = "logging"))]
fn log_stats(_label: &str, _stage: &str, _stats: &ConditioningStats) {}

pub fn log_conditioning(label: &str, opts: &ConditioningOptions) {
    #[cfg(feature = "logging")]
    if log::log_enabled!(log::Level::Info) {
        let mut parts: Vec<String> = Vec::new();
        if opts.fix_diag {
            parts.push("fixdiag".to_string());
        }
        if let Some(shift) = opts.shift_diag {
            parts.push(format!("shift_diag={shift:.3e}"));
        }
        if let Some(tau) = opts.diag_inject_tau {
            parts.push(format!("diag_inject_tau={tau:.3e}"));
        }
        if let Some(scale) = opts.scale {
            parts.push(format!("scale={scale:?}"));
            parts.push(format!("scale_norm={:?}", opts.scale_norm));
        }
        if !parts.is_empty() {
            log::info!("Conditioning {label}: applied {}", parts.join(", "));
        }
    }
}

fn row_norms_dense<M: DenseMatRef<f64>>(matrix: &M, norm: ScaleNorm) -> Vec<f64> {
    let nrows = matrix.nrows();
    let ncols = matrix.ncols();
    let mut out = vec![0.0; nrows];
    for i in 0..nrows {
        let mut acc = 0.0;
        for j in 0..ncols {
            let v = matrix.get(i, j).abs();
            match norm {
                ScaleNorm::One => acc += v,
                ScaleNorm::Inf => {
                    if v > acc {
                        acc = v;
                    }
                }
            }
        }
        out[i] = acc;
    }
    out
}

fn col_norms_dense<M: DenseMatRef<f64>>(matrix: &M, norm: ScaleNorm) -> Vec<f64> {
    let nrows = matrix.nrows();
    let ncols = matrix.ncols();
    let mut out = vec![0.0; ncols];
    for j in 0..ncols {
        let mut acc = 0.0;
        for i in 0..nrows {
            let v = matrix.get(i, j).abs();
            match norm {
                ScaleNorm::One => acc += v,
                ScaleNorm::Inf => {
                    if v > acc {
                        acc = v;
                    }
                }
            }
        }
        out[j] = acc;
    }
    out
}

fn apply_dense_scale<M: DenseMatMut<f64>>(matrix: &mut M, dir: ScaleDirection, norm: ScaleNorm) {
    let nrows = matrix.nrows();
    let ncols = matrix.ncols();
    match dir {
        ScaleDirection::Row => {
            let norms = row_norms_dense(matrix, norm);
            for i in 0..nrows {
                let scale = norms[i];
                if scale == 0.0 {
                    continue;
                }
                for j in 0..ncols {
                    let val = matrix.get(i, j);
                    matrix.set(i, j, val / scale);
                }
            }
        }
        ScaleDirection::Col => {
            let norms = col_norms_dense(matrix, norm);
            for j in 0..ncols {
                let scale = norms[j];
                if scale == 0.0 {
                    continue;
                }
                for i in 0..nrows {
                    let val = matrix.get(i, j);
                    matrix.set(i, j, val / scale);
                }
            }
        }
        ScaleDirection::Both => {
            apply_dense_scale(matrix, ScaleDirection::Row, norm);
            apply_dense_scale(matrix, ScaleDirection::Col, norm);
        }
    }
}

pub fn apply_dense_transforms<M: DenseMatMut<f64>>(
    label: &str,
    matrix: &mut M,
    opts: &ConditioningOptions,
) -> Result<(), KError> {
    if !opts.is_active() {
        return Ok(());
    }
    opts.validate()?;
    let before = analyze_dense(matrix, opts.tiny_threshold);
    log_conditioning(label, opts);
    log_stats(label, "before", &before);

    let diag_len = matrix.nrows().min(matrix.ncols());
    if opts.fix_diag {
        for i in 0..diag_len {
            let val = matrix.get(i, i);
            if val.abs() <= opts.tiny_threshold {
                let replacement = if val == 0.0 { 1.0 } else { val.signum() };
                matrix.set(i, i, replacement * opts.tiny_threshold);
            }
        }
    }
    if let Some(shift) = opts.shift_diag {
        for i in 0..diag_len {
            let val = matrix.get(i, i);
            matrix.set(i, i, val + shift);
        }
    }
    if let Some(tau) = opts.diag_inject_tau {
        let norms = row_norms_dense(matrix, opts.scale_norm);
        for i in 0..diag_len {
            let val = matrix.get(i, i);
            matrix.set(i, i, val + tau * norms[i]);
        }
    }
    if let Some(scale) = opts.scale {
        apply_dense_scale(matrix, scale, opts.scale_norm);
    }

    let after = analyze_dense(matrix, opts.tiny_threshold);
    log_stats(label, "after", &after);
    Ok(())
}

fn row_norms_csr(a: &CsrMatrix<f64>, norm: ScaleNorm) -> Vec<f64> {
    let nrows = a.nrows();
    let mut out = vec![0.0; nrows];
    let row_ptr = a.row_ptr();
    let values = a.values();
    for i in 0..nrows {
        let mut acc = 0.0;
        for p in row_ptr[i]..row_ptr[i + 1] {
            let v = values[p].abs();
            match norm {
                ScaleNorm::One => acc += v,
                ScaleNorm::Inf => {
                    if v > acc {
                        acc = v;
                    }
                }
            }
        }
        out[i] = acc;
    }
    out
}

fn col_norms_csr(a: &CsrMatrix<f64>, norm: ScaleNorm) -> Vec<f64> {
    let ncols = a.ncols();
    let mut out = vec![0.0; ncols];
    let row_ptr = a.row_ptr();
    let col_idx = a.col_idx();
    let values = a.values();
    for i in 0..a.nrows() {
        for p in row_ptr[i]..row_ptr[i + 1] {
            let j = col_idx[p];
            let v = values[p].abs();
            match norm {
                ScaleNorm::One => out[j] += v,
                ScaleNorm::Inf => {
                    if v > out[j] {
                        out[j] = v;
                    }
                }
            }
        }
    }
    out
}

fn ensure_diag_entries(a: &mut CsrMatrix<f64>) {
    let nrows = a.nrows();
    let ncols = a.ncols();
    let row_ptr = a.row_ptr();
    let col_idx = a.col_idx();
    let values = a.values();

    let mut new_row_ptr = Vec::with_capacity(nrows + 1);
    let mut new_col = Vec::with_capacity(values.len() + nrows);
    let mut new_vals = Vec::with_capacity(values.len() + nrows);
    new_row_ptr.push(0);

    for i in 0..nrows {
        let start = row_ptr[i];
        let end = row_ptr[i + 1];
        let mut inserted = false;
        for (col, val) in col_idx[start..end]
            .iter()
            .copied()
            .zip(values[start..end].iter().copied())
        {
            if !inserted && i < ncols && col > i {
                new_col.push(i);
                new_vals.push(0.0);
                inserted = true;
            }
            if i < ncols && col == i {
                inserted = true;
            }
            new_col.push(col);
            new_vals.push(val);
        }
        if !inserted && i < ncols {
            new_col.push(i);
            new_vals.push(0.0);
        }
        new_row_ptr.push(new_col.len());
    }

    *a = CsrMatrix::from_csr(nrows, ncols, new_row_ptr, new_col, new_vals);
}

fn apply_csr_scale(a: &mut CsrMatrix<f64>, dir: ScaleDirection, norm: ScaleNorm) {
    match dir {
        ScaleDirection::Row => {
            let row_norms = row_norms_csr(a, norm);
            let nrows = a.nrows();
            let row_ptr = a.row_ptr().to_vec();
            let values = a.values_mut();
            for i in 0..nrows {
                let scale = row_norms[i];
                if scale == 0.0 {
                    continue;
                }
                for p in row_ptr[i]..row_ptr[i + 1] {
                    values[p] /= scale;
                }
            }
        }
        ScaleDirection::Col => {
            let col_norms = col_norms_csr(a, norm);
            let col_idx = a.col_idx().to_vec();
            let values = a.values_mut();
            for p in 0..values.len() {
                let scale = col_norms[col_idx[p]];
                if scale == 0.0 {
                    continue;
                }
                values[p] /= scale;
            }
        }
        ScaleDirection::Both => {
            apply_csr_scale(a, ScaleDirection::Row, norm);
            apply_csr_scale(a, ScaleDirection::Col, norm);
        }
    }
}

pub fn apply_csr_transforms(
    label: &str,
    a: &mut CsrMatrix<f64>,
    opts: &ConditioningOptions,
) -> Result<(), KError> {
    if !opts.is_active() {
        return Ok(());
    }
    opts.validate()?;
    log_conditioning(label, opts);
    let before = analyze_csr(a, opts.tiny_threshold);
    log_stats(label, "before", &before);

    let needs_diag = opts.fix_diag || opts.shift_diag.is_some() || opts.diag_inject_tau.is_some();
    if needs_diag {
        ensure_diag_entries(a);
    }
    if opts.fix_diag {
        for i in 0..a.nrows().min(a.ncols()) {
            if let Some(diag) = a.diag_mut(i)
                && diag.abs() <= opts.tiny_threshold {
                    let replacement = if *diag == 0.0 { 1.0 } else { diag.signum() };
                    *diag = replacement * opts.tiny_threshold;
                }
        }
    }
    if let Some(shift) = opts.shift_diag {
        for i in 0..a.nrows().min(a.ncols()) {
            if let Some(diag) = a.diag_mut(i) {
                *diag += shift;
            }
        }
    }
    if let Some(tau) = opts.diag_inject_tau {
        let row_norms = row_norms_csr(a, opts.scale_norm);
        for i in 0..a.nrows().min(a.ncols()) {
            if let Some(diag) = a.diag_mut(i) {
                *diag += tau * row_norms[i];
            }
        }
    }
    if let Some(scale) = opts.scale {
        apply_csr_scale(a, scale, opts.scale_norm);
    }

    let after = analyze_csr(a, opts.tiny_threshold);
    log_stats(label, "after", &after);
    Ok(())
}
