//! Matrix Market file format I/O utilities for Kryst.
//!
//! This module provides functions to read and write matrices and vectors in the
//! [Matrix Market exchange format](https://math.nist.gov/MatrixMarket/formats.html).
//! It supports both coordinate (sparse) and array (dense) formats, and can
//! convert the data into Kryst's matrix and vector types.
//!
//! # Supported Formats
//!
//! - **Coordinate format**: Sparse matrices with (row, col, value) triplets
//! - **Array format**: Dense matrices/vectors stored column-wise
//! - **Pattern format**: Binary matrices (values default to 1.0)
//!
//! # Example Usage
//!
//! ```rust,ignore
//! use kryst::utils::matrix_market::{read_matrix_market, MatrixMarketData};
//! use kryst::matrix::sparse::CsrMatrix;
//!
//! // Read a Matrix Market file
//! let data = read_matrix_market("matrix.mtx")?;
//!
//! // Convert to Kryst sparse matrix
//! let matrix = data.to_csr_matrix()?;
//!
//! // Or extract raw data
//! let (rows, cols, nnz, row_indices, col_indices, values, imag) = data.into_triplets();
//! assert!(imag.is_none());
//! ```

use crate::algebra::prelude::*;
use crate::algebra::scalar::copy_scalar_to_real_in;
use crate::error::KError;
use crate::matrix::sparse::CsrMatrix;
use faer::Mat;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

/// Helper function to map parse errors to KError.
fn parse_error<E>(err: E) -> KError
where
    E: std::fmt::Debug,
{
    KError::SolveError(format!("Parse error: {err:?}"))
}

/// Supported numeric field types in a Matrix Market file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixMarketField {
    Real,
    Integer,
    Complex,
    Pattern,
}

/// Symmetry metadata for Matrix Market data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixMarketSymmetry {
    General,
    Symmetric,
    SkewSymmetric,
    Hermitian,
}

impl MatrixMarketSymmetry {
    fn as_str(self) -> &'static str {
        match self {
            MatrixMarketSymmetry::General => "general",
            MatrixMarketSymmetry::Symmetric => "symmetric",
            MatrixMarketSymmetry::SkewSymmetric => "skew-symmetric",
            MatrixMarketSymmetry::Hermitian => "hermitian",
        }
    }
}

/// Matrix Market data representation.
///
/// This struct holds the raw data from a Matrix Market file and provides
/// methods to convert it into various Kryst matrix formats.
#[derive(Debug, Clone)]
pub struct MatrixMarketData {
    /// Number of rows in the matrix
    pub rows: usize,
    /// Number of columns in the matrix
    pub cols: usize,
    /// Number of non-zero entries (may be different from triplets.len() for symmetric matrices)
    pub nonzeros: usize,
    /// Row indices (0-based)
    pub row_indices: Vec<usize>,
    /// Column indices (0-based)
    pub col_indices: Vec<usize>,
    /// Real component of the matrix values
    pub values: Vec<f64>,
    /// Imaginary component of the matrix values (if complex field)
    pub imag_values: Option<Vec<f64>>,
    /// The numeric field type of the original Matrix Market file
    pub field_type: MatrixMarketField,
    /// Symmetry metadata as declared in the header
    pub symmetry: MatrixMarketSymmetry,
    /// Whether the matrix is in coordinate format (sparse) or array format (dense)
    pub is_coordinate: bool,
}

impl MatrixMarketData {
    /// Create a new MatrixMarketData instance.
    pub fn new(
        rows: usize,
        cols: usize,
        nonzeros: usize,
        row_indices: Vec<usize>,
        col_indices: Vec<usize>,
        values: Vec<f64>,
        symmetry: MatrixMarketSymmetry,
        is_coordinate: bool,
    ) -> Self {
        Self::with_field(
            rows,
            cols,
            nonzeros,
            row_indices,
            col_indices,
            values,
            None,
            MatrixMarketField::Real,
            symmetry,
            is_coordinate,
        )
    }

    /// Construct a `MatrixMarketData` instance with explicit numeric field metadata.
    pub fn with_field(
        rows: usize,
        cols: usize,
        nonzeros: usize,
        row_indices: Vec<usize>,
        col_indices: Vec<usize>,
        values: Vec<f64>,
        imag_values: Option<Vec<f64>>,
        field_type: MatrixMarketField,
        symmetry: MatrixMarketSymmetry,
        is_coordinate: bool,
    ) -> Self {
        Self {
            rows,
            cols,
            nonzeros,
            row_indices,
            col_indices,
            values,
            imag_values,
            field_type,
            symmetry,
            is_coordinate,
        }
    }

    /// Convert to CSR format triplets (rows, cols, nnz, row_indices, col_indices, values).
    ///
    /// Returns the real component of the values along with the optional imaginary
    /// component for complex matrices. Callers that only understand real data can
    /// ignore the imaginary slice.
    pub fn into_triplets(
        self,
    ) -> (
        usize,
        usize,
        usize,
        Vec<usize>,
        Vec<usize>,
        Vec<f64>,
        Option<Vec<f64>>,
    ) {
        (
            self.rows,
            self.cols,
            self.nonzeros,
            self.row_indices,
            self.col_indices,
            self.values,
            self.imag_values,
        )
    }

    /// Convert to a Kryst CSR matrix.
    pub fn to_csr_matrix(&self) -> Result<CsrMatrix<f64>, KError> {
        if self.field_type == MatrixMarketField::Complex {
            return Err(KError::SolveError(
                "Complex Matrix Market data cannot be converted to real CSR; use to_csr_matrix_scalar instead"
                    .to_string(),
            ));
        }

        let scalar = self.to_csr_matrix_scalar()?;
        let mut values = vec![0.0; scalar.nnz()];
        copy_scalar_to_real_in(scalar.values(), &mut values);
        let row_ptr = scalar.row_ptr().to_vec();
        let col_idx = scalar.col_idx().to_vec();
        Ok(CsrMatrix::from_csr(
            scalar.nrows(),
            scalar.ncols(),
            row_ptr,
            col_idx,
            values,
        ))
    }

    /// Convert to a Kryst CSR matrix with entries of type `S`.
    pub fn to_csr_matrix_scalar(&self) -> Result<CsrMatrix<S>, KError> {
        if self.row_indices.len() != self.col_indices.len()
            || self.row_indices.len() != self.values.len()
        {
            return Err(KError::SolveError(
                "Inconsistent triplet lengths".to_string(),
            ));
        }

        if let Some(imag) = &self.imag_values
            && imag.len() != self.values.len()
        {
            return Err(KError::SolveError(
                "Inconsistent imaginary component length".to_string(),
            ));
        }

        let values = self.values_as_scalars()?;
        let mut triplets: Vec<_> = self
            .row_indices
            .iter()
            .zip(self.col_indices.iter())
            .enumerate()
            .map(|(idx, (&r, &c))| (r, c, values[idx]))
            .collect();

        triplets.sort_by_key(|&(r, c, _)| (r, c));

        triplets = match self.symmetry {
            MatrixMarketSymmetry::General => triplets,
            MatrixMarketSymmetry::Symmetric => {
                let mut expanded = Vec::with_capacity(triplets.len() * 2);
                for &(r, c, v) in &triplets {
                    expanded.push((r, c, v));
                    if r != c {
                        expanded.push((c, r, v));
                    }
                }
                expanded.sort_by_key(|&(r, c, _)| (r, c));
                expanded
            }
            MatrixMarketSymmetry::SkewSymmetric => {
                let mut expanded = Vec::with_capacity(triplets.len() * 2);
                for &(r, c, v) in &triplets {
                    if r == c {
                        if v != S::zero() {
                            return Err(KError::SolveError(
                                "Diagonal entries must be zero for skew-symmetric matrices"
                                    .to_string(),
                            ));
                        }
                        expanded.push((r, c, v));
                    } else {
                        expanded.push((r, c, v));
                        expanded.push((c, r, -v));
                    }
                }
                expanded.sort_by_key(|&(r, c, _)| (r, c));
                expanded
            }
            MatrixMarketSymmetry::Hermitian => {
                let mut expanded = Vec::with_capacity(triplets.len() * 2);
                for &(r, c, v) in &triplets {
                    expanded.push((r, c, v));
                    if r != c {
                        expanded.push((c, r, v.conj()));
                    } else if v.imag().abs() > 1e-12 {
                        return Err(KError::SolveError(
                            "Diagonal entries of Hermitian matrices must be real".to_string(),
                        ));
                    }
                }
                expanded.sort_by_key(|&(r, c, _)| (r, c));
                expanded
            }
        };

        let mut row_ptr = vec![0; self.rows + 1];
        for &(r, _, _) in &triplets {
            if r >= self.rows {
                return Err(KError::SolveError(format!("Row index {r} out of bounds")));
            }
            row_ptr[r + 1] += 1;
        }

        for i in 1..=self.rows {
            row_ptr[i] += row_ptr[i - 1];
        }

        let col_idx: Vec<usize> = triplets.iter().map(|&(_, c, _)| c).collect();
        let values: Vec<S> = triplets.iter().map(|&(_, _, v)| v).collect();

        Ok(CsrMatrix::from_csr(
            self.rows, self.cols, row_ptr, col_idx, values,
        ))
    }

    /// Convert to a dense vector (for right-hand sides).
    ///
    /// This assumes the matrix is a column vector (cols == 1).
    pub fn to_vector(&self) -> Result<Vec<f64>, KError> {
        if self.field_type == MatrixMarketField::Complex {
            return Err(KError::SolveError(
                "Complex Matrix Market data cannot be converted to real vector; use to_vector_scalar instead"
                    .to_string(),
            ));
        }

        if self.cols != 1 {
            return Err(KError::SolveError(format!(
                "Cannot convert {}x{} matrix to vector (must be Nx1)",
                self.rows, self.cols
            )));
        }

        let mut vector = vec![0.0; self.rows];
        for ((idx, &r), &_c) in self
            .row_indices
            .iter()
            .enumerate()
            .zip(self.col_indices.iter())
        {
            if r >= self.rows {
                return Err(KError::SolveError(format!("Row index {r} out of bounds")));
            }
            vector[r] = self.values[idx];
        }

        Ok(vector)
    }

    /// Convert to a dense vector of Kryst scalars.
    pub fn to_vector_scalar(&self) -> Result<Vec<S>, KError> {
        if self.cols != 1 {
            return Err(KError::SolveError(format!(
                "Cannot convert {}x{} matrix to vector (must be Nx1)",
                self.rows, self.cols
            )));
        }

        let mut vector = vec![S::zero(); self.rows];
        for ((idx, &r), &_c) in self
            .row_indices
            .iter()
            .enumerate()
            .zip(self.col_indices.iter())
        {
            if r >= self.rows {
                return Err(KError::SolveError(format!("Row index {r} out of bounds")));
            }
            vector[r] = self.value_as_scalar(idx)?;
        }

        Ok(vector)
    }

    /// Convert to a dense matrix of Kryst scalars.
    pub fn to_dense_matrix_scalar(&self) -> Result<Mat<S>, KError> {
        if self.is_coordinate {
            return Ok(self.to_csr_matrix_scalar()?.to_dense());
        }

        let mut dense = Mat::zeros(self.rows, self.cols);
        for ((idx, &row), &col) in self
            .row_indices
            .iter()
            .enumerate()
            .zip(self.col_indices.iter())
        {
            if row >= self.rows || col >= self.cols {
                return Err(KError::SolveError(format!(
                    "Index ({row}, {col}) out of bounds for {}x{} matrix",
                    self.rows, self.cols
                )));
            }
            dense[(row, col)] = self.value_as_scalar(idx)?;
        }

        Ok(dense)
    }

    fn values_as_scalars(&self) -> Result<Vec<S>, KError> {
        if self.row_indices.len() != self.values.len() {
            return Err(KError::SolveError("Inconsistent value length".to_string()));
        }
        (0..self.values.len())
            .map(|idx| self.value_as_scalar(idx))
            .collect()
    }

    fn value_as_scalar(&self, idx: usize) -> Result<S, KError> {
        match self.field_type {
            MatrixMarketField::Real | MatrixMarketField::Integer => {
                Ok(S::from_real(self.values[idx]))
            }
            MatrixMarketField::Pattern => Ok(S::one()),
            MatrixMarketField::Complex => {
                let imag_values = self.imag_values.as_ref().ok_or_else(|| {
                    KError::SolveError(
                        "Missing imaginary component for complex Matrix Market data".to_string(),
                    )
                })?;
                Ok(S::from_parts(self.values[idx], imag_values[idx]))
            }
        }
    }
}

fn split_scalar_components(
    values: &[S],
    force_complex: bool,
) -> (Vec<f64>, Option<Vec<f64>>, MatrixMarketField) {
    let real: Vec<f64> = values.iter().map(|v| v.real()).collect();

    #[cfg(feature = "complex")]
    {
        let imag: Vec<f64> = values.iter().map(|v| v.imag()).collect();
        if force_complex {
            return (real, Some(imag), MatrixMarketField::Complex);
        }

        if imag.iter().any(|&im| im != 0.0) {
            (real, Some(imag), MatrixMarketField::Complex)
        } else {
            (real, None, MatrixMarketField::Real)
        }
    }

    #[cfg(not(feature = "complex"))]
    {
        let _ = force_complex;
        (real, None, MatrixMarketField::Real)
    }
}

fn validate_array_symmetry(
    values: &[S],
    rows: usize,
    cols: usize,
    symmetry: MatrixMarketSymmetry,
) -> Result<(), KError> {
    if matches!(
        symmetry,
        MatrixMarketSymmetry::Symmetric
            | MatrixMarketSymmetry::SkewSymmetric
            | MatrixMarketSymmetry::Hermitian
    ) && rows != cols
    {
        return Err(KError::SolveError(
            "Symmetric, skew-symmetric, and Hermitian matrices must be square".to_string(),
        ));
    }

    const TOL: f64 = 1e-12;

    match symmetry {
        MatrixMarketSymmetry::General => Ok(()),
        MatrixMarketSymmetry::Symmetric => {
            for col in 0..cols {
                for row in 0..rows {
                    if row <= col {
                        continue;
                    }
                    let idx = col * rows + row;
                    let mirror_idx = row * rows + col;
                    let diff = values[idx] - values[mirror_idx];
                    if diff.abs() > TOL {
                        return Err(KError::SolveError(format!(
                            "Entries ({row}, {col}) and ({col}, {row}) must be equal for symmetric matrices"
                        )));
                    }
                }
            }
            Ok(())
        }
        MatrixMarketSymmetry::SkewSymmetric => {
            for i in 0..rows {
                let diag = values[i * rows + i];
                if diag.abs() > TOL {
                    return Err(KError::SolveError(
                        "Diagonal entries must be zero for skew-symmetric matrices".to_string(),
                    ));
                }
            }

            for col in 0..cols {
                for row in 0..rows {
                    if row <= col {
                        continue;
                    }
                    let idx = col * rows + row;
                    let mirror_idx = row * rows + col;
                    let sum = values[idx] + values[mirror_idx];
                    if sum.abs() > TOL {
                        return Err(KError::SolveError(format!(
                            "Entries ({row}, {col}) and ({col}, {row}) must be negatives for skew-symmetric matrices"
                        )));
                    }
                }
            }
            Ok(())
        }
        MatrixMarketSymmetry::Hermitian => {
            #[cfg(not(feature = "complex"))]
            {
                let _ = values;
                let _ = TOL;
                Err(KError::SolveError(
                    "Writing Hermitian matrices requires the `complex` feature".to_string(),
                ))
            }

            #[cfg(feature = "complex")]
            {
                for i in 0..rows {
                    let diag = values[i * rows + i];
                    if diag.imag().abs() > TOL {
                        return Err(KError::SolveError(
                            "Diagonal entries of Hermitian matrices must be real".to_string(),
                        ));
                    }
                }

                for col in 0..cols {
                    for row in 0..rows {
                        if row <= col {
                            continue;
                        }
                        let idx = col * rows + row;
                        let mirror_idx = row * rows + col;
                        let diff = values[idx] - values[mirror_idx].conj();
                        if diff.abs() > TOL {
                            return Err(KError::SolveError(format!(
                                "Entries ({row}, {col}) and ({col}, {row}) must be conjugates for Hermitian matrices"
                            )));
                        }
                    }
                }
                Ok(())
            }
        }
    }
}

/// Reads a Matrix Market file and returns the data as a MatrixMarketData struct.
pub fn read_matrix_market<P: AsRef<Path>>(file_path: P) -> Result<MatrixMarketData, KError> {
    let file = File::open(&file_path).map_err(|e| {
        KError::SolveError(format!(
            "Failed to open file {:?}: {}",
            file_path.as_ref(),
            e
        ))
    })?;
    let reader = BufReader::new(file);

    let mut lines = reader.lines();
    let header = lines
        .next()
        .ok_or_else(|| KError::SolveError("Empty file".to_string()))?
        .map_err(|e| KError::SolveError(format!("Failed to read header: {e}")))?;

    // Check the Matrix Market banner
    if !header.starts_with("%%MatrixMarket") {
        return Err(KError::SolveError(
            "Invalid Matrix Market banner".to_string(),
        ));
    }

    // Parse header format information
    let header_parts: Vec<&str> = header.split_whitespace().collect();
    if header_parts.len() < 5 {
        return Err(KError::SolveError(format!(
            "Invalid Matrix Market header format: expected at least 5 parts, got {}: {}",
            header_parts.len(),
            header
        )));
    }

    // Format: %%MatrixMarket matrix <format> <field> <symmetry>
    // Example: %%MatrixMarket matrix coordinate real general
    let is_coordinate = header_parts[2].eq_ignore_ascii_case("coordinate");
    let is_array = header_parts[2].eq_ignore_ascii_case("array");
    let symmetry = match header_parts[4].to_ascii_lowercase().as_str() {
        "general" => MatrixMarketSymmetry::General,
        "symmetric" => MatrixMarketSymmetry::Symmetric,
        "skew-symmetric" => MatrixMarketSymmetry::SkewSymmetric,
        "hermitian" => MatrixMarketSymmetry::Hermitian,
        other => {
            return Err(KError::SolveError(format!(
                "Unsupported Matrix Market symmetry type: {other}"
            )));
        }
    };

    let field_type = match header_parts[3].to_ascii_lowercase().as_str() {
        "real" => MatrixMarketField::Real,
        "integer" => MatrixMarketField::Integer,
        "complex" => {
            #[cfg(not(feature = "complex"))]
            {
                return Err(KError::SolveError(
                    "Complex Matrix Market files require the `complex` feature".to_string(),
                ));
            }
            #[cfg(feature = "complex")]
            {
                MatrixMarketField::Complex
            }
        }
        "pattern" => MatrixMarketField::Pattern,
        other => {
            return Err(KError::SolveError(format!(
                "Unsupported Matrix Market field type: {other}"
            )));
        }
    };

    if !is_coordinate && !is_array {
        return Err(KError::SolveError(format!(
            "Unsupported Matrix Market format: {}",
            header_parts[3]
        )));
    }

    if matches!(field_type, MatrixMarketField::Pattern) && !is_coordinate {
        return Err(KError::SolveError(
            "Pattern matrices must be in coordinate format".to_string(),
        ));
    }

    // Skip comments
    let size_line = lines
        .find(|line| {
            if let Ok(content) = line {
                !content.starts_with('%')
            } else {
                false
            }
        })
        .ok_or_else(|| KError::SolveError("Missing size information".to_string()))?
        .map_err(|e| KError::SolveError(format!("Failed to read size line: {e}")))?;

    // Parse size line
    let size_parts: Vec<&str> = size_line.split_whitespace().collect();
    let (rows, cols, declared_nonzeros) = if is_array {
        // Array format: rows cols
        if size_parts.len() != 2 {
            return Err(KError::SolveError(
                "Invalid size format for array".to_string(),
            ));
        }
        let rows = size_parts[0].parse::<usize>().map_err(parse_error)?;
        let cols = size_parts[1].parse::<usize>().map_err(parse_error)?;
        (rows, cols, rows * cols)
    } else {
        // Coordinate format: rows cols nonzeros
        if size_parts.len() != 3 {
            return Err(KError::SolveError(
                "Invalid size format for coordinate".to_string(),
            ));
        }
        let rows = size_parts[0].parse::<usize>().map_err(parse_error)?;
        let cols = size_parts[1].parse::<usize>().map_err(parse_error)?;
        let nonzeros = size_parts[2].parse::<usize>().map_err(parse_error)?;
        (rows, cols, nonzeros)
    };

    if matches!(
        symmetry,
        MatrixMarketSymmetry::SkewSymmetric | MatrixMarketSymmetry::Hermitian
    ) && rows != cols
    {
        return Err(KError::SolveError(
            "Skew-symmetric and Hermitian matrices must be square".to_string(),
        ));
    }

    let mut row_indices = Vec::new();
    let mut col_indices = Vec::new();
    let mut values = Vec::new();
    let mut imag_values = match field_type {
        MatrixMarketField::Complex => Some(Vec::new()),
        _ => None,
    };

    if is_array {
        // Parse dense array data (column-major order in Matrix Market)
        let mut entry_count = 0;
        for line in lines {
            let line = line.map_err(|e| KError::SolveError(format!("Failed to read line: {e}")))?;
            let trimmed_line = line.trim();
            if trimmed_line.is_empty() {
                continue;
            }

            match field_type {
                MatrixMarketField::Complex => {
                    let mut parts = trimmed_line.split_whitespace();
                    let re = parts
                        .next()
                        .ok_or_else(|| {
                            KError::SolveError(
                                "Missing real part for complex Matrix Market entry".to_string(),
                            )
                        })?
                        .parse::<f64>()
                        .map_err(parse_error)?;
                    let im = parts
                        .next()
                        .ok_or_else(|| {
                            KError::SolveError(
                                "Missing imaginary part for complex Matrix Market entry"
                                    .to_string(),
                            )
                        })?
                        .parse::<f64>()
                        .map_err(parse_error)?;

                    values.push(re);
                    if let Some(imag) = imag_values.as_mut() {
                        imag.push(im);
                    }
                }
                MatrixMarketField::Pattern => {
                    values.push(1.0);
                }
                _ => {
                    let value = trimmed_line.parse::<f64>().map_err(parse_error)?;
                    values.push(value);
                }
            }

            // Matrix Market array format is column-major
            let col = entry_count / rows;
            let row = entry_count % rows;

            if col >= cols || row >= rows {
                return Err(KError::SolveError(format!(
                    "Entry index ({row}, {col}) out of bounds for {rows}x{cols} matrix"
                )));
            }

            row_indices.push(row);
            col_indices.push(col);
            entry_count += 1;

            // Stop if we've read all expected entries
            if entry_count >= declared_nonzeros {
                break;
            }
        }
    } else {
        // Parse sparse coordinate data
        for line in lines {
            let line = line.map_err(|e| KError::SolveError(format!("Failed to read line: {e}")))?;
            let trimmed_line = line.trim();
            if trimmed_line.is_empty() {
                continue;
            }

            let parts: Vec<&str> = trimmed_line.split_whitespace().collect();
            if parts.len() < 2 {
                continue; // Skip invalid lines
            }

            let row = parts[0].parse::<usize>().map_err(parse_error)? - 1; // Convert to 0-based
            let col = parts[1].parse::<usize>().map_err(parse_error)? - 1; // Convert to 0-based
            let (value, imag) = match field_type {
                MatrixMarketField::Pattern => (1.0, 0.0),
                MatrixMarketField::Complex => {
                    if parts.len() < 4 {
                        return Err(KError::SolveError(
                            "Complex coordinate entry must provide real and imaginary parts"
                                .to_string(),
                        ));
                    }
                    let re = parts[2].parse::<f64>().map_err(parse_error)?;
                    let im = parts[3].parse::<f64>().map_err(parse_error)?;
                    (re, im)
                }
                _ => {
                    if parts.len() > 2 {
                        (parts[2].parse::<f64>().map_err(parse_error)?, 0.0)
                    } else {
                        (1.0, 0.0)
                    }
                }
            };

            if row >= rows || col >= cols {
                return Err(KError::SolveError(format!(
                    "Entry index ({row}, {col}) out of bounds"
                )));
            }

            row_indices.push(row);
            col_indices.push(col);
            values.push(value);
            if let Some(imag_vec) = imag_values.as_mut() {
                imag_vec.push(imag);
            }
        }
    }

    let _actual_nonzeros = row_indices.len();

    Ok(MatrixMarketData::with_field(
        rows,
        cols,
        declared_nonzeros,
        row_indices,
        col_indices,
        values,
        imag_values,
        field_type,
        symmetry,
        is_coordinate,
    ))
}

/// Writes matrix data to a Matrix Market file.
pub fn write_matrix_market<P: AsRef<Path>>(
    file_path: P,
    data: &MatrixMarketData,
) -> Result<(), KError> {
    let mut file = File::create(&file_path).map_err(|e| {
        KError::SolveError(format!(
            "Failed to create file {:?}: {}",
            file_path.as_ref(),
            e
        ))
    })?;

    // Write header
    let format_type = if data.is_coordinate {
        "coordinate"
    } else {
        "array"
    };
    let symmetry = data.symmetry.as_str();
    let field = match data.field_type {
        MatrixMarketField::Real | MatrixMarketField::Integer => "real",
        MatrixMarketField::Complex => "complex",
        MatrixMarketField::Pattern => "pattern",
    };
    writeln!(
        file,
        "%%MatrixMarket matrix {format_type} {field} {symmetry}"
    )
    .map_err(|e| KError::SolveError(format!("Failed to write header: {e}")))?;

    // Write size information
    if data.is_coordinate {
        writeln!(file, "{} {} {}", data.rows, data.cols, data.nonzeros)
    } else {
        writeln!(file, "{} {}", data.rows, data.cols)
    }
    .map_err(|e| KError::SolveError(format!("Failed to write size: {e}")))?;

    // Write data
    if data.is_coordinate {
        // Write coordinate format
        for (idx, (&row, &col)) in data
            .row_indices
            .iter()
            .zip(data.col_indices.iter())
            .enumerate()
        {
            match data.field_type {
                MatrixMarketField::Complex => {
                    let imag_values = data.imag_values.as_ref().ok_or_else(|| {
                        KError::SolveError(
                            "Missing imaginary values when writing complex Matrix Market data"
                                .to_string(),
                        )
                    })?;
                    writeln!(
                        file,
                        "{} {} {} {}",
                        row + 1,
                        col + 1,
                        data.values[idx],
                        imag_values[idx]
                    )
                    .map_err(|e| KError::SolveError(format!("Failed to write entry: {e}")))?;
                }
                MatrixMarketField::Pattern => {
                    writeln!(file, "{} {}", row + 1, col + 1)
                        .map_err(|e| KError::SolveError(format!("Failed to write entry: {e}")))?;
                }
                _ => {
                    writeln!(file, "{} {} {}", row + 1, col + 1, data.values[idx])
                        .map_err(|e| KError::SolveError(format!("Failed to write entry: {e}")))?;
                }
            }
        }
    } else {
        // Write array format (column-major order)
        // First, reorganize data into column-major order if needed
        let mut dense_re = vec![0.0; data.rows * data.cols];
        let mut dense_im = match data.field_type {
            MatrixMarketField::Complex => Some(vec![0.0; data.rows * data.cols]),
            _ => None,
        };
        for (idx, (&row, &col)) in data
            .row_indices
            .iter()
            .zip(data.col_indices.iter())
            .enumerate()
        {
            let index = col * data.rows + row; // column-major indexing
            dense_re[index] = data.values[idx];
            if let Some(imag) = dense_im.as_mut() {
                let imag_values = data.imag_values.as_ref().ok_or_else(|| {
                    KError::SolveError(
                        "Missing imaginary values when writing complex Matrix Market data"
                            .to_string(),
                    )
                })?;
                imag[index] = imag_values[idx];
            }
        }

        match dense_im {
            Some(imag) => {
                for (re, im) in dense_re.iter().zip(imag.iter()) {
                    writeln!(file, "{re} {im}")
                        .map_err(|e| KError::SolveError(format!("Failed to write value: {e}")))?;
                }
            }
            None => {
                for &value in &dense_re {
                    writeln!(file, "{value}")
                        .map_err(|e| KError::SolveError(format!("Failed to write value: {e}")))?;
                }
            }
        }
    }

    Ok(())
}

/// Convenience function to write matrix data in coordinate format.
pub fn write_matrix_market_coordinate<P: AsRef<Path>>(
    file_path: P,
    rows: usize,
    cols: usize,
    row_indices: &[usize],
    col_indices: &[usize],
    values: &[f64],
    symmetry: MatrixMarketSymmetry,
) -> Result<(), KError> {
    let data = MatrixMarketData::new(
        rows,
        cols,
        row_indices.len(),
        row_indices.to_vec(),
        col_indices.to_vec(),
        values.to_vec(),
        symmetry,
        true, // coordinate format
    );
    write_matrix_market(file_path, &data)
}

/// Convenience function to write scalar matrix data in coordinate format.
pub fn write_matrix_market_coordinate_scalar<P: AsRef<Path>>(
    file_path: P,
    rows: usize,
    cols: usize,
    row_indices: &[usize],
    col_indices: &[usize],
    values: &[S],
    symmetry: MatrixMarketSymmetry,
) -> Result<(), KError> {
    if row_indices.len() != col_indices.len() || row_indices.len() != values.len() {
        return Err(KError::SolveError(
            "Row, column, and value arrays must have the same length".to_string(),
        ));
    }

    for &r in row_indices {
        if r >= rows {
            return Err(KError::SolveError(format!(
                "Row index {r} out of bounds for matrix with {rows} rows"
            )));
        }
    }

    for &c in col_indices {
        if c >= cols {
            return Err(KError::SolveError(format!(
                "Column index {c} out of bounds for matrix with {cols} columns"
            )));
        }
    }

    if matches!(symmetry, MatrixMarketSymmetry::SkewSymmetric) {
        for ((&r, &c), &v) in row_indices
            .iter()
            .zip(col_indices.iter())
            .zip(values.iter())
        {
            if r == c && v != S::zero() {
                return Err(KError::SolveError(
                    "Diagonal entries must be zero for skew-symmetric matrices".to_string(),
                ));
            }
        }
    }

    #[cfg(not(feature = "complex"))]
    if matches!(symmetry, MatrixMarketSymmetry::Hermitian) {
        return Err(KError::SolveError(
            "Writing Hermitian matrices requires the `complex` feature".to_string(),
        ));
    }

    if matches!(symmetry, MatrixMarketSymmetry::Hermitian) {
        for ((&r, &c), &v) in row_indices
            .iter()
            .zip(col_indices.iter())
            .zip(values.iter())
        {
            if r == c && v.imag().abs() > 1e-12 {
                return Err(KError::SolveError(
                    "Diagonal entries of Hermitian matrices must be real".to_string(),
                ));
            }
        }
    }

    let force_complex = matches!(symmetry, MatrixMarketSymmetry::Hermitian);
    let (values, imag_values, field_type) = split_scalar_components(values, force_complex);

    let data = MatrixMarketData::with_field(
        rows,
        cols,
        row_indices.len(),
        row_indices.to_vec(),
        col_indices.to_vec(),
        values,
        imag_values,
        field_type,
        symmetry,
        true,
    );

    write_matrix_market(file_path, &data)
}

/// Convenience function to write scalar matrix data in array (dense) format.
///
/// The input `values` slice is interpreted in column-major order to match the
/// Matrix Market array convention.
pub fn write_matrix_market_array_scalar<P: AsRef<Path>>(
    file_path: P,
    rows: usize,
    cols: usize,
    values: &[S],
    symmetry: MatrixMarketSymmetry,
) -> Result<(), KError> {
    if values.len() != rows * cols {
        return Err(KError::SolveError(format!(
            "Expected {} entries for a {rows}x{cols} matrix, found {}",
            rows * cols,
            values.len()
        )));
    }

    validate_array_symmetry(values, rows, cols, symmetry)?;

    let force_complex = matches!(symmetry, MatrixMarketSymmetry::Hermitian);
    let (values_re, imag_values, field_type) = split_scalar_components(values, force_complex);

    let mut row_indices = Vec::with_capacity(values.len());
    let mut col_indices = Vec::with_capacity(values.len());
    for (idx, _) in values.iter().enumerate() {
        let col = idx / rows;
        let row = idx % rows;
        row_indices.push(row);
        col_indices.push(col);
    }

    let data = MatrixMarketData::with_field(
        rows,
        cols,
        values.len(),
        row_indices,
        col_indices,
        values_re,
        imag_values,
        field_type,
        symmetry,
        false,
    );

    write_matrix_market(file_path, &data)
}

/// Convenience function to write vector data in array format.
pub fn write_vector_market<P: AsRef<Path>>(file_path: P, vector: &[f64]) -> Result<(), KError> {
    let row_indices: Vec<usize> = (0..vector.len()).collect();
    let col_indices = vec![0; vector.len()];

    let data = MatrixMarketData::new(
        vector.len(),
        1,
        vector.len(),
        row_indices,
        col_indices,
        vector.to_vec(),
        MatrixMarketSymmetry::General,
        false, // array format
    );
    write_matrix_market(file_path, &data)
}

/// Convenience function to write vector data containing Kryst scalars.
///
/// When the `complex` feature is enabled the output automatically upgrades to a
/// complex Matrix Market file if any entry has a non-zero imaginary part.
pub fn write_vector_market_scalar<P: AsRef<Path>>(
    file_path: P,
    vector: &[S],
) -> Result<(), KError> {
    let row_indices: Vec<usize> = (0..vector.len()).collect();
    let col_indices = vec![0; vector.len()];
    let (values, imag_values, field_type) = split_scalar_components(vector, false);

    let data = MatrixMarketData::with_field(
        vector.len(),
        1,
        vector.len(),
        row_indices,
        col_indices,
        values,
        imag_values,
        field_type,
        MatrixMarketSymmetry::General,
        false,
    );

    write_matrix_market(file_path, &data)
}

#[cfg(test)]
mod tests {
    use super::*;

    const MATRIX_FILE: &str = "examples/e05r0000/e05r0000.mtx";
    const RHS_FILE: &str = "examples/e05r0000/e05r0000_rhs1.mtx";
    const OUTPUT_FILE_COORD: &str = "test_output_coord.mtx";
    const OUTPUT_FILE_ARRAY: &str = "test_output_array.mtx";
    const OUTPUT_FILE_VECTOR: &str = "test_output_vector.mtx";
    const OUTPUT_FILE_VECTOR_SCALAR_REAL: &str = "test_output_vector_scalar_real.mtx";
    #[cfg(feature = "complex")]
    const OUTPUT_FILE_VECTOR_SCALAR_COMPLEX: &str = "test_output_vector_scalar_complex.mtx";
    const OUTPUT_FILE_COORD_SCALAR_REAL: &str = "test_output_coord_scalar_real.mtx";
    #[cfg(feature = "complex")]
    const OUTPUT_FILE_COORD_SCALAR_COMPLEX: &str = "test_output_coord_scalar_complex.mtx";
    const OUTPUT_FILE_ARRAY_SCALAR_REAL: &str = "test_output_array_scalar_real.mtx";
    #[cfg(feature = "complex")]
    const OUTPUT_FILE_ARRAY_SCALAR_COMPLEX: &str = "test_output_array_scalar_complex.mtx";

    fn temp_path(name: &str) -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().expect("create temp directory");
        let path = dir.path().join(name);
        (dir, path)
    }

    #[test]
    fn test_read_sparse_matrix_market() {
        // Skip test if example file doesn't exist
        if !std::path::Path::new(MATRIX_FILE).exists() {
            println!(
                "Skipping test_read_sparse_matrix_market: {} not found",
                MATRIX_FILE
            );
            return;
        }

        let data =
            read_matrix_market(MATRIX_FILE).expect("Failed to read sparse matrix market file");

        // Assertions based on known metadata
        assert_eq!(data.rows, 236, "Unexpected number of rows");
        assert_eq!(data.cols, 236, "Unexpected number of columns");
        assert_eq!(data.nonzeros, 5856, "Unexpected number of non-zero entries");
        assert!(data.is_coordinate, "Should be coordinate format");

        // Validate the first entry
        assert_eq!(data.row_indices[0], 6, "First row index mismatch");
        assert_eq!(data.col_indices[0], 0, "First column index mismatch");
        assert!(
            (data.values[0] - (-5.3333331478961e-01)).abs() < 1e-12,
            "First value mismatch: expected {}, got {}",
            -5.3333331478961e-01,
            data.values[0]
        );
    }

    #[test]
    fn test_read_dense_matrix_market() {
        // Skip test if example file doesn't exist
        if !std::path::Path::new(RHS_FILE).exists() {
            println!(
                "Skipping test_read_dense_matrix_market: {} not found",
                RHS_FILE
            );
            return;
        }

        let data = read_matrix_market(RHS_FILE).expect("Failed to read dense matrix market file");

        // Assertions based on known metadata
        assert_eq!(data.rows, 236, "Unexpected number of rows");
        assert_eq!(data.cols, 1, "Unexpected number of columns");
        assert_eq!(data.nonzeros, 236, "Unexpected number of non-zero entries");
        assert!(!data.is_coordinate, "Should be array format");

        // Validate the first entry
        assert_eq!(data.row_indices[0], 0, "First row index mismatch");
        assert_eq!(data.col_indices[0], 0, "First column index mismatch");
        assert!((data.values[0] - 0.0).abs() < 1e-12, "First value mismatch");
    }

    #[test]
    fn test_to_csr_matrix() {
        // Skip test if example file doesn't exist, create test data instead
        let data = if std::path::Path::new(MATRIX_FILE).exists() {
            read_matrix_market(MATRIX_FILE).expect("Failed to read matrix file")
        } else {
            // Create test data
            MatrixMarketData::new(
                3,
                3,
                4,
                vec![0, 1, 2, 0],
                vec![0, 1, 2, 2],
                vec![1.0, 2.0, 3.0, 4.0],
                MatrixMarketSymmetry::General,
                true,
            )
        };

        let csr_matrix = data
            .to_csr_matrix()
            .expect("Failed to convert to CSR matrix");

        assert_eq!(csr_matrix.nrows(), data.rows);
        assert_eq!(csr_matrix.ncols(), data.cols);
    }

    #[test]
    fn test_to_vector() {
        // Skip test if example file doesn't exist, create test data instead
        let data = if std::path::Path::new(RHS_FILE).exists() {
            read_matrix_market(RHS_FILE).expect("Failed to read RHS file")
        } else {
            // Create test vector data
            let row_indices: Vec<usize> = (0..5).collect();
            let col_indices = vec![0; 5];
            let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];

            MatrixMarketData::new(
                5,
                1,
                5,
                row_indices,
                col_indices,
                values,
                MatrixMarketSymmetry::General,
                false, // array format
            )
        };

        let vector = data.to_vector().expect("Failed to convert to vector");

        assert_eq!(vector.len(), data.rows);
        if std::path::Path::new(RHS_FILE).exists() {
            assert!((vector[0] - 0.0).abs() < 1e-12);
        } else {
            assert!((vector[0] - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn test_write_read_cycle_coordinate() {
        // First check if the matrix file exists, if not create test data
        let original_data = if std::path::Path::new(MATRIX_FILE).exists() {
            read_matrix_market(MATRIX_FILE).expect("Failed to read original matrix")
        } else {
            // Create test coordinate data if matrix file doesn't exist
            let row_indices = vec![0, 1, 2, 0];
            let col_indices = vec![0, 1, 2, 2];
            let values = vec![1.0, 2.0, 3.0, 4.0];

            MatrixMarketData::new(
                3,
                3,
                4,
                row_indices,
                col_indices,
                values,
                MatrixMarketSymmetry::General,
                true, // coordinate format
            )
        };

        // Ensure we're testing coordinate format
        assert!(
            original_data.is_coordinate,
            "Test data should be in coordinate format"
        );

        let (_tmp_dir, path) = temp_path(OUTPUT_FILE_COORD);
        write_matrix_market(&path, &original_data).expect("Failed to write matrix");

        // Verify the file was actually created
        assert!(path.exists(), "Output file was not created");

        let read_data = read_matrix_market(&path).expect("Failed to re-read matrix");

        // Validate dimensions
        assert_eq!(original_data.rows, read_data.rows);
        assert_eq!(original_data.cols, read_data.cols);
        assert_eq!(original_data.row_indices.len(), read_data.row_indices.len());

        // Validate format
        assert_eq!(original_data.is_coordinate, read_data.is_coordinate);
        assert_eq!(original_data.symmetry, read_data.symmetry);

        // temp_dir cleanup handled automatically
    }

    #[test]
    fn test_write_read_cycle_array() {
        // First check if the RHS file exists, if not create test data
        let original_data = if std::path::Path::new(RHS_FILE).exists() {
            read_matrix_market(RHS_FILE).expect("Failed to read original RHS")
        } else {
            // Create test array data if RHS file doesn't exist
            let row_indices: Vec<usize> = (0..5).collect();
            let col_indices = vec![0; 5];
            let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];

            MatrixMarketData::new(
                5,
                1,
                5,
                row_indices,
                col_indices,
                values,
                MatrixMarketSymmetry::General,
                false, // array format
            )
        };

        // Ensure we're testing array format
        assert!(
            !original_data.is_coordinate,
            "Test data should be in array format"
        );

        let (_tmp_dir, path) = temp_path(OUTPUT_FILE_ARRAY);
        write_matrix_market(&path, &original_data).expect("Failed to write RHS");

        // Verify the file was actually created
        assert!(path.exists(), "Output file was not created");

        let read_data = read_matrix_market(&path).expect("Failed to re-read RHS");

        // Validate dimensions
        assert_eq!(original_data.rows, read_data.rows);
        assert_eq!(original_data.cols, read_data.cols);
        assert_eq!(original_data.values.len(), read_data.values.len());

        // Validate format
        assert_eq!(original_data.is_coordinate, read_data.is_coordinate);
        assert_eq!(original_data.symmetry, read_data.symmetry);

        // temp_dir cleanup handled automatically
    }

    #[test]
    fn test_write_vector_market() {
        let vector = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let (_tmp_dir, path) = temp_path(OUTPUT_FILE_VECTOR);
        write_vector_market(&path, &vector).expect("Failed to write vector");

        let data = read_matrix_market(&path).expect("Failed to read vector file");

        assert_eq!(data.rows, 5);
        assert_eq!(data.cols, 1);
        assert!(!data.is_coordinate);
        assert_eq!(data.symmetry, MatrixMarketSymmetry::General);

        let read_vector = data.to_vector().expect("Failed to convert back to vector");

        assert_eq!(vector, read_vector);

        // temp_dir cleanup handled automatically
    }

    #[test]
    fn test_write_vector_market_scalar_real() {
        let vector: Vec<S> = (1..=4).map(|i| S::from_real(i as f64)).collect();

        let (_tmp_dir, path) = temp_path(OUTPUT_FILE_VECTOR_SCALAR_REAL);
        write_vector_market_scalar(&path, &vector).expect("Failed to write scalar vector");

        let data = read_matrix_market(&path).expect("Failed to read vector");

        assert_eq!(data.rows, vector.len());
        assert_eq!(data.cols, 1);
        assert_eq!(data.field_type, MatrixMarketField::Real);
        assert!(data.imag_values.is_none());

        let read_back = data
            .to_vector_scalar()
            .expect("Failed to reconstruct scalar vector");
        assert_eq!(read_back, vector);

        // temp_dir cleanup handled automatically
    }

    #[cfg(feature = "complex")]
    #[test]
    fn test_write_vector_market_scalar_complex() {
        let vector = vec![
            S::from_parts(1.0, 0.5),
            S::from_parts(0.0, 0.0),
            S::from_parts(-2.0, -1.25),
        ];

        let (_tmp_dir, path) = temp_path(OUTPUT_FILE_VECTOR_SCALAR_COMPLEX);
        write_vector_market_scalar(&path, &vector).expect("Failed to write complex scalar vector");

        let data = read_matrix_market(&path).expect("Failed to read complex vector");

        assert_eq!(data.field_type, MatrixMarketField::Complex);
        let imag = data
            .imag_values
            .as_ref()
            .expect("Complex vector should retain imaginary components");
        assert_eq!(imag, &vec![0.5, 0.0, -1.25]);

        let round_trip = data
            .to_vector_scalar()
            .expect("Failed to reconstruct complex vector");
        assert_eq!(round_trip, vector);

        // temp_dir cleanup handled automatically
    }

    #[test]
    fn test_write_matrix_market_coordinate_scalar_real() {
        let rows = 3;
        let cols = 3;
        let row_indices = vec![0, 1, 2];
        let col_indices = vec![0, 1, 2];
        let values: Vec<S> = vec![S::from_real(1.0), S::from_real(-2.5), S::from_real(4.25)];

        let (_tmp_dir, path) = temp_path(OUTPUT_FILE_COORD_SCALAR_REAL);
        write_matrix_market_coordinate_scalar(
            &path,
            rows,
            cols,
            &row_indices,
            &col_indices,
            &values,
            MatrixMarketSymmetry::General,
        )
        .expect("Failed to write scalar coordinate matrix");

        let data = read_matrix_market(&path).expect("Failed to read scalar coordinate matrix");

        assert_eq!(data.rows, rows);
        assert_eq!(data.cols, cols);
        assert_eq!(data.field_type, MatrixMarketField::Real);
        assert!(data.imag_values.is_none());

        let csr = data
            .to_csr_matrix_scalar()
            .expect("Failed to rebuild CSR from scalar matrix");
        assert_eq!(csr.values(), values.as_slice());

        // temp_dir cleanup handled automatically
    }

    #[cfg(feature = "complex")]
    #[test]
    fn test_write_matrix_market_coordinate_scalar_complex() {
        let rows = 2;
        let cols = 2;
        let row_indices = vec![0, 0, 1];
        let col_indices = vec![0, 1, 1];
        let values = vec![
            S::from_parts(2.0, 0.0),
            S::from_parts(1.5, -3.0),
            S::from_parts(-1.0, 2.0),
        ];

        let (_tmp_dir, path) = temp_path(OUTPUT_FILE_COORD_SCALAR_COMPLEX);
        write_matrix_market_coordinate_scalar(
            &path,
            rows,
            cols,
            &row_indices,
            &col_indices,
            &values,
            MatrixMarketSymmetry::General,
        )
        .expect("Failed to write complex scalar coordinate matrix");

        let data = read_matrix_market(&path).expect("Failed to read complex scalar matrix");

        assert_eq!(data.field_type, MatrixMarketField::Complex);
        let imag = data
            .imag_values
            .as_ref()
            .expect("Imaginary components should be preserved");
        assert_eq!(imag.len(), values.len());

        let csr = data
            .to_csr_matrix_scalar()
            .expect("Failed to rebuild complex CSR");
        assert_eq!(csr.values(), values.as_slice());

        // temp_dir cleanup handled automatically
    }

    #[cfg(feature = "complex")]
    #[test]
    fn test_write_matrix_market_coordinate_scalar_hermitian_checks() {
        let rows = 2;
        let cols = 2;
        let row_indices = vec![0, 1];
        let col_indices = vec![0, 1];
        let invalid_diag = vec![S::from_parts(1.0, 0.5), S::from_parts(2.0, 0.0)];

        let (_tmp_dir, path) = temp_path(OUTPUT_FILE_COORD_SCALAR_COMPLEX);
        let err = write_matrix_market_coordinate_scalar(
            &path,
            rows,
            cols,
            &row_indices,
            &col_indices,
            &invalid_diag,
            MatrixMarketSymmetry::Hermitian,
        )
        .expect_err("Hermitian matrices should reject complex diagonals");
        assert!(
            err.to_string()
                .contains("Diagonal entries of Hermitian matrices must be real")
        );

        let values = vec![
            S::from_parts(3.0, 0.0),
            S::from_parts(1.0, -2.0),
            S::from_parts(0.0, 0.0),
        ];
        let row_indices = vec![0, 0, 1];
        let col_indices = vec![0, 1, 1];

        write_matrix_market_coordinate_scalar(
            &path,
            rows,
            cols,
            &row_indices,
            &col_indices,
            &values,
            MatrixMarketSymmetry::Hermitian,
        )
        .expect("Hermitian matrix with real diagonal should succeed");

        let data = read_matrix_market(&path).expect("Failed to read hermitian scalar matrix");
        assert_eq!(data.symmetry, MatrixMarketSymmetry::Hermitian);
        assert_eq!(data.field_type, MatrixMarketField::Complex);

        let csr = data
            .to_csr_matrix_scalar()
            .expect("Failed to rebuild hermitian CSR");
        assert_eq!(csr.nrows(), rows);

        // temp_dir cleanup handled automatically
    }

    #[test]
    fn test_write_matrix_market_array_scalar_real() {
        let rows = 2;
        let cols = 3;
        let values: Vec<S> = vec![
            S::from_real(1.0),
            S::from_real(4.0),
            S::from_real(2.0),
            S::from_real(5.0),
            S::from_real(3.0),
            S::from_real(6.0),
        ];

        let (_tmp_dir, path) = temp_path(OUTPUT_FILE_ARRAY_SCALAR_REAL);
        write_matrix_market_array_scalar(&path, rows, cols, &values, MatrixMarketSymmetry::General)
            .expect("Failed to write scalar array matrix");

        let data = read_matrix_market(&path).expect("Failed to read scalar array matrix");

        assert_eq!(data.rows, rows);
        assert_eq!(data.cols, cols);
        assert!(!data.is_coordinate);
        assert_eq!(data.field_type, MatrixMarketField::Real);

        let round_trip: Vec<S> = data
            .values
            .iter()
            .enumerate()
            .map(|(idx, &re)| {
                if let Some(imag) = data.imag_values.as_ref() {
                    S::from_parts(re, imag[idx])
                } else {
                    S::from_real(re)
                }
            })
            .collect();

        assert_eq!(round_trip, values);

        // temp_dir cleanup handled automatically
    }

    #[cfg(feature = "complex")]
    #[test]
    fn test_write_matrix_market_array_scalar_complex() {
        let rows = 2;
        let cols = 2;
        let values = vec![
            S::from_parts(1.0, 0.0),
            S::from_parts(2.0, -1.0),
            S::from_parts(-3.0, 4.0),
            S::from_parts(0.5, 0.0),
        ];

        let (_tmp_dir, path) = temp_path(OUTPUT_FILE_ARRAY_SCALAR_COMPLEX);
        write_matrix_market_array_scalar(&path, rows, cols, &values, MatrixMarketSymmetry::General)
            .expect("Failed to write complex scalar array matrix");

        let data = read_matrix_market(&path).expect("Failed to read complex scalar array matrix");

        assert_eq!(data.field_type, MatrixMarketField::Complex);
        let imag = data
            .imag_values
            .as_ref()
            .expect("Imaginary parts should round trip");
        assert_eq!(imag.len(), values.len());

        let round_trip: Vec<S> = data
            .values
            .iter()
            .enumerate()
            .map(|(idx, &re)| S::from_parts(re, imag[idx]))
            .collect();

        assert_eq!(round_trip, values);

        // temp_dir cleanup handled automatically
    }

    #[cfg(feature = "complex")]
    #[test]
    fn test_write_matrix_market_array_scalar_hermitian_checks() {
        let rows = 2;
        let cols = 2;

        let invalid_diag = vec![
            S::from_parts(1.0, 0.5),
            S::from_parts(0.0, 0.0),
            S::from_parts(0.0, 0.0),
            S::from_parts(2.0, 0.0),
        ];

        let (_tmp_dir, path) = temp_path(OUTPUT_FILE_ARRAY_SCALAR_COMPLEX);
        let err = write_matrix_market_array_scalar(
            &path,
            rows,
            cols,
            &invalid_diag,
            MatrixMarketSymmetry::Hermitian,
        )
        .expect_err("Hermitian matrices must have real diagonal entries");
        assert!(
            err.to_string()
                .contains("Diagonal entries of Hermitian matrices must be real")
        );

        let invalid_offdiag = vec![
            S::from_parts(3.0, 0.0),
            S::from_parts(1.0, 2.0),
            S::from_parts(0.5, -2.0),
            S::from_parts(4.0, 0.0),
        ];

        let err = write_matrix_market_array_scalar(
            &path,
            rows,
            cols,
            &invalid_offdiag,
            MatrixMarketSymmetry::Hermitian,
        )
        .expect_err("Hermitian matrices require conjugate symmetry");
        assert!(
            err.to_string()
                .contains("must be conjugates for Hermitian matrices")
        );

        let valid = vec![
            S::from_parts(3.0, 0.0),
            S::from_parts(1.0, 2.0),
            S::from_parts(1.0, -2.0),
            S::from_parts(4.0, 0.0),
        ];

        write_matrix_market_array_scalar(
            &path,
            rows,
            cols,
            &valid,
            MatrixMarketSymmetry::Hermitian,
        )
        .expect("Valid Hermitian matrix should succeed");

        let data = read_matrix_market(&path).expect("Failed to read Hermitian scalar array matrix");
        assert_eq!(data.symmetry, MatrixMarketSymmetry::Hermitian);
        assert_eq!(data.field_type, MatrixMarketField::Complex);

        // temp_dir cleanup handled automatically
    }

    #[cfg(feature = "complex")]
    #[test]
    fn test_complex_matrix_to_scalar_csr() {
        let data = MatrixMarketData::with_field(
            2,
            2,
            2,
            vec![0, 1],
            vec![0, 1],
            vec![1.0, 2.0],
            Some(vec![0.5, -0.25]),
            MatrixMarketField::Complex,
            MatrixMarketSymmetry::General,
            true,
        );

        let csr = data
            .to_csr_matrix_scalar()
            .expect("Failed to convert to scalar CSR");
        assert_eq!(csr.nrows(), 2);
        assert_eq!(csr.nnz(), 2);
        let values = csr.values();
        assert_eq!(values[0], S::from_parts(1.0, 0.5));
        assert_eq!(values[1], S::from_parts(2.0, -0.25));

        let (rows, cols, nnz, row_idx, col_idx, real_vals, imag_vals) =
            data.clone().into_triplets();
        assert_eq!(rows, 2);
        assert_eq!(cols, 2);
        assert_eq!(nnz, 2);
        assert_eq!(row_idx, vec![0, 1]);
        assert_eq!(col_idx, vec![0, 1]);
        assert_eq!(real_vals, vec![1.0, 2.0]);
        assert_eq!(imag_vals, Some(vec![0.5, -0.25]));

        let err = data
            .to_csr_matrix()
            .err()
            .expect("Real CSR conversion should reject complex data");
        assert!(err.to_string().contains("Complex Matrix Market data"));
    }

    #[cfg(feature = "complex")]
    #[test]
    fn test_complex_vector_to_scalar() {
        let data = MatrixMarketData::with_field(
            3,
            1,
            3,
            vec![0, 1, 2],
            vec![0, 0, 0],
            vec![1.0, 0.0, -2.0],
            Some(vec![0.0, 1.0, 0.5]),
            MatrixMarketField::Complex,
            MatrixMarketSymmetry::General,
            false,
        );

        let vec_s = data
            .to_vector_scalar()
            .expect("Failed to convert complex vector");
        assert_eq!(vec_s.len(), 3);
        assert_eq!(vec_s[0], S::from_parts(1.0, 0.0));
        assert_eq!(vec_s[1], S::from_parts(0.0, 1.0));
        assert_eq!(vec_s[2], S::from_parts(-2.0, 0.5));

        let err = data
            .to_vector()
            .err()
            .expect("Real vector conversion should reject complex data");
        assert!(
            err.to_string()
                .contains("Complex Matrix Market data cannot be converted")
        );
    }

    #[cfg(feature = "complex")]
    #[test]
    fn test_complex_hermitian_expansion() {
        let data = MatrixMarketData::with_field(
            2,
            2,
            3,
            vec![0, 0, 1],
            vec![0, 1, 1],
            vec![1.0, 2.0, 4.0],
            Some(vec![0.0, 3.0, 0.0]),
            MatrixMarketField::Complex,
            MatrixMarketSymmetry::Hermitian,
            true,
        );

        let csr = data
            .to_csr_matrix_scalar()
            .expect("Hermitian expansion should succeed");
        assert_eq!(csr.row_ptr(), &[0, 2, 4]);
        assert_eq!(csr.col_idx(), &[0, 1, 0, 1]);
        let vals = csr.values();
        assert_eq!(vals[0], S::from_parts(1.0, 0.0));
        assert_eq!(vals[1], S::from_parts(2.0, 3.0));
        assert_eq!(vals[2], S::from_parts(2.0, -3.0));
        assert_eq!(vals[3], S::from_parts(4.0, 0.0));

        let bad = MatrixMarketData::with_field(
            2,
            2,
            2,
            vec![0, 1],
            vec![0, 1],
            vec![1.0, 2.0],
            Some(vec![1.0, 0.0]),
            MatrixMarketField::Complex,
            MatrixMarketSymmetry::Hermitian,
            true,
        );

        let err = bad
            .to_csr_matrix_scalar()
            .err()
            .expect("Hermitian matrices require real diagonals");
        assert!(
            err.to_string()
                .contains("Diagonal entries of Hermitian matrices must be real")
        );
    }

    #[test]
    fn test_skew_symmetric_expansion() {
        let data = MatrixMarketData::with_field(
            3,
            3,
            2,
            vec![0, 1],
            vec![1, 2],
            vec![2.0, -3.0],
            None,
            MatrixMarketField::Real,
            MatrixMarketSymmetry::SkewSymmetric,
            true,
        );

        let csr = data
            .to_csr_matrix_scalar()
            .expect("Skew-symmetric expansion should succeed");
        assert_eq!(csr.row_ptr(), &[0, 1, 3, 4]);
        assert_eq!(csr.col_idx(), &[1, 0, 2, 1]);
        let vals = csr.values();
        assert_eq!(vals[0], S::from_real(2.0));
        assert_eq!(vals[1], S::from_real(-2.0));
        assert_eq!(vals[2], S::from_real(-3.0));
        assert_eq!(vals[3], S::from_real(3.0));

        let bad = MatrixMarketData::with_field(
            2,
            2,
            1,
            vec![0],
            vec![0],
            vec![1.0],
            None,
            MatrixMarketField::Real,
            MatrixMarketSymmetry::SkewSymmetric,
            true,
        );

        let err = bad
            .to_csr_matrix_scalar()
            .err()
            .expect("Skew-symmetric matrices require zero diagonal");
        assert!(
            err.to_string()
                .contains("Diagonal entries must be zero for skew-symmetric")
        );
    }

    #[test]
    fn test_matrix_market_data_methods() {
        let data = MatrixMarketData::new(
            3,
            3,
            4,
            vec![0, 1, 2, 0],
            vec![0, 1, 2, 2],
            vec![1.0, 2.0, 3.0, 4.0],
            MatrixMarketSymmetry::General,
            true,
        );

        let (rows, cols, nnz, row_idx, col_idx, vals, imag) = data.clone().into_triplets();
        assert_eq!(rows, 3);
        assert_eq!(cols, 3);
        assert_eq!(nnz, 4);
        assert_eq!(row_idx, vec![0, 1, 2, 0]);
        assert_eq!(col_idx, vec![0, 1, 2, 2]);
        assert_eq!(vals, vec![1.0, 2.0, 3.0, 4.0]);
        assert!(imag.is_none());
        assert_eq!(data.symmetry, MatrixMarketSymmetry::General);
    }

    #[test]
    fn test_error_handling() {
        // Test invalid file
        let result = read_matrix_market("nonexistent.mtx");
        assert!(result.is_err());

        // Test invalid vector conversion
        let data = MatrixMarketData::new(
            2,
            2,
            2,
            vec![0, 1],
            vec![0, 1],
            vec![1.0, 2.0],
            MatrixMarketSymmetry::General,
            true,
        );
        let result = data.to_vector();
        assert!(result.is_err());
    }

    #[test]
    fn test_to_dense_matrix_scalar_array_real() {
        let data = MatrixMarketData::with_field(
            2,
            2,
            4,
            vec![0, 1, 0, 1],
            vec![0, 0, 1, 1],
            vec![1.0, 2.0, 3.0, 4.0],
            None,
            MatrixMarketField::Real,
            MatrixMarketSymmetry::General,
            false,
        );

        let dense = data
            .to_dense_matrix_scalar()
            .expect("Dense scalar conversion failed");
        crate::assert_s_close!("a00", dense[(0, 0)], S::from_real(1.0));
        crate::assert_s_close!("a10", dense[(1, 0)], S::from_real(2.0));
        crate::assert_s_close!("a01", dense[(0, 1)], S::from_real(3.0));
        crate::assert_s_close!("a11", dense[(1, 1)], S::from_real(4.0));
    }

    #[cfg(feature = "complex")]
    #[test]
    fn test_to_dense_matrix_scalar_array_complex() {
        let data = MatrixMarketData::with_field(
            2,
            2,
            4,
            vec![0, 1, 0, 1],
            vec![0, 0, 1, 1],
            vec![1.0, 2.0, 3.0, 4.0],
            Some(vec![0.5, -0.5, 1.5, -1.5]),
            MatrixMarketField::Complex,
            MatrixMarketSymmetry::General,
            false,
        );

        let dense = data
            .to_dense_matrix_scalar()
            .expect("Dense complex scalar conversion failed");
        crate::assert_s_close!("a00", dense[(0, 0)], S::from_parts(1.0, 0.5));
        crate::assert_s_close!("a10", dense[(1, 0)], S::from_parts(2.0, -0.5));
        crate::assert_s_close!("a01", dense[(0, 1)], S::from_parts(3.0, 1.5));
        crate::assert_s_close!("a11", dense[(1, 1)], S::from_parts(4.0, -1.5));
    }

    #[test]
    fn test_to_dense_matrix_scalar_coordinate() {
        let data = MatrixMarketData::with_field(
            2,
            2,
            3,
            vec![0, 0, 1],
            vec![0, 1, 1],
            vec![5.0, -1.0, 2.0],
            None,
            MatrixMarketField::Real,
            MatrixMarketSymmetry::General,
            true,
        );

        let dense = data
            .to_dense_matrix_scalar()
            .expect("Dense coordinate scalar conversion failed");
        crate::assert_s_close!("a00", dense[(0, 0)], S::from_real(5.0));
        crate::assert_s_close!("a01", dense[(0, 1)], S::from_real(-1.0));
        crate::assert_s_close!("a10", dense[(1, 0)], S::from_real(0.0));
        crate::assert_s_close!("a11", dense[(1, 1)], S::from_real(2.0));
    }
}
