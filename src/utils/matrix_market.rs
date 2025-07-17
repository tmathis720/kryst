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
//! let (rows, cols, nnz, row_indices, col_indices, values) = data.into_triplets();
//! ```

use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use crate::matrix::sparse::CsrMatrix;
use crate::error::KError;

/// Helper function to map parse errors to KError.
fn parse_error<E>(err: E) -> KError
where
    E: std::fmt::Debug,
{
    KError::SolveError(format!("Parse error: {:?}", err))
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
    /// Matrix values
    pub values: Vec<f64>,
    /// Whether the matrix is symmetric (affects storage and conversion)
    pub is_symmetric: bool,
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
        is_symmetric: bool,
        is_coordinate: bool,
    ) -> Self {
        Self {
            rows,
            cols,
            nonzeros,
            row_indices,
            col_indices,
            values,
            is_symmetric,
            is_coordinate,
        }
    }

    /// Convert to CSR format triplets (rows, cols, nnz, row_indices, col_indices, values).
    pub fn into_triplets(self) -> (usize, usize, usize, Vec<usize>, Vec<usize>, Vec<f64>) {
        (self.rows, self.cols, self.nonzeros, self.row_indices, self.col_indices, self.values)
    }

    /// Convert to a Kryst CSR matrix.
    pub fn to_csr_matrix(&self) -> Result<CsrMatrix<f64>, KError> {
        if self.row_indices.len() != self.col_indices.len() || 
           self.row_indices.len() != self.values.len() {
            return Err(KError::SolveError("Inconsistent triplet lengths".to_string()));
        }

        // Build CSR structure from triplets
        let mut row_ptr = vec![0; self.rows + 1];
        let mut triplets: Vec<_> = self.row_indices.iter()
            .zip(self.col_indices.iter())
            .zip(self.values.iter())
            .map(|((&r, &c), &v)| (r, c, v))
            .collect();

        // Sort by row, then column
        triplets.sort_by_key(|&(r, c, _)| (r, c));

        // Handle symmetric matrices by expanding to full storage
        if self.is_symmetric {
            let mut expanded_triplets = Vec::new();
            for &(r, c, v) in &triplets {
                expanded_triplets.push((r, c, v));
                if r != c {
                    expanded_triplets.push((c, r, v)); // Add symmetric entry
                }
            }
            triplets = expanded_triplets;
            triplets.sort_by_key(|&(r, c, _)| (r, c));
        }

        // Count entries per row
        for &(r, _, _) in &triplets {
            if r >= self.rows {
                return Err(KError::SolveError(format!("Row index {} out of bounds", r)));
            }
            row_ptr[r + 1] += 1;
        }

        // Convert counts to pointers
        for i in 1..=self.rows {
            row_ptr[i] += row_ptr[i - 1];
        }

        // Extract column indices and values in CSR order
        let col_idx: Vec<usize> = triplets.iter().map(|&(_, c, _)| c).collect();
        let values: Vec<f64> = triplets.iter().map(|&(_, _, v)| v).collect();

        Ok(CsrMatrix::from_csr(self.rows, self.cols, row_ptr, col_idx, values))
    }

    /// Convert to a dense vector (for right-hand sides).
    /// 
    /// This assumes the matrix is a column vector (cols == 1).
    pub fn to_vector(&self) -> Result<Vec<f64>, KError> {
        if self.cols != 1 {
            return Err(KError::SolveError(format!(
                "Cannot convert {}x{} matrix to vector (must be Nx1)", 
                self.rows, self.cols
            )));
        }

        let mut vector = vec![0.0; self.rows];
        for ((&r, &_c), &v) in self.row_indices.iter()
            .zip(self.col_indices.iter())
            .zip(self.values.iter()) {
            if r >= self.rows {
                return Err(KError::SolveError(format!("Row index {} out of bounds", r)));
            }
            vector[r] = v;
        }

        Ok(vector)
    }
}

/// Reads a Matrix Market file and returns the data as a MatrixMarketData struct.
pub fn read_matrix_market<P: AsRef<Path>>(file_path: P) -> Result<MatrixMarketData, KError> {
    let file = File::open(&file_path).map_err(|e| 
        KError::SolveError(format!("Failed to open file {:?}: {}", file_path.as_ref(), e)))?;
    let reader = BufReader::new(file);

    let mut lines = reader.lines();
    let header = lines
        .next()
        .ok_or_else(|| KError::SolveError("Empty file".to_string()))?
        .map_err(|e| KError::SolveError(format!("Failed to read header: {}", e)))?;

    // Check the Matrix Market banner
    if !header.starts_with("%%MatrixMarket") {
        return Err(KError::SolveError("Invalid Matrix Market banner".to_string()));
    }

    // Parse header format information
    let header_parts: Vec<&str> = header.split_whitespace().collect();
    if header_parts.len() < 5 {
        return Err(KError::SolveError(format!(
            "Invalid Matrix Market header format: expected at least 5 parts, got {}: {}", 
            header_parts.len(), header
        )));
    }

    // Format: %%MatrixMarket matrix <format> <field> <symmetry>
    // Example: %%MatrixMarket matrix coordinate real general
    let is_coordinate = header_parts[2] == "coordinate";
    let is_array = header_parts[2] == "array";
    let is_symmetric = header_parts.len() > 4 && header_parts[4] == "symmetric";

    if !is_coordinate && !is_array {
        return Err(KError::SolveError(format!(
            "Unsupported Matrix Market format: {}", header_parts[3]
        )));
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
        .map_err(|e| KError::SolveError(format!("Failed to read size line: {}", e)))?;

    // Parse size line
    let size_parts: Vec<&str> = size_line.split_whitespace().collect();
    let (rows, cols, declared_nonzeros) = if is_array {
        // Array format: rows cols
        if size_parts.len() != 2 {
            return Err(KError::SolveError("Invalid size format for array".to_string()));
        }
        let rows = size_parts[0].parse::<usize>().map_err(parse_error)?;
        let cols = size_parts[1].parse::<usize>().map_err(parse_error)?;
        (rows, cols, rows * cols)
    } else {
        // Coordinate format: rows cols nonzeros
        if size_parts.len() != 3 {
            return Err(KError::SolveError("Invalid size format for coordinate".to_string()));
        }
        let rows = size_parts[0].parse::<usize>().map_err(parse_error)?;
        let cols = size_parts[1].parse::<usize>().map_err(parse_error)?;
        let nonzeros = size_parts[2].parse::<usize>().map_err(parse_error)?;
        (rows, cols, nonzeros)
    };

    let mut row_indices = Vec::new();
    let mut col_indices = Vec::new();
    let mut values = Vec::new();

    if is_array {
        // Parse dense array data (column-major order in Matrix Market)
        let mut entry_count = 0;
        for line in lines {
            let line = line.map_err(|e| KError::SolveError(format!("Failed to read line: {}", e)))?;
            let trimmed_line = line.trim();
            if trimmed_line.is_empty() {
                continue;
            }
            
            let value = trimmed_line.parse::<f64>().map_err(parse_error)?;
            
            // Matrix Market array format is column-major
            let col = entry_count / rows;
            let row = entry_count % rows;
            
            if col >= cols || row >= rows {
                return Err(KError::SolveError(format!(
                    "Entry index ({}, {}) out of bounds for {}x{} matrix", 
                    row, col, rows, cols
                )));
            }
            
            row_indices.push(row);
            col_indices.push(col);
            values.push(value);
            entry_count += 1;
            
            // Stop if we've read all expected entries
            if entry_count >= declared_nonzeros {
                break;
            }
        }
    } else {
        // Parse sparse coordinate data
        for line in lines {
            let line = line.map_err(|e| KError::SolveError(format!("Failed to read line: {}", e)))?;
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
            let value = if parts.len() > 2 {
                parts[2].parse::<f64>().map_err(parse_error)?
            } else {
                1.0 // Default value for pattern matrices
            };
            
            if row >= rows || col >= cols {
                return Err(KError::SolveError(format!(
                    "Entry index ({}, {}) out of bounds", row, col
                )));
            }
            
            row_indices.push(row);
            col_indices.push(col);
            values.push(value);
        }
    }

    let _actual_nonzeros = row_indices.len();
    
    Ok(MatrixMarketData::new(
        rows,
        cols,
        declared_nonzeros,
        row_indices,
        col_indices,
        values,
        is_symmetric,
        is_coordinate,
    ))
}

/// Writes matrix data to a Matrix Market file.
pub fn write_matrix_market<P: AsRef<Path>>(
    file_path: P,
    data: &MatrixMarketData,
) -> Result<(), KError> {
    let mut file = File::create(&file_path).map_err(|e| 
        KError::SolveError(format!("Failed to create file {:?}: {}", file_path.as_ref(), e)))?;

    // Write header
    let format_type = if data.is_coordinate { "coordinate" } else { "array" };
    let symmetry = if data.is_symmetric { "symmetric" } else { "general" };
    writeln!(file, "%%MatrixMarket matrix {} real {}", format_type, symmetry)
        .map_err(|e| KError::SolveError(format!("Failed to write header: {}", e)))?;

    // Write size information
    if data.is_coordinate {
        writeln!(file, "{} {} {}", data.rows, data.cols, data.nonzeros)
    } else {
        writeln!(file, "{} {}", data.rows, data.cols)
    }.map_err(|e| KError::SolveError(format!("Failed to write size: {}", e)))?;

    // Write data
    if data.is_coordinate {
        // Write coordinate format
        for ((&row, &col), &value) in data.row_indices.iter()
            .zip(data.col_indices.iter())
            .zip(data.values.iter()) {
            writeln!(file, "{} {} {}", row + 1, col + 1, value)
                .map_err(|e| KError::SolveError(format!("Failed to write entry: {}", e)))?;
        }
    } else {
        // Write array format (column-major order)
        // First, reorganize data into column-major order if needed
        let mut dense_data = vec![0.0; data.rows * data.cols];
        for ((&row, &col), &value) in data.row_indices.iter()
            .zip(data.col_indices.iter())
            .zip(data.values.iter()) {
            let index = col * data.rows + row; // column-major indexing
            dense_data[index] = value;
        }
        
        for &value in &dense_data {
            writeln!(file, "{}", value)
                .map_err(|e| KError::SolveError(format!("Failed to write value: {}", e)))?;
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
    is_symmetric: bool,
) -> Result<(), KError> {
    let data = MatrixMarketData::new(
        rows,
        cols,
        row_indices.len(),
        row_indices.to_vec(),
        col_indices.to_vec(),
        values.to_vec(),
        is_symmetric,
        true, // coordinate format
    );
    write_matrix_market(file_path, &data)
}

/// Convenience function to write vector data in array format.
pub fn write_vector_market<P: AsRef<Path>>(
    file_path: P,
    vector: &[f64],
) -> Result<(), KError> {
    let row_indices: Vec<usize> = (0..vector.len()).collect();
    let col_indices = vec![0; vector.len()];
    
    let data = MatrixMarketData::new(
        vector.len(),
        1,
        vector.len(),
        row_indices,
        col_indices,
        vector.to_vec(),
        false, // not symmetric
        false, // array format
    );
    write_matrix_market(file_path, &data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::sparse::SparseMatrix;
    use std::fs;

    const MATRIX_FILE: &str = "examples/e05r0000/e05r0000.mtx";
    const RHS_FILE: &str = "examples/e05r0000/e05r0000_rhs1.mtx";
    const OUTPUT_FILE: &str = "test_output.mtx";

    #[test]
    fn test_read_sparse_matrix_market() {
        let data = read_matrix_market(MATRIX_FILE)
            .expect("Failed to read sparse matrix market file");

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
        let data = read_matrix_market(RHS_FILE)
            .expect("Failed to read dense matrix market file");

        // Assertions based on known metadata
        assert_eq!(data.rows, 236, "Unexpected number of rows");
        assert_eq!(data.cols, 1, "Unexpected number of columns");
        assert_eq!(data.nonzeros, 236, "Unexpected number of non-zero entries");
        assert!(!data.is_coordinate, "Should be array format");

        // Validate the first entry
        assert_eq!(data.row_indices[0], 0, "First row index mismatch");
        assert_eq!(data.col_indices[0], 0, "First column index mismatch");
        assert!(
            (data.values[0] - 0.0).abs() < 1e-12,
            "First value mismatch"
        );
    }

    #[test]
    fn test_to_csr_matrix() {
        let data = read_matrix_market(MATRIX_FILE)
            .expect("Failed to read matrix file");

        let csr_matrix = data.to_csr_matrix()
            .expect("Failed to convert to CSR matrix");

        assert_eq!(csr_matrix.nrows(), 236);
        assert_eq!(csr_matrix.ncols(), 236);
    }

    #[test]
    fn test_to_vector() {
        let data = read_matrix_market(RHS_FILE)
            .expect("Failed to read RHS file");

        let vector = data.to_vector()
            .expect("Failed to convert to vector");

        assert_eq!(vector.len(), 236);
        assert!((vector[0] - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_write_read_cycle_coordinate() {
        let original_data = read_matrix_market(MATRIX_FILE)
            .expect("Failed to read original matrix");

        write_matrix_market(OUTPUT_FILE, &original_data)
            .expect("Failed to write matrix");

        let read_data = read_matrix_market(OUTPUT_FILE)
            .expect("Failed to re-read matrix");

        // Validate dimensions
        assert_eq!(original_data.rows, read_data.rows);
        assert_eq!(original_data.cols, read_data.cols);
        assert_eq!(original_data.row_indices.len(), read_data.row_indices.len());

        // Clean up
        let _ = fs::remove_file(OUTPUT_FILE);
    }

    #[test]
    fn test_write_read_cycle_array() {
        let original_data = read_matrix_market(RHS_FILE)
            .expect("Failed to read original RHS");

        write_matrix_market(OUTPUT_FILE, &original_data)
            .expect("Failed to write RHS");

        let read_data = read_matrix_market(OUTPUT_FILE)
            .expect("Failed to re-read RHS");

        // Validate dimensions
        assert_eq!(original_data.rows, read_data.rows);
        assert_eq!(original_data.cols, read_data.cols);
        assert_eq!(original_data.values.len(), read_data.values.len());

        // Clean up
        let _ = fs::remove_file(OUTPUT_FILE);
    }

    #[test]
    fn test_write_vector_market() {
        let vector = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        
        write_vector_market(OUTPUT_FILE, &vector)
            .expect("Failed to write vector");

        let data = read_matrix_market(OUTPUT_FILE)
            .expect("Failed to read vector file");

        assert_eq!(data.rows, 5);
        assert_eq!(data.cols, 1);
        assert!(!data.is_coordinate);

        let read_vector = data.to_vector()
            .expect("Failed to convert back to vector");

        assert_eq!(vector, read_vector);

        // Clean up
        let _ = fs::remove_file(OUTPUT_FILE);
    }

    #[test]
    fn test_matrix_market_data_methods() {
        let data = MatrixMarketData::new(
            3, 3, 4,
            vec![0, 1, 2, 0],
            vec![0, 1, 2, 2],
            vec![1.0, 2.0, 3.0, 4.0],
            false,
            true,
        );

        let (rows, cols, nnz, row_idx, col_idx, vals) = data.clone().into_triplets();
        assert_eq!(rows, 3);
        assert_eq!(cols, 3);
        assert_eq!(nnz, 4);
        assert_eq!(row_idx, vec![0, 1, 2, 0]);
        assert_eq!(col_idx, vec![0, 1, 2, 2]);
        assert_eq!(vals, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_error_handling() {
        // Test invalid file
        let result = read_matrix_market("nonexistent.mtx");
        assert!(result.is_err());

        // Test invalid vector conversion
        let data = MatrixMarketData::new(
            2, 2, 2,
            vec![0, 1],
            vec![0, 1],
            vec![1.0, 2.0],
            false,
            true,
        );
        let result = data.to_vector();
        assert!(result.is_err());
    }
}
