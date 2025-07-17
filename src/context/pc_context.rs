//! Preconditioner context and configuration for Krylov solvers.
//!
//! This module defines the `PC` enum, which provides a unified interface for specifying
//! and configuring all supported preconditioner types in the library. Preconditioners
//! are used to accelerate the convergence of iterative solvers by transforming the
//! linear system into a more favorable form. Each variant of the `PC` enum corresponds
//! to a specific preconditioning strategy, with associated parameters where applicable.
//!
//! # Supported Preconditioners
//!
//! - Jacobi: Diagonal scaling preconditioner.
//! - Ssor: Symmetric Successive Over-Relaxation.
//! - Ilu0: Incomplete LU factorization with zero fill-in.
//! - Ilup: Incomplete LU with fixed fill-in level.
//! - Ilut: Incomplete LU with threshold-based dropping.
//! - Chebyshev: Polynomial preconditioner using Chebyshev polynomials.
//! - ApproxInv: Approximate inverse preconditioner with configurable sparsity.
//! - BlockJacobi: Block-diagonal Jacobi preconditioner.
//! - Multicolor: Multicoloring-based preconditioner.
//! - AMG: Algebraic Multigrid preconditioner.
//! - AdditiveSchwarz: Additive Schwarz domain decomposition preconditioner.
//!
//! # Example
//!
//! ```rust,ignore
//! use crate::context::pc_context::{PC, SparsityPattern};
//! let pc = PC::Ilut { fill: 10, droptol: 1e-3 };
//! ```

/// Unified preconditioner enum for all supported types.
///
/// Each variant represents a different preconditioning strategy. Some variants
/// include parameters to control their behavior (e.g., fill level, drop tolerance,
/// polynomial degree, etc.).
#[derive(Debug, Clone)]
pub enum PC<T> {
    /// Jacobi (diagonal scaling) preconditioner.
    Jacobi,
    /// Symmetric Successive Over-Relaxation (SSOR) preconditioner.
    Ssor,
    /// Incomplete LU factorization with zero fill-in (ILU(0)).
    Ilu0,
    /// Incomplete LU factorization with fixed fill-in level (ILU(p)).
    ///
    /// - `fill`: The fill-in level (number of allowed extra nonzeros per row).
    Ilup { fill: usize },
    /// Incomplete LU factorization with threshold-based dropping (ILUT).
    ///
    /// - `fill`: Maximum number of nonzeros per row.
    /// - `droptol`: Drop tolerance for discarding small elements.
    Ilut { fill: usize, droptol: T },
    /// Chebyshev polynomial preconditioner.
    ///
    /// - `degree`: Degree of the Chebyshev polynomial.
    /// - `emin`: Optional lower bound on the spectrum.
    /// - `emax`: Optional upper bound on the spectrum.
    Chebyshev { degree: usize, emin: Option<T>, emax: Option<T> },
    /// Approximate inverse preconditioner.
    ///
    /// - `pattern`: Sparsity pattern for the approximate inverse.
    /// - `tol`: Convergence tolerance for the iterative construction.
    /// - `max_iter`: Maximum number of iterations for the construction algorithm.
    ApproxInv { pattern: SparsityPattern, tol: T, max_iter: usize },
    /// Block Jacobi preconditioner.
    ///
    /// - `blocks`: List of index blocks, each block is a list of row/column indices.
    BlockJacobi { blocks: Vec<Vec<usize>> },
    /// Multicolor preconditioner.
    ///
    /// - `colors`: Color assignment for each row/column (for parallelization).
    Multicolor { colors: Vec<usize> },
    /// Algebraic Multigrid (AMG) preconditioner.
    AMG,
    /// Additive Schwarz domain decomposition preconditioner.
    AdditiveSchwarz,
}

/// Sparsity pattern for approximate inverse preconditioners.
///
/// Used to control the nonzero structure of the approximate inverse. The `Auto` variant
/// lets the library choose a pattern automatically, while `Manual` allows the user to
/// specify the sparsity structure explicitly.
#[derive(Debug, Clone)]
pub enum SparsityPattern {
    /// Let the library choose the sparsity pattern automatically.
    Auto,
    /// User-specified sparsity pattern.
    ///
    /// Each inner vector contains the column indices for the corresponding row.
    Manual(Vec<Vec<usize>>), // for each row, the list of column indices
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pc_jacobi() {
        let pc: PC<f64> = PC::Jacobi;
        match pc {
            PC::Jacobi => assert!(true),
            _ => panic!("Expected Jacobi variant"),
        }
    }

    #[test]
    fn test_pc_ssor() {
        let pc: PC<f64> = PC::Ssor;
        match pc {
            PC::Ssor => assert!(true),
            _ => panic!("Expected Ssor variant"),
        }
    }

    #[test]
    fn test_pc_ilu0() {
        let pc: PC<f64> = PC::Ilu0;
        match pc {
            PC::Ilu0 => assert!(true),
            _ => panic!("Expected Ilu0 variant"),
        }
    }

    #[test]
    fn test_pc_ilup() {
        let pc: PC<f64> = PC::Ilup { fill: 5 };
        match pc {
            PC::Ilup { fill } => assert_eq!(fill, 5),
            _ => panic!("Expected Ilup variant"),
        }
    }

    #[test]
    fn test_pc_ilut() {
        let pc: PC<f64> = PC::Ilut { fill: 10, droptol: 1e-3 };
        match pc {
            PC::Ilut { fill, droptol } => {
                assert_eq!(fill, 10);
                assert_eq!(droptol, 1e-3);
            },
            _ => panic!("Expected Ilut variant"),
        }
    }

    #[test]
    fn test_pc_chebyshev() {
        let pc: PC<f64> = PC::Chebyshev { 
            degree: 3, 
            emin: Some(0.1), 
            emax: Some(10.0) 
        };
        match pc {
            PC::Chebyshev { degree, emin, emax } => {
                assert_eq!(degree, 3);
                assert_eq!(emin, Some(0.1));
                assert_eq!(emax, Some(10.0));
            },
            _ => panic!("Expected Chebyshev variant"),
        }
    }

    #[test]
    fn test_pc_chebyshev_no_bounds() {
        let pc: PC<f64> = PC::Chebyshev { 
            degree: 5, 
            emin: None, 
            emax: None 
        };
        match pc {
            PC::Chebyshev { degree, emin, emax } => {
                assert_eq!(degree, 5);
                assert_eq!(emin, None);
                assert_eq!(emax, None);
            },
            _ => panic!("Expected Chebyshev variant"),
        }
    }

    #[test]
    fn test_pc_approx_inv() {
        let pattern = SparsityPattern::Auto;
        let pc: PC<f64> = PC::ApproxInv { 
            pattern, 
            tol: 1e-6, 
            max_iter: 100 
        };
        match pc {
            PC::ApproxInv { tol, max_iter, .. } => {
                assert_eq!(tol, 1e-6);
                assert_eq!(max_iter, 100);
            },
            _ => panic!("Expected ApproxInv variant"),
        }
    }

    #[test]
    fn test_pc_block_jacobi() {
        let blocks = vec![vec![0, 1], vec![2, 3], vec![4]];
        let pc: PC<f64> = PC::BlockJacobi { blocks: blocks.clone() };
        match pc {
            PC::BlockJacobi { blocks: b } => assert_eq!(b, blocks),
            _ => panic!("Expected BlockJacobi variant"),
        }
    }

    #[test]
    fn test_pc_multicolor() {
        let colors = vec![0, 1, 0, 2, 1];
        let pc: PC<f64> = PC::Multicolor { colors: colors.clone() };
        match pc {
            PC::Multicolor { colors: c } => assert_eq!(c, colors),
            _ => panic!("Expected Multicolor variant"),
        }
    }

    #[test]
    fn test_pc_amg() {
        let pc: PC<f64> = PC::AMG;
        match pc {
            PC::AMG => assert!(true),
            _ => panic!("Expected AMG variant"),
        }
    }

    #[test]
    fn test_pc_additive_schwarz() {
        let pc: PC<f64> = PC::AdditiveSchwarz;
        match pc {
            PC::AdditiveSchwarz => assert!(true),
            _ => panic!("Expected AdditiveSchwarz variant"),
        }
    }

    #[test]
    fn test_sparsity_pattern_auto() {
        let pattern = SparsityPattern::Auto;
        match pattern {
            SparsityPattern::Auto => assert!(true),
            _ => panic!("Expected Auto variant"),
        }
    }

    #[test]
    fn test_sparsity_pattern_manual() {
        let structure = vec![vec![0, 1], vec![1, 2], vec![0, 2]];
        let pattern = SparsityPattern::Manual(structure.clone());
        match pattern {
            SparsityPattern::Manual(s) => assert_eq!(s, structure),
            _ => panic!("Expected Manual variant"),
        }
    }

    #[test]
    fn test_pc_clone() {
        let pc1: PC<f64> = PC::Ilut { fill: 5, droptol: 1e-4 };
        let pc2 = pc1.clone();
        
        match (pc1, pc2) {
            (PC::Ilut { fill: f1, droptol: d1 }, PC::Ilut { fill: f2, droptol: d2 }) => {
                assert_eq!(f1, f2);
                assert_eq!(d1, d2);
            },
            _ => panic!("Clone should preserve variant and values"),
        }
    }

    #[test]
    fn test_sparsity_pattern_clone() {
        let pattern1 = SparsityPattern::Manual(vec![vec![0, 1], vec![1]]);
        let pattern2 = pattern1.clone();
        
        match (pattern1, pattern2) {
            (SparsityPattern::Manual(s1), SparsityPattern::Manual(s2)) => {
                assert_eq!(s1, s2);
            },
            _ => panic!("Clone should preserve sparsity pattern"),
        }
    }

    #[test]
    fn test_pc_debug() {
        let pc: PC<f64> = PC::Jacobi;
        let debug_str = format!("{:?}", pc);
        assert!(debug_str.contains("Jacobi"));

        let pc2: PC<f64> = PC::Ilut { fill: 3, droptol: 1e-5 };
        let debug_str2 = format!("{:?}", pc2);
        assert!(debug_str2.contains("Ilut"));
        assert!(debug_str2.contains("3"));
    }

    #[test]
    fn test_sparsity_pattern_debug() {
        let pattern = SparsityPattern::Auto;
        let debug_str = format!("{:?}", pattern);
        assert!(debug_str.contains("Auto"));

        let pattern2 = SparsityPattern::Manual(vec![vec![0]]);
        let debug_str2 = format!("{:?}", pattern2);
        assert!(debug_str2.contains("Manual"));
    }

    #[test]
    fn test_pc_with_different_types() {
        // Test with f32
        let pc_f32: PC<f32> = PC::Ilut { fill: 2, droptol: 1e-3f32 };
        match pc_f32 {
            PC::Ilut { fill, droptol } => {
                assert_eq!(fill, 2);
                assert_eq!(droptol, 1e-3f32);
            },
            _ => panic!("Expected Ilut variant for f32"),
        }

        // Test with f64
        let pc_f64: PC<f64> = PC::Chebyshev { 
            degree: 4, 
            emin: Some(0.5), 
            emax: Some(5.0) 
        };
        match pc_f64 {
            PC::Chebyshev { degree, emin, emax } => {
                assert_eq!(degree, 4);
                assert_eq!(emin, Some(0.5));
                assert_eq!(emax, Some(5.0));
            },
            _ => panic!("Expected Chebyshev variant for f64"),
        }
    }

    #[test]
    fn test_complex_pc_configurations() {
        // Test ApproxInv with manual sparsity pattern
        let manual_pattern = SparsityPattern::Manual(vec![
            vec![0, 1, 2],
            vec![0, 1],
            vec![1, 2],
        ]);
        let pc: PC<f64> = PC::ApproxInv { 
            pattern: manual_pattern, 
            tol: 1e-8, 
            max_iter: 50 
        };
        
        match pc {
            PC::ApproxInv { pattern, tol, max_iter } => {
                assert_eq!(tol, 1e-8);
                assert_eq!(max_iter, 50);
                match pattern {
                    SparsityPattern::Manual(s) => {
                        assert_eq!(s.len(), 3);
                        assert_eq!(s[0], vec![0, 1, 2]);
                    },
                    _ => panic!("Expected manual pattern"),
                }
            },
            _ => panic!("Expected ApproxInv variant"),
        }
    }

    #[test]
    fn test_empty_configurations() {
        // Test empty blocks for BlockJacobi
        let pc: PC<f64> = PC::BlockJacobi { blocks: vec![] };
        match pc {
            PC::BlockJacobi { blocks } => assert!(blocks.is_empty()),
            _ => panic!("Expected BlockJacobi variant"),
        }

        // Test empty colors for Multicolor
        let pc2: PC<f64> = PC::Multicolor { colors: vec![] };
        match pc2 {
            PC::Multicolor { colors } => assert!(colors.is_empty()),
            _ => panic!("Expected Multicolor variant"),
        }

        // Test empty sparsity pattern
        let pattern = SparsityPattern::Manual(vec![]);
        match pattern {
            SparsityPattern::Manual(s) => assert!(s.is_empty()),
            _ => panic!("Expected Manual pattern"),
        }
    }
}
