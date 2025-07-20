//! Preconditioner chaining for composite preconditioning strategies.
//!
//! This module implements PC-chaining that allows multiple preconditioners to be applied
//! in sequence. This is useful for composite strategies like "Chebyshev → ILUTP" or 
//! "Scaling → ILUTP → AMG".
//!
//! # Example Usage
//!
//! ```rust,ignore
//! use kryst::preconditioner::{PcChain, Jacobi, Ilu0};
//! 
//! let mut chain = PcChain::new();
//! chain.add_preconditioner(Box::new(Jacobi::new()));
//! chain.add_preconditioner(Box::new(Ilu0::new()));
//! 
//! // Apply the chain: Jacobi followed by ILU(0)
//! chain.apply(PcSide::Left, &r, &mut z)?;
//! ```

use crate::error::KError;
use crate::preconditioner::{Preconditioner, PcSide};
use faer::Mat;

/// Preconditioner chain that applies multiple preconditioners in sequence.
///
/// Each preconditioner in the chain is applied to the result of the previous one.
/// For a chain [PC1, PC2, PC3], the application is: z = PC3(PC2(PC1(r)))
pub struct PcChain {
    /// Sequence of preconditioners to apply
    chain: Vec<Box<dyn Preconditioner<Mat<f64>, Vec<f64>>>>,
    /// Temporary vector for intermediate results
    tmp: Option<Vec<f64>>,
}

impl PcChain {
    /// Create a new empty preconditioner chain.
    pub fn new() -> Self {
        Self {
            chain: Vec::new(),
            tmp: None,
        }
    }

    /// Add a preconditioner to the end of the chain.
    pub fn add_preconditioner(&mut self, pc: Box<dyn Preconditioner<Mat<f64>, Vec<f64>>>) {
        self.chain.push(pc);
    }

    /// Get the number of preconditioners in the chain.
    pub fn len(&self) -> usize {
        self.chain.len()
    }

    /// Check if the chain is empty.
    pub fn is_empty(&self) -> bool {
        self.chain.is_empty()
    }

    /// Parse a comma-separated string of preconditioner names into preconditioner types.
    /// 
    /// # Arguments
    /// * `chain_str` - Comma-separated list like "jacobi,ilu0,chebyshev"
    /// 
    /// Returns a vector of preconditioner type names.
    pub fn parse_chain_string(chain_str: &str) -> Vec<String> {
        chain_str
            .split(',')
            .map(|s| s.trim().to_lowercase())
            .filter(|s| !s.is_empty())
            .collect()
    }
}

impl Default for PcChain {
    fn default() -> Self {
        Self::new()
    }
}

impl Preconditioner<Mat<f64>, Vec<f64>> for PcChain {
    /// Setup all preconditioners in the chain.
    fn setup(&mut self, a: &Mat<f64>) -> Result<(), KError> {
        // Initialize temporary vector
        self.tmp = Some(vec![0.0; a.nrows()]);
        
        // Setup each preconditioner in the chain
        for pc in &mut self.chain {
            pc.setup(a)?;
        }
        
        Ok(())
    }

    /// Apply the preconditioner chain.
    /// 
    /// For a chain [PC1, PC2, PC3], applies: z = PC3(PC2(PC1(r)))
    fn apply(&self, side: PcSide, r: &Vec<f64>, z: &mut Vec<f64>) -> Result<(), KError> {
        if self.chain.is_empty() {
            // Empty chain: just copy input to output
            z.copy_from_slice(r);
            return Ok(());
        }

        let tmp = self.tmp.as_ref().ok_or_else(|| {
            KError::SolveError("PcChain not properly initialized - call setup() first".to_string())
        })?;

        if self.chain.len() == 1 {
            // Single preconditioner: apply directly
            self.chain[0].apply(side, r, z)?;
            return Ok(());
        }

        // Multiple preconditioners: use temporary vectors
        let mut current_in = r;
        let mut current_out = vec![0.0; r.len()];
        let mut next_tmp = vec![0.0; r.len()];

        for (i, pc) in self.chain.iter().enumerate() {
            if i == self.chain.len() - 1 {
                // Last preconditioner: output to z
                pc.apply(side, current_in, z)?;
            } else {
                // Intermediate preconditioner: output to temporary vector
                pc.apply(side, current_in, &mut current_out)?;
                
                // Swap vectors for next iteration
                std::mem::swap(&mut current_out, &mut next_tmp);
                current_in = &next_tmp;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preconditioner::jacobi::Jacobi;

    #[test]
    fn test_pc_chain_creation() {
        let chain = PcChain::new();
        assert_eq!(chain.len(), 0);
        assert!(chain.is_empty());
    }

    #[test]
    fn test_pc_chain_parse_string() {
        let parsed = PcChain::parse_chain_string("jacobi,ilu0,chebyshev");
        assert_eq!(parsed, vec!["jacobi", "ilu0", "chebyshev"]);

        let parsed = PcChain::parse_chain_string("jacobi, ilu0 , chebyshev ");
        assert_eq!(parsed, vec!["jacobi", "ilu0", "chebyshev"]);

        let parsed = PcChain::parse_chain_string("");
        assert_eq!(parsed, Vec::<String>::new());
    }

    #[test]
    fn test_pc_chain_single_preconditioner() {
        use faer::Mat;
        
        let matrix = Mat::<f64>::from_fn(3, 3, |i, j| {
            if i == j { 2.0 } else { 0.0 }
        });

        let mut chain = PcChain::new();
        chain.add_preconditioner(Box::new(Jacobi::new()));
        
        chain.setup(&matrix).unwrap();
        
        let r = vec![1.0, 2.0, 3.0];
        let mut z = vec![0.0; 3];
        
        chain.apply(PcSide::Left, &r, &mut z).unwrap();
        
        // Jacobi on diagonal matrix should give [0.5, 1.0, 1.5]
        assert!((z[0] - 0.5).abs() < 1e-10);
        assert!((z[1] - 1.0).abs() < 1e-10);
        assert!((z[2] - 1.5).abs() < 1e-10);
    }
}
