//! Command-line or API options for preconditioners.
//!
//! This module provides the `PcOptions` struct, which is used to specify
//! options for various types of preconditioners via command-line arguments
//! or API calls. The available preconditioners include Jacobi, SSOR, and
//! ILU(0). Additionally, parameters such as the relaxation factor for SSOR
//! and the drop tolerance for ILU(p) can be set.

/// Preconditioner types & parameters.
#[derive(Debug)]
pub struct PcOptions {
    /// Type of preconditioner (none, jacobi, ssor, ilu0)
    pub pc_type: String,

    /// Relaxation factor ω for SSOR
    pub omega: f64,

    /// Drop tolerance for ILU(p)
    pub drop_tol: f64,
}
