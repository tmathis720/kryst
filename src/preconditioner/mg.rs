use crate::algebra::scalar::S;
use crate::error::KError;
use crate::matrix::op::LinOp;
use crate::preconditioner::{PcSide, Preconditioner};

pub struct MgPc {
    pub levels: usize,
    pub cycle_type: Option<String>,
}

impl MgPc {
    pub fn new(levels: usize, cycle_type: Option<String>) -> Self {
        Self { levels, cycle_type }
    }
}

impl Preconditioner for MgPc {
    fn setup(&mut self, _a: &dyn LinOp<S = S>) -> Result<(), KError> {
        if self.levels < 2 {
            return Err(KError::InvalidInput("pc_mg_levels must be >= 2".into()));
        }
        Ok(())
    }

    fn apply(&self, _side: PcSide, x: &[S], y: &mut [S]) -> Result<(), KError> {
        if x.len() != y.len() {
            return Err(KError::InvalidInput(
                "mg input/output length mismatch".into(),
            ));
        }
        y.copy_from_slice(x);
        Ok(())
    }
}
