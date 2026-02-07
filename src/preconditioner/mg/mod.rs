use crate::algebra::scalar::S;
use crate::error::KError;
use crate::matrix::op::LinOp;
use crate::preconditioner::{PcSide, Preconditioner};

pub struct MgLevel {
    pub level: usize,
    pub smoother: Option<Box<dyn Preconditioner>>,
}

impl MgLevel {
    pub fn new(level: usize) -> Self {
        Self {
            level,
            smoother: None,
        }
    }
}

pub struct MgHierarchy {
    levels: Vec<MgLevel>,
}

impl MgHierarchy {
    pub fn new(num_levels: usize) -> Self {
        let levels = (0..num_levels).map(MgLevel::new).collect();
        Self { levels }
    }

    pub fn set_smoother(&mut self, level: usize, smoother: Box<dyn Preconditioner>) {
        if let Some(entry) = self.levels.get_mut(level) {
            entry.smoother = Some(smoother);
        }
    }

    pub fn levels(&self) -> &[MgLevel] {
        &self.levels
    }
}

pub struct MgPc {
    pub levels: usize,
    pub cycle_type: Option<String>,
    pub smoother: Option<String>,
    pub smoother_steps: Option<usize>,
    hierarchy: MgHierarchy,
}

impl MgPc {
    pub fn new(
        levels: usize,
        cycle_type: Option<String>,
        smoother: Option<String>,
        smoother_steps: Option<usize>,
    ) -> Self {
        let hierarchy = MgHierarchy::new(levels.max(1));
        Self {
            levels,
            cycle_type,
            smoother,
            smoother_steps,
            hierarchy,
        }
    }

    pub fn hierarchy(&self) -> &MgHierarchy {
        &self.hierarchy
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
