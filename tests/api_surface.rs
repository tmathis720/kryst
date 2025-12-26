use kryst::prelude::*;
use std::sync::Arc;

struct IdentityOp(usize);

impl LinOp for IdentityOp {
    type S = R;

    fn dims(&self) -> (usize, usize) {
        (self.0, self.0)
    }

    fn matvec(&self, x: &[R], y: &mut [R]) {
        y.copy_from_slice(x);
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[test]
fn api_surface_lifecycle() -> Result<(), Box<dyn std::error::Error>> {
    let _ = PcType::None;
    let _ksp_opts = KspOptions::default();
    let _pc_opts = PcOptions::default();
    let _ = ConvergedReason::Continued;

    let op = Arc::new(IdentityOp(4)) as Arc<dyn LinOp<S = R>>;
    let b = vec![S::from_real(1.0); 4];
    let mut x = vec![S::zero(); 4];

    let mut ksp = KspContext::new();
    ksp.set_type(SolverType::Gmres)?;
    ksp.set_operators(op, None);
    ksp.setup()?;

    let result = ksp.solve(&b, &mut x);
    if cfg!(feature = "complex") {
        assert!(result.is_err());
        return Ok(());
    }

    let stats = result?;
    assert!(stats.iterations <= ksp.maxits);

    Ok(())
}
