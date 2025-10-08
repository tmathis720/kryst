#[cfg(test)]
mod tests_gmres_workspace_reuse {
    use crate::algebra::prelude::*;
    use crate::context::ksp_context::Workspace;
    use crate::error::KError;
    use crate::parallel::{NoComm, UniverseComm};
    use crate::preconditioner::PcSide;
    use crate::solver::gmres::GmresSolver;
    use faer::Mat;

    #[test]
    fn workspace_buffers_reused_between_solves() -> Result<(), KError> {
        let a = Mat::<R>::from_fn(
            2,
            2,
            |i, j| {
                if i == j { R::from(4.0) } else { R::from(1.0) }
            },
        );
        let b = [R::from(1.0), R::from(2.0)];
        let mut x = [R::default(); 2];

        let mut solver = GmresSolver::new(2, 1e-12, 10);
        let mut ws = Workspace::default();
        let comm = UniverseComm::NoComm(NoComm);

        // First solve to allocate workspace
        solver.solve_f64(
            &a,
            None,
            &b,
            &mut x,
            PcSide::Left,
            &comm,
            None,
            Some(&mut ws),
        )?;
        let v_ptr = ws.v_mem.as_ptr();
        let h_ptr = ws.h_mem.as_ptr();
        let v_cap = ws.v_mem.capacity();
        let h_cap = ws.h_mem.capacity();

        // Solve again and ensure buffers are reused
        x = [R::default(), R::default()];
        solver.solve_f64(
            &a,
            None,
            &b,
            &mut x,
            PcSide::Left,
            &comm,
            None,
            Some(&mut ws),
        )?;
        assert_eq!(v_ptr, ws.v_mem.as_ptr());
        assert_eq!(h_ptr, ws.h_mem.as_ptr());
        assert!(ws.v_mem.capacity() >= v_cap);
        assert!(ws.h_mem.capacity() >= h_cap);
        let n = b.len();
        let m = solver.restart;
        assert_eq!(ws.v_mem.len(), (m + 1) * n);
        assert_eq!(ws.h_mem.len(), (m + 1) * m);
        Ok(())
    }
}
