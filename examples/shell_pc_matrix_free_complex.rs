//! Complex-scalar shell PC callbacks with transpose/conjugate/symmetric hooks.

#[cfg(not(feature = "complex"))]
fn main() {
    eprintln!("shell_pc_matrix_free_complex.rs requires --features complex");
}

#[cfg(feature = "complex")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use kryst::algebra::scalar::{KrystScalar, S};
    use kryst::matrix::op::LinOp;
    use kryst::parallel::{NoComm, UniverseComm};
    use kryst::preconditioner::shell::{
        ShellPc, register_shell_apply_conjugate_transpose_typed,
        register_shell_apply_symmetric_left_typed, register_shell_apply_symmetric_right_typed,
        register_shell_apply_transpose_typed, register_shell_apply_typed,
        register_shell_context_typed,
    };
    use kryst::preconditioner::{Op, PcSide, Preconditioner};

    #[derive(Clone)]
    struct DiagMat {
        d: Vec<S>,
    }
    impl LinOp for DiagMat {
        type S = S;
        fn dims(&self) -> (usize, usize) {
            (self.d.len(), self.d.len())
        }
        fn matvec(&self, x: &[S], y: &mut [S]) {
            for (yi, (di, xi)) in y.iter_mut().zip(self.d.iter().zip(x.iter())) {
                *yi = *di * *xi;
            }
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
        fn comm(&self) -> UniverseComm {
            UniverseComm::NoComm(NoComm)
        }
    }

    #[derive(Default)]
    struct Ctx {
        calls: Vec<&'static str>,
    }

    let tag = "shell_pc_complex_demo";
    register_shell_context_typed(format!("{tag}_ctx"), Ctx::default);

    register_shell_apply_typed(format!("{tag}_apply"), |_side, x, y, ctx: &mut Ctx| {
        ctx.calls.push("apply");
        y.copy_from_slice(x);
        Ok(())
    });
    register_shell_apply_transpose_typed(
        format!("{tag}_transpose"),
        |_side, x, y, ctx: &mut Ctx| {
            ctx.calls.push("transpose");
            y.copy_from_slice(x);
            Ok(())
        },
    );
    register_shell_apply_conjugate_transpose_typed(
        format!("{tag}_conj"),
        |_side, x, y, ctx: &mut Ctx| {
            ctx.calls.push("conj_transpose");
            for (yi, xi) in y.iter_mut().zip(x.iter()) {
                *yi = xi.conj();
            }
            Ok(())
        },
    );
    register_shell_apply_symmetric_left_typed(
        format!("{tag}_sym_l"),
        |_side, x, y, ctx: &mut Ctx| {
            ctx.calls.push("sym_left");
            y.copy_from_slice(x);
            Ok(())
        },
    );
    register_shell_apply_symmetric_right_typed(
        format!("{tag}_sym_r"),
        |_side, x, y, ctx: &mut Ctx| {
            ctx.calls.push("sym_right");
            y.copy_from_slice(x);
            Ok(())
        },
    );

    let mut pc = ShellPc::new(
        Some(format!("{tag}_apply")),
        Some(format!("{tag}_transpose")),
        Some(format!("{tag}_conj")),
        None,
        Some(format!("{tag}_sym_l")),
        Some(format!("{tag}_sym_r")),
        None,
        None,
        Some(format!("{tag}_ctx")),
    );
    pc.setup(&DiagMat {
        d: vec![
            S::from_parts(2.0, 1.0),
            S::from_parts(3.0, -1.0),
            S::from_parts(4.0, 0.5),
        ],
    })?;

    let x = vec![S::from_parts(1.0, 2.0), S::from_parts(2.0, 1.0), S::one()];
    let mut y = vec![S::zero(); 3];

    pc.apply(PcSide::Left, &x, &mut y)?;
    pc.apply_op(Op::Trans, &x, &mut y)?;
    pc.apply_op(Op::ConjTrans, &x, &mut y)?;
    pc.apply(PcSide::Symmetric, &x, &mut y)?;

    pc.with_typed_context_ref::<Ctx, _, _>(|ctx| {
        println!("hook call order: {:?}", ctx.calls);
        Ok(())
    })?;
    Ok(())
}
