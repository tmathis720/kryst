//! Matrix-free `PCSHELL`-style preconditioner callbacks with communicator-aware context.

#[cfg(feature = "complex")]
fn main() {
    eprintln!("shell_pc_matrix_free.rs is unavailable when built with --features complex");
}

#[cfg(not(feature = "complex"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use kryst::algebra::scalar::S;
    use kryst::matrix::op::LinOp;
    use kryst::parallel::{Comm, NoComm, UniverseComm};
    use kryst::preconditioner::shell::{
        ShellPc, register_shell_apply_conjugate_transpose_typed,
        register_shell_apply_symmetric_left_typed, register_shell_apply_symmetric_right_typed,
        register_shell_apply_transpose_typed, register_shell_apply_typed,
        register_shell_context_typed, register_shell_setup, shell_context_downcast_mut,
        shell_setup_with_context,
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
    struct ShellState {
        rank: usize,
        size: usize,
        left_calls: usize,
        right_calls: usize,
    }

    let tag = "shell_pc_matrix_free_demo";
    register_shell_context_typed(format!("{tag}_ctx"), ShellState::default);
    register_shell_setup(
        format!("{tag}_setup"),
        shell_setup_with_context(|a, ctx| {
            let state = shell_context_downcast_mut::<ShellState>(ctx)?;
            let comm = a.comm();
            state.rank = comm.rank();
            state.size = comm.size();
            Ok(())
        }),
    );
    register_shell_apply_typed(
        format!("{tag}_apply"),
        |_side, x, y, _ctx: &mut ShellState| {
            y.copy_from_slice(x);
            Ok(())
        },
    );
    register_shell_apply_transpose_typed(
        format!("{tag}_transpose"),
        |_side, x, y, _ctx: &mut ShellState| {
            y.copy_from_slice(x);
            Ok(())
        },
    );
    register_shell_apply_conjugate_transpose_typed(
        format!("{tag}_conj"),
        |_side, x, y, _ctx: &mut ShellState| {
            y.copy_from_slice(x);
            Ok(())
        },
    );
    register_shell_apply_symmetric_left_typed(
        format!("{tag}_sym_l"),
        |_side, x, y, ctx: &mut ShellState| {
            ctx.left_calls += 1;
            y.copy_from_slice(x);
            Ok(())
        },
    );
    register_shell_apply_symmetric_right_typed(
        format!("{tag}_sym_r"),
        |_side, x, y, ctx: &mut ShellState| {
            ctx.right_calls += 1;
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
        Some(format!("{tag}_setup")),
        None,
        Some(format!("{tag}_ctx")),
    );
    pc.setup(&DiagMat {
        d: vec![4.0, 5.0, 6.0],
    })?;

    let x = vec![1.0; 3];
    let mut y = vec![0.0; 3];
    pc.apply(PcSide::Left, &x, &mut y)?;
    pc.apply_op(Op::Trans, &x, &mut y)?;
    pc.apply_op(Op::ConjTrans, &x, &mut y)?;
    pc.apply(PcSide::Symmetric, &x, &mut y)?;

    pc.with_typed_context_ref::<ShellState, _, _>(|state| {
        println!("comm rank={} size={}", state.rank, state.size);
        println!(
            "symmetric callbacks: left={} right={}",
            state.left_calls, state.right_calls
        );
        Ok(())
    })?;
    Ok(())
}
