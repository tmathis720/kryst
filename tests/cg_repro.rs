#![cfg(feature = "backend-faer")]

use kryst::config::options::{CgVariant, KspOptions};
use kryst::context::ksp_context::{KspContext, SolverType};
use kryst::matrix::op::{CsrOp, LinOp};
use kryst::matrix::sparse::CsrMatrix;
use kryst::parallel::{NoComm, UniverseComm};
use kryst::preconditioner::PcSide;
use std::sync::{Arc, Mutex};

#[cfg(feature = "mpi")]
use kryst::parallel::MpiComm;

fn make_spd_operator(comm: &UniverseComm, n: usize) -> Arc<dyn LinOp<S = f64>> {
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::with_capacity(3 * n);
    let mut values = Vec::with_capacity(3 * n);
    row_ptr.push(0);
    for i in 0..n {
        if i > 0 {
            col_idx.push(i - 1);
            values.push(-1.0);
        }
        col_idx.push(i);
        values.push(4.0);
        if i + 1 < n {
            col_idx.push(i + 1);
            values.push(-1.0);
        }
        row_ptr.push(col_idx.len());
    }
    let csr = Arc::new(CsrMatrix::from_csr(n, n, row_ptr, col_idx, values));
    let op = CsrOp::new(csr).with_comm(comm.clone());
    Arc::new(op)
}

fn run_once(comm: &UniverseComm, threads: Option<usize>) -> Vec<f64> {
    let n = 32;
    let a = make_spd_operator(comm, n);
    let rhs = vec![1.0; n];
    let mut x = vec![0.0; n];

    let mut ksp = KspContext::new();
    ksp.set_type(SolverType::Cg).unwrap();
    ksp.try_set_pc_side(PcSide::Left).unwrap();
    ksp.set_operators(a, None);

    let mut opts = KspOptions::default();
    opts.reproducible = Some(true);
    opts.cg_variant = Some(CgVariant::Classic);
    opts.threads = threads;
    ksp.set_from_options(&opts).unwrap();
    ksp.set_tolerances(1e-12, 0.0, 1e6, 100);

    let history = Arc::new(Mutex::new(Vec::new()));
    let monitor_history = Arc::clone(&history);
    ksp.clear_monitors();
    ksp.add_monitor(move |_iter, residual| {
        if let Ok(mut guard) = monitor_history.lock() {
            guard.push(residual);
        }
    });

    ksp.solve(&rhs, &mut x).unwrap();

    history.lock().unwrap().clone()
}

fn test_comm() -> UniverseComm {
    #[cfg(feature = "mpi")]
    {
        UniverseComm::Mpi(Arc::new(MpiComm::new()))
    }
    #[cfg(not(feature = "mpi"))]
    {
        UniverseComm::NoComm(NoComm)
    }
}

#[test]
fn reproducible_history_single_thread() {
    let comm = test_comm();
    let h1 = run_once(&comm, Some(1));
    let h2 = run_once(&comm, Some(1));
    assert_eq!(h1.len(), h2.len());
    for (a, b) in h1.iter().zip(h2.iter()) {
        assert_eq!(a.to_bits(), b.to_bits(), "residual mismatch: {a} vs {b}");
    }
}

#[cfg(feature = "rayon")]
#[test]
fn reproducible_history_multi_thread() {
    let comm = test_comm();
    let h1 = run_once(&comm, Some(4));
    let h2 = run_once(&comm, Some(4));
    assert_eq!(h1.len(), h2.len());
    for (a, b) in h1.iter().zip(h2.iter()) {
        let scale = a.abs().max(b.abs()).max(1.0);
        assert!(
            (a - b).abs() <= 1e-12 * scale,
            "residual mismatch: {a} vs {b}"
        );
    }
}
