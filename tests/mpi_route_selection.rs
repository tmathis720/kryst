#![cfg(all(feature = "backend-faer", feature = "mpi", not(feature = "complex")))]

mod fixtures;

use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

use kryst::config::options::{KspOptions, PcOptions};
use kryst::context::ksp_context::{KspContext, SolverType};
use kryst::matrix::dist_csr::DistCsrOp;
use kryst::matrix::sparse::CsrMatrix;
use kryst::parallel::{Comm, MpiComm, UniverseComm};
use kryst::utils::convergence::ConvergedReason;

fn mpi_test_guard() -> MutexGuard<'static, ()> {
    static MPI_TEST_MUTEX: OnceLock<Mutex<()>> = OnceLock::new();
    MPI_TEST_MUTEX
        .get_or_init(|| Mutex::new(()))
        .lock()
        .expect("mpi test mutex poisoned")
}

fn local_rows_from_global(
    global: &CsrMatrix<f64>,
    row_start: usize,
    n_local: usize,
) -> CsrMatrix<f64> {
    let mut row_ptr = Vec::with_capacity(n_local + 1);
    let mut col_idx = Vec::new();
    let mut values = Vec::new();
    row_ptr.push(0);
    for i in 0..n_local {
        let (cols, vals) = global.row(row_start + i);
        col_idx.extend_from_slice(cols);
        values.extend_from_slice(vals);
        row_ptr.push(col_idx.len());
    }
    CsrMatrix::from_csr(n_local, global.ncols(), row_ptr, col_idx, values)
}

fn make_dist_poisson(comm: &UniverseComm, n_per: usize) -> DistCsrOp {
    let rank = comm.rank();
    let size = comm.size();
    let n_global = n_per * size;
    let row_start = rank * n_per;
    let global = fixtures::csr_poisson_1d(n_global);
    let local = local_rows_from_global(&global, row_start, n_per);
    let part_prefix: Vec<usize> = (0..=size).map(|p| p * n_per).collect();
    DistCsrOp::from_local_rows(n_global, row_start, &local, &part_prefix, comm.clone())
        .expect("dist csr")
}

fn make_dist_with_partition(
    comm: &UniverseComm,
    part_prefix: &[usize],
    global: &CsrMatrix<f64>,
) -> DistCsrOp {
    let rank = comm.rank();
    let row_start = part_prefix[rank];
    let n_local = part_prefix[rank + 1] - row_start;
    let local = local_rows_from_global(global, row_start, n_local);
    DistCsrOp::from_local_rows(global.nrows(), row_start, &local, part_prefix, comm.clone())
        .expect("dist csr")
}

fn solve_with_pc(
    solver_type: SolverType,
    dist: Arc<DistCsrOp>,
    rhs: &[f64],
    ksp_opts: &KspOptions,
    pc_opts: &PcOptions,
) -> ConvergedReason {
    let mut ksp = KspContext::new();
    ksp.set_type(solver_type).expect("set solver");
    ksp.set_operators(dist, None);
    ksp.set_from_all_options(ksp_opts, pc_opts)
        .expect("set options");
    ksp.setup().expect("ksp setup");

    let mut x = vec![0.0; rhs.len()];
    let stats = ksp.solve(rhs, &mut x).expect("solve");
    stats.reason
}

#[test]
fn mpi_matrix_route_policy_solver_and_pc_combinations() {
    let _guard = mpi_test_guard();
    let comm = UniverseComm::Mpi(Arc::new(MpiComm::new()));
    if comm.size() <= 1 {
        return;
    }

    let n_per = 6;
    let dist = Arc::new(make_dist_poisson(&comm, n_per));
    let rhs = vec![1.0; n_per];

    let cases = [
        (SolverType::Gmres, "jacobi", None, None),
        (SolverType::Gmres, "block_jacobi", None, None),
        (SolverType::Cg, "jacobi", None, None),
        (SolverType::Gmres, "mg", Some("block_jacobi"), Some("ilu")),
    ];

    for (solver_type, pc_type, pc_global, pc_local) in cases {
        for route in ["native", "adapted"] {
            let ksp_opts = KspOptions {
                ksp_type: Some(match solver_type {
                    SolverType::Cg => "cg".to_string(),
                    _ => "gmres".to_string(),
                }),
                rtol: Some(1e-10),
                atol: Some(1e-12),
                maxits: Some(250),
                ..Default::default()
            };
            let mut pc_opts = PcOptions {
                pc_type: Some(pc_type.to_string()),
                pc_dist_route: Some(route.to_string()),
                ..Default::default()
            };
            pc_opts.pc_global = pc_global.map(ToString::to_string);
            pc_opts.pc_local = pc_local.map(ToString::to_string);

            if route == "adapted" && pc_type == "mg" && pc_global.is_none() {
                let mut ksp = KspContext::new();
                ksp.set_type(solver_type).expect("set solver");
                ksp.set_operators(dist.clone(), None);
                ksp.set_from_all_options(&ksp_opts, &pc_opts)
                    .expect("set opts");
                let err = ksp
                    .setup()
                    .expect_err("adapted local-only mg requires explicit fallback chain");
                assert!(
                    err.to_string().contains("pc_dist_route native")
                        || err.to_string().contains("distributed operator detected"),
                    "unexpected error for route={route} pc={pc_type}: {err}"
                );
                continue;
            }

            let reason = solve_with_pc(solver_type, dist.clone(), &rhs, &ksp_opts, &pc_opts);
            assert!(
                matches!(
                    reason,
                    ConvergedReason::ConvergedRtol | ConvergedReason::ConvergedAtol
                ),
                "expected convergence for route={route}, solver={solver_type:?}, pc={pc_type}; got {reason:?}"
            );
        }
    }
}

#[test]
fn mpi_stress_empty_rank_nnz_imbalance_and_disconnected_partitions() {
    let _guard = mpi_test_guard();
    let comm = UniverseComm::Mpi(Arc::new(MpiComm::new()));
    if comm.size() <= 1 {
        return;
    }

    let size = comm.size();
    let rank = comm.rank();

    // Empty ranks: make fewer rows than ranks.
    let n_global_small = (size - 1).max(1);
    let mut part_prefix = vec![0usize; size + 1];
    for r in 0..size {
        part_prefix[r + 1] = part_prefix[r] + usize::from(r < n_global_small);
    }
    let global_small = fixtures::csr_poisson_1d(n_global_small);
    let dist_empty = Arc::new(make_dist_with_partition(&comm, &part_prefix, &global_small));
    let n_local = part_prefix[rank + 1] - part_prefix[rank];
    let rhs_empty = vec![1.0; n_local];

    let reason_empty = solve_with_pc(
        SolverType::Gmres,
        dist_empty,
        &rhs_empty,
        &KspOptions {
            ksp_type: Some("gmres".to_string()),
            maxits: Some(100),
            ..Default::default()
        },
        &PcOptions {
            pc_type: Some("jacobi".to_string()),
            pc_dist_route: Some("native".to_string()),
            ..Default::default()
        },
    );
    assert!(matches!(
        reason_empty,
        ConvergedReason::ConvergedRtol | ConvergedReason::ConvergedAtol
    ));

    // NNZ imbalance + disconnected partitions: block diagonal with uneven block sizes.
    let block_a = 3usize;
    let block_b = (2 * size).max(3);
    let n_global = block_a + block_b;
    let mut row_ptr = Vec::with_capacity(n_global + 1);
    let mut col_idx = Vec::new();
    let mut vals = Vec::new();
    row_ptr.push(0);
    for i in 0..n_global {
        let (lo, hi) = if i < block_a {
            (0, block_a)
        } else {
            (block_a, n_global)
        };
        col_idx.push(i);
        vals.push(4.0);
        if i > lo {
            col_idx.push(i - 1);
            vals.push(-1.0);
        }
        if i + 1 < hi {
            col_idx.push(i + 1);
            vals.push(-1.0);
        }
        if i >= block_a {
            for j in (block_a..n_global).step_by(3) {
                if j != i {
                    col_idx.push(j);
                    vals.push(-0.05);
                }
            }
        }
        row_ptr.push(col_idx.len());
    }
    let global = CsrMatrix::from_csr(n_global, n_global, row_ptr, col_idx, vals);
    let skewed_part: Vec<usize> = (0..=size)
        .map(|r| {
            if r == 0 {
                0
            } else if r == 1 {
                (n_global / 2).max(1)
            } else {
                (n_global / 2) + ((n_global - (n_global / 2)) * (r - 1)) / (size - 1).max(1)
            }
        })
        .collect();
    let dist_skewed = Arc::new(make_dist_with_partition(&comm, &skewed_part, &global));
    let n_local_skewed = skewed_part[rank + 1] - skewed_part[rank];
    let rhs_skewed = vec![1.0; n_local_skewed];
    let reason_skewed = solve_with_pc(
        SolverType::Gmres,
        dist_skewed,
        &rhs_skewed,
        &KspOptions {
            ksp_type: Some("gmres".to_string()),
            maxits: Some(200),
            ..Default::default()
        },
        &PcOptions {
            pc_type: Some("block_jacobi".to_string()),
            pc_dist_route: Some("native".to_string()),
            ..Default::default()
        },
    );
    assert!(matches!(
        reason_skewed,
        ConvergedReason::ConvergedRtol | ConvergedReason::ConvergedAtol
    ));
}

#[test]
fn mpi_fault_injection_native_setup_failure_and_replay_tokens() {
    let _guard = mpi_test_guard();
    let comm = UniverseComm::Mpi(Arc::new(MpiComm::new()));
    if comm.size() <= 1 {
        return;
    }

    let dist = Arc::new(make_dist_poisson(&comm, 5));
    let mut ksp = KspContext::new();
    ksp.set_type(SolverType::Gmres).expect("set gmres");
    ksp.set_operators(dist, None);
    ksp.set_from_all_options(
        &KspOptions {
            ksp_type: Some("gmres".to_string()),
            maxits: Some(20),
            ..Default::default()
        },
        &PcOptions {
            pc_type: Some("mg".to_string()),
            pc_dist_route: Some("adapted".to_string()),
            ..Default::default()
        },
    )
    .expect("set options");

    let err = ksp
        .setup()
        .expect_err("adapted route without global fallback should fail");
    assert!(
        err.to_string().contains("pc_dist_route native")
            || err.to_string().contains("distributed operator detected")
    );

    let diag = ksp.view();
    let replay = diag
        .solver_config
        .get("pc_dist_replay_tokens")
        .and_then(|v| v.as_object())
        .expect("replay token map should be present after setup path");
    assert!(
        replay.contains_key("native_setup") || replay.contains_key("configured_setup"),
        "expected setup replay token, got {replay:?}"
    );
}
