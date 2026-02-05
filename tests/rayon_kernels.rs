#[cfg(feature = "rayon")]
mod rayon_tests {
    use kryst::algebra::parallel::{dot_conj_local_with_mode, par_dot_conj_local};
    use kryst::algebra::parallel_cfg::{parallel_tune, set_parallel_tune, ParallelTune};
    use kryst::algebra::prelude::*;
    use kryst::matrix::spmv::{spmv_csr_parallel, spmv_csr_serial};
    use kryst::matrix::sparse::CsrMatrix;
    use kryst::parallel::set_global_reduction_mode_scoped;
    use kryst::reduction::ReproMode;
    use rayon::ThreadPoolBuilder;

    struct TuneGuard {
        prev: ParallelTune,
    }

    impl TuneGuard {
        fn new(tune: ParallelTune) -> Self {
            let prev = parallel_tune();
            set_parallel_tune(tune);
            Self { prev }
        }
    }

    impl Drop for TuneGuard {
        fn drop(&mut self) {
            set_parallel_tune(self.prev);
        }
    }

    #[test]
    fn parallel_spmv_and_repro_vector_kernel_deterministic() {
        let pool = ThreadPoolBuilder::new().num_threads(2).build().unwrap();
        pool.install(|| {
            let _repro = set_global_reduction_mode_scoped(ReproMode::Deterministic);
            let _tune = TuneGuard::new(ParallelTune {
                min_len_vec: 1,
                min_rows_spmv: 1,
                chunk_rows_spmv: 1,
                min_rows_ilu_factorization: 1,
                min_rows_ilu_triangular: 1,
                min_rows_asm_apply: 1,
            });

            let a = CsrMatrix::from_csr(
                4,
                4,
                vec![0, 2, 4, 6, 8],
                vec![0, 1, 1, 2, 2, 3, 0, 3],
                vec![
                    S::from_real(4.0),
                    S::from_real(1.0),
                    S::from_real(-1.0),
                    S::from_real(3.0),
                    S::from_real(2.0),
                    S::from_real(1.0),
                    S::from_real(0.5),
                    S::from_real(2.0),
                ],
            );
            let x = vec![S::one(); 4];
            let mut y_par = vec![S::zero(); 4];
            let mut y_ser = vec![S::zero(); 4];
            spmv_csr_parallel(&a, &x, &mut y_par).unwrap();
            spmv_csr_serial(&a, &x, &mut y_ser).unwrap();
            assert_eq!(y_par, y_ser);

            let x_vec: Vec<S> = (0..2048)
                .map(|i| S::from_real((i as f64).sin()))
                .collect();
            let y_vec: Vec<S> = (0..2048)
                .map(|i| S::from_real((i as f64).cos()))
                .collect();
            let expected = dot_conj_local_with_mode(&x_vec, &y_vec, ReproMode::Deterministic);
            let actual = par_dot_conj_local(&x_vec, &y_vec);
            let diff = (expected - actual).abs();
            assert!(diff < 1e-12, "diff={diff:?}");
        });
    }

    #[cfg(all(feature = "backend-faer", not(feature = "complex")))]
    #[test]
    fn parallel_ilu_triangular_solve_matches_serial_factorization() {
        use kryst::preconditioner::ilu::{Ilu, IluConfig, TriSolveType};
        use kryst::preconditioner::legacy::Preconditioner as LegacyPreconditioner;
        use kryst::preconditioner::PcSide;

        let pool = ThreadPoolBuilder::new().num_threads(2).build().unwrap();
        pool.install(|| {
            let _tune = TuneGuard::new(ParallelTune {
                min_len_vec: 1,
                min_rows_spmv: 1,
                chunk_rows_spmv: 1,
                min_rows_ilu_factorization: 1,
                min_rows_ilu_triangular: 1,
                min_rows_asm_apply: 1,
            });

            let n = 12;
            let matrix = faer::Mat::from_fn(n, n, |i, j| {
                if i == j {
                    4.0
                } else if (i as i32 - j as i32).abs() == 1 {
                    -1.0
                } else {
                    0.0
                }
            });

            let mut cfg_serial = IluConfig::default();
            cfg_serial.triangular_solve = TriSolveType::Exact;
            cfg_serial.enable_parallel_factorization = false;
            cfg_serial.enable_parallel_triangular_solve = false;

            let mut cfg_par = IluConfig::default();
            cfg_par.triangular_solve = TriSolveType::Exact;
            // Parallel ILU(0) factorization in this implementation is block-diagonal by design,
            // so keep factorization serial and only validate the parallel triangular solve path.
            cfg_par.enable_parallel_factorization = false;
            cfg_par.enable_parallel_triangular_solve = true;
            cfg_par.parallel_chunk_size = 1;

            let mut ilu_serial = Ilu::new_with_config(cfg_serial).unwrap();
            let mut ilu_par = Ilu::new_with_config(cfg_par).unwrap();
            ilu_serial.setup(&matrix).unwrap();
            ilu_par.setup(&matrix).unwrap();

            let rhs = vec![1.0; n];
            let mut y_serial = vec![0.0; n];
            let mut y_par = vec![0.0; n];
            ilu_serial.apply(PcSide::Left, &rhs, &mut y_serial).unwrap();
            ilu_par.apply(PcSide::Left, &rhs, &mut y_par).unwrap();
            for (&s, &p) in y_serial.iter().zip(y_par.iter()) {
                assert!((s - p).abs() < 1e-10, "serial {s} vs parallel {p}");
            }
        });
    }
}
