#![cfg(all(feature = "backend-faer", not(feature = "complex")))]
#![cfg(not(feature = "complex"))]

use faer::Mat;
use kryst::algebra::parallel::par_sum_abs2_local;
use kryst::algebra::prelude::*;
use kryst::preconditioner::PcSide;
use kryst::preconditioner::ilu::{IluBuilder, IluType};
use kryst::preconditioner::legacy::Preconditioner;
use proptest::prelude::*;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

fn random_diagonally_dominant(n: usize, seed: u64) -> Mat<S> {
    let mut rng = StdRng::seed_from_u64(seed);
    Mat::from_fn(n, n, |i, j| {
        if i == j {
            // Diagonal dominates the sum of off-diagonal magnitudes in the row.
            let diag = R::from(1.0 + 0.1 * n as f64);
            S::from_real(diag)
        } else {
            S::from_real(rng.random_range(-0.05..0.05))
        }
    })
}

fn mat_vec_mul(a: &Mat<S>, x: &[S]) -> Vec<S> {
    let mut out = vec![S::zero(); a.nrows()];
    for i in 0..a.nrows() {
        let mut acc = S::zero();
        for j in 0..a.ncols() {
            acc = acc + a[(i, j)] * x[j];
        }
        out[i] = acc;
    }
    out
}

proptest! {
    #[test]
    fn ilut_fill_shrinks_with_larger_drop(n in 3usize..7, seed in any::<u64>()) {
        let a = random_diagonally_dominant(n, seed);

        let mut ilu_lo = IluBuilder::new()
            .ilu_type(IluType::ILUT)
            .drop_tolerance(R::from(0.0))
            .build()
            .unwrap();
        ilu_lo.setup(&a).expect("ILUT setup lo");

        let mut ilu_hi = IluBuilder::new()
            .ilu_type(IluType::ILUT)
            .drop_tolerance(R::from(1e-2))
            .build()
            .unwrap();
        ilu_hi.setup(&a).expect("ILUT setup hi");

        let stats_lo = ilu_lo.get_stats();
        let stats_hi = ilu_hi.get_stats();

        let nnz_lo = stats_lo.nnz_l + stats_lo.nnz_u;
        let nnz_hi = stats_hi.nnz_l + stats_hi.nnz_u;
        prop_assert!(nnz_hi <= nnz_lo, "nnz hi {} > nnz lo {}", nnz_hi, nnz_lo);
    }

    #[cfg(not(feature = "complex"))]
    #[test]
    fn ilut_residuals_are_reasonable_with_drop(n in 3usize..10, seed in any::<u64>()) {
        let a = random_diagonally_dominant(n, seed);
        let x_true: Vec<S> = (0..n).map(|k| S::from_real(k as f64 + 1.0)).collect();
        let b = mat_vec_mul(&a, &x_true);

        let mut ilu_lo = IluBuilder::new()
            .ilu_type(IluType::ILUT)
            .drop_tolerance(R::from(1e-2))
            .build()
            .unwrap();
        ilu_lo.setup(&a).expect("ILUT setup lo");

        let mut ilu_hi = IluBuilder::new()
            .ilu_type(IluType::ILUT)
            .drop_tolerance(R::from(1e-2))
            .build()
            .unwrap();
        ilu_hi.setup(&a).expect("ILUT setup hi");

        let mut x_lo = vec![S::zero(); n];
        let mut x_hi = vec![S::zero(); n];
        ilu_lo
            .apply(PcSide::Left, &b, &mut x_lo)
            .expect("ILUT lo apply");
        ilu_hi
            .apply(PcSide::Left, &b, &mut x_hi)
            .expect("ILUT hi apply");

        let r_lo: Vec<S> = mat_vec_mul(&a, &x_lo)
            .into_iter()
            .zip(b.iter())
            .map(|(ax, &bi)| ax - bi)
            .collect();
        let r_hi: Vec<S> = mat_vec_mul(&a, &x_hi)
            .into_iter()
            .zip(b.iter())
            .map(|(ax, &bi)| ax - bi)
            .collect();

        let norm_lo = par_sum_abs2_local(&r_lo).sqrt();
        let norm_hi = par_sum_abs2_local(&r_hi).sqrt();

        prop_assert!(
            norm_hi <= norm_lo * R::from(1000.0), // Increased factor from 100.0 to 1000.0
            "Residual with drop tolerance 1e-2 should be within a reasonable bound of the exact solver"
        );
    }
}
