use kryst::solver::superlu_dist::{Panel, PivotingStrategy};
use std::time::Instant;

#[test]
#[ignore]
fn faer_trsm_gemm_microbench() {
    let m = 256usize;
    let n = 64usize;
    let data: Vec<f64> = (0..m * n).map(|i| (i % 97) as f64 / 97.0).collect();
    let mut panel_old = Panel {
        width: n,
        height: m,
        data: data.clone(),
        row_indices: (0..m).collect(),
        col_start: 0,
    };
    let mut panel_new = Panel {
        width: n,
        height: m,
        data,
        row_indices: (0..m).collect(),
        col_start: 0,
    };
    // Naive triple-loop LU factorization (old style)
    let start_old = Instant::now();
    {
        let m = panel_old.height;
        let n = panel_old.width;
        for j in 0..n.min(m) {
            let diag = panel_old.data[j * m + j];
            for i in j + 1..m {
                panel_old.data[j * m + i] /= diag;
            }
            for k in j + 1..n {
                let val = panel_old.data[k * m + j];
                for i in j + 1..m {
                    let idx = k * m + i;
                    panel_old.data[idx] -= panel_old.data[j * m + i] * val;
                }
            }
        }
    }
    let old_time = start_old.elapsed();
    // New faer-based factorization
    let start_new = Instant::now();
    panel_new
        .factorize_lu(1.0, PivotingStrategy::Static)
        .unwrap();
    let new_time = start_new.elapsed();
    assert!(new_time < old_time);
}
