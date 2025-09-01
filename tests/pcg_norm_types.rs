use faer::Mat;
use kryst::parallel::UniverseComm;
use kryst::preconditioner::PcSide;
use kryst::solver::pcg::CgNormType;
use kryst::solver::{LinearSolver, PcgSolver};
use std::sync::{Arc, Mutex};

#[test]
fn natural_norm_matches_preconditioned() {
    let comm = UniverseComm::NoComm(kryst::parallel::NoComm);
    // Simple SPD 2x2 system
    let mut a = Mat::<f64>::zeros(2, 2);
    a[(0, 0)] = 4.0;
    a[(0, 1)] = 1.0;
    a[(1, 0)] = 1.0;
    a[(1, 1)] = 3.0;
    let b = [1.0, 2.0];
    let mut x1 = [0.0, 0.0];
    let mut x2 = [0.0, 0.0];

    // Preconditioned norm (default)
    let mut solver_pre = PcgSolver::new(1e-8, 10);
    let hist_pre = Arc::new(Mutex::new(Vec::new()));
    let m_pre = {
        let hist = hist_pre.clone();
        Box::new(move |i: usize, r: f64| hist.lock().unwrap().push((i, r)))
    };
    let _stats_pre = solver_pre
        .solve(
            &a,
            None,
            &b,
            &mut x1,
            PcSide::Left,
            &comm,
            Some(&[m_pre]),
            None,
        )
        .unwrap();

    // Natural norm
    let mut solver_nat = PcgSolver::new(1e-8, 10).with_norm(CgNormType::Natural);
    let hist_nat = Arc::new(Mutex::new(Vec::new()));
    let m_nat = {
        let hist = hist_nat.clone();
        Box::new(move |i: usize, r: f64| hist.lock().unwrap().push((i, r)))
    };
    let _stats_nat = solver_nat
        .solve(
            &a,
            None,
            &b,
            &mut x2,
            PcSide::Left,
            &comm,
            Some(&[m_nat]),
            None,
        )
        .unwrap();

    let h_pre = hist_pre.lock().unwrap();
    let h_nat = hist_nat.lock().unwrap();
    assert_eq!(h_pre.len(), h_nat.len());
    for (a, b) in h_pre.iter().zip(h_nat.iter()) {
        assert!((a.1 - b.1).abs() < 1e-12);
    }
}
