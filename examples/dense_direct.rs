// Example: Solve a random SPD system using LU and QR direct solvers from kryst.

use faer::Mat;
use kryst::preconditioner::PcSide;
use kryst::solver::legacy::LinearSolver;
use kryst::solver::{LuSolver, QrSolver};
use rand::Rng;

fn main() {
    let comm = kryst::parallel::UniverseComm::NoComm(kryst::parallel::NoComm);
    let n = 10;
    // Build a random SPD matrix: A = MᵀM + I
    let mut rng = rand::thread_rng();
    let data: Vec<f64> = (0..n * n).map(|_| rng.r#gen()).collect();
    let m = Mat::from_fn(n, n, |i, j| data[j * n + i]);
    let m_t = m.transpose();
    // a = m^T * m
    let mut a = &m_t * &m;
    // a = a + I
    for i in 0..n {
        a[(i, i)] = a[(i, i)] + 1.0;
    }

    // Right-hand side
    let b: Vec<f64> = (0..n).map(|_| rng.r#gen()).collect();
    let mut x = vec![0.0; n];

    // LU solve
    let mut lus = LuSolver::new();
    let stats_lu = lus
        .solve(&a, None, &b, &mut x, PcSide::Left, &comm, None, None)
        .unwrap();
    println!("LU x = {:?}, stats = {:?}", x, stats_lu);

    // QR solve
    let mut qrs = QrSolver::new();
    let stats_qr = qrs
        .solve(&a, None, &b, &mut x, PcSide::Left, &comm, None, None)
        .unwrap();
    println!("QR x = {:?}, stats = {:?}", x, stats_qr);
}
