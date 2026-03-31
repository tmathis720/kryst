#[cfg(feature = "complex")]
fn main() {
    eprintln!("dense_direct.rs is unavailable when built with --features complex");
}

// Example: Solve a random SPD system using LU and QR direct solvers from kryst.
// to run:
// cargo run --features=dense-direct,mpi --example dense_direct

#[cfg(feature = "dense-direct")]
use faer::Mat;
#[cfg(feature = "dense-direct")]
use kryst::preconditioner::PcSide;
#[cfg(feature = "dense-direct")]
use kryst::solver::legacy::LinearSolver;
#[cfg(feature = "dense-direct")]
use kryst::solver::{LuSolver, QrSolver};
#[cfg(feature = "dense-direct")]
use rand::rng;

#[cfg(feature = "dense-direct")]
#[cfg(not(feature = "complex"))]
fn main() {
    use kryst::matrix::utils;

    let comm = kryst::parallel::UniverseComm::NoComm(kryst::parallel::NoComm);
    let n = 10;
    // Build a random SPD matrix: A = MᵀM + I
    // let mut rng = rand::rngs::StdRng::seed_from_u64(0);
    let data: Vec<f64> = utils::random_rhs(n, 0);
    let m = Mat::from_fn(n, n, |i, j| data[j * n + i]);
    let m_t = m.transpose();
    // a = m^T * m
    let mut a = &m_t * &m;
    // a = a + I
    for i in 0..n {
        a[(i, i)] = a[(i, i)] + 1.0;
    }

    // Right-hand side
    let b: Vec<f64> = utils::random_rhs(n, 0);
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

#[cfg(not(feature = "dense-direct"))]
#[cfg(not(feature = "complex"))]
fn main() {}
