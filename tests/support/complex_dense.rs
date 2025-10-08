#![cfg(feature = "complex")]

use kryst::algebra::blas::nrm2;
use kryst::algebra::bridge::BridgeScratch;
use kryst::algebra::prelude::*;
use kryst::ops::klinop::KLinOp;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Dense operator used by complex solver regression tests.
#[derive(Clone, Debug)]
pub struct DenseOp {
    n: usize,
    data: Vec<S>,
}

impl DenseOp {
    pub fn new(n: usize, data: Vec<S>) -> Self {
        Self { n, data }
    }

    pub fn size(&self) -> usize {
        self.n
    }

    pub fn apply(&self, x: &[S], y: &mut [S]) {
        debug_assert_eq!(x.len(), self.n);
        debug_assert_eq!(y.len(), self.n);
        for i in 0..self.n {
            let mut acc = S::zero();
            let row = &self.data[i * self.n..(i + 1) * self.n];
            for (&aij, &xj) in row.iter().zip(x.iter()) {
                acc = acc + aij * xj;
            }
            y[i] = acc;
        }
    }

    pub fn residual_norm(&self, x: &[S], b: &[S]) -> f64 {
        let mut tmp = vec![S::zero(); self.n];
        self.apply(x, &mut tmp);
        for (ti, &bi) in tmp.iter_mut().zip(b.iter()) {
            *ti -= bi;
        }
        nrm2(&tmp)
    }

    pub fn data(&self) -> &[S] {
        &self.data
    }
}

impl KLinOp for DenseOp {
    type Scalar = S;

    fn dims(&self) -> (usize, usize) {
        (self.n, self.n)
    }

    fn matvec_s(&self, x: &[S], y: &mut [S], _scratch: &mut BridgeScratch) {
        self.apply(x, y);
    }
}

fn random_solution(rng: &mut StdRng, n: usize) -> Vec<S> {
    (0..n)
        .map(|_| {
            let re = rng.gen_range(-0.5..0.5);
            let im = rng.gen_range(-0.5..0.5);
            S::from_parts(re, im)
        })
        .collect()
}

/// Build a diagonally dominant complex system with a known solution.
pub fn diagonally_dominant_system(
    n: usize,
    seed: u64,
    row_scale: f64,
    diag_shift: f64,
) -> (DenseOp, Vec<S>, Vec<S>) {
    let mut rng = StdRng::seed_from_u64(seed + n as u64);
    let mut data = vec![S::zero(); n * n];
    for i in 0..n {
        let mut row_sum = 0.0;
        for j in 0..n {
            if i == j {
                continue;
            }
            let re = rng.gen_range(-row_scale..row_scale);
            let im = rng.gen_range(-row_scale..row_scale);
            let val = S::from_parts(re, im);
            data[i * n + j] = val;
            row_sum += val.abs();
        }
        let diag_re = diag_shift + row_sum + rng.r#gen::<f64>().abs();
        let diag_im = rng.gen_range(-row_scale..row_scale);
        data[i * n + i] = S::from_parts(diag_re, diag_im);
    }

    let op = DenseOp::new(n, data);
    let x_true = random_solution(&mut rng, n);
    let mut b = vec![S::zero(); n];
    op.apply(&x_true, &mut b);
    (op, x_true, b)
}

/// Build a Hermitian positive definite system suitable for CG/MINRES regressions.
pub fn hermitian_pos_def_system(n: usize, seed: u64, shift: f64) -> (DenseOp, Vec<S>, Vec<S>) {
    let mut rng = StdRng::seed_from_u64(seed + n as u64);
    let mut base = vec![S::zero(); n * n];
    for val in base.iter_mut() {
        let re = rng.gen_range(-0.4..0.4);
        let im = rng.gen_range(-0.4..0.4);
        *val = S::from_parts(re, im);
    }

    let mut data = vec![S::zero(); n * n];
    for i in 0..n {
        for j in 0..n {
            let mut sum = S::zero();
            for k in 0..n {
                let bih = base[k * n + i].conj();
                let bj = base[k * n + j];
                sum = sum + bih * bj;
            }
            if i == j {
                sum = sum + S::from_real(shift);
            }
            data[i * n + j] = sum;
        }
    }

    for i in 0..n {
        for j in 0..i {
            let conj = data[j * n + i].conj();
            data[i * n + j] = conj;
        }
    }

    let op = DenseOp::new(n, data);
    let x_true = random_solution(&mut rng, n);
    let mut b = vec![S::zero(); n];
    op.apply(&x_true, &mut b);
    (op, x_true, b)
}
