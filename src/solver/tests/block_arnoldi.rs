#![cfg(feature = "backend-faer")]

use crate::algebra::blas::dot_conj;
use crate::algebra::prelude::*;
use crate::context::ksp_context::Workspace;
use crate::parallel::{NoComm, UniverseComm};
use crate::solver::block::arnoldi::{ArnoldiOutput, block_arnoldi_step};
use crate::solver::block::block_vec::BlockVec;
use crate::solver::block::kernels::{gram_pxp, tall_t_times_block};
use crate::{assert_s_close, testkit::ATOL};

fn setup_basis() -> Vec<BlockVec> {
    let mut v0 = BlockVec::new(4, 2);
    v0.col_mut(0)
        .copy_from_slice(&[S::from_real(1.0), S::zero(), S::zero(), S::zero()]);
    v0.col_mut(1)
        .copy_from_slice(&[S::zero(), S::from_real(1.0), S::zero(), S::zero()]);
    vec![v0]
}

fn build_candidate() -> (BlockVec, BlockVec) {
    let mut w = BlockVec::new(4, 2);
    w.col_mut(0).copy_from_slice(&[
        S::from_real(0.5),
        S::from_real(-0.2),
        S::from_real(1.0),
        S::from_real(0.1),
    ]);
    w.col_mut(1).copy_from_slice(&[
        S::from_real(0.25),
        S::from_real(0.4),
        S::from_real(-0.1),
        S::from_real(0.9),
    ]);
    (w.clone(), w)
}

fn apply_step(basis: Vec<BlockVec>, mut w: BlockVec) -> (ArnoldiOutput, BlockVec) {
    let comm = UniverseComm::NoComm(NoComm);
    let mut workspace = Workspace::new(4);
    let original = w.clone();
    let output = block_arnoldi_step(&basis, &mut w, &comm, &mut workspace, 1e8)
        .expect("block Arnoldi step should succeed");
    // Verify Arnoldi identity: basis^T * original should match coeffs
    let mut columns: Vec<&[S]> = Vec::new();
    for block in &basis {
        for col in 0..2 {
            columns.push(block.col(col));
        }
    }
    let mut actual: Vec<S> = vec![S::zero(); columns.len() * 2];
    tall_t_times_block(&columns, &original, &mut actual);
    for (idx, block_coeffs) in output.coeffs.chunks(4).enumerate() {
        for row in 0..2 {
            for col in 0..2 {
                let expected = actual[(idx * 2 + row) * 2 + col];
                assert_s_close!("block coeff", block_coeffs[row * 2 + col], expected);
            }
        }
    }
    (output, w)
}

#[test]
fn arnoldi_orthonormalises_block() {
    let basis = setup_basis();
    let (_, w_raw) = build_candidate();
    let (output, q_block) = apply_step(basis.clone(), w_raw);

    assert_eq!(output.coeffs.len(), basis.len() * 4);

    // New block should be orthonormal
    let mut gram: Vec<S> = vec![S::zero(); 4];
    gram_pxp(&q_block, &q_block, &mut gram);
    assert_s_close!("gram[0]", gram[0], S::one());
    assert_s_close!("gram[3]", gram[3], S::one());
    assert_s_close!("gram[1]", gram[1], S::zero());
    assert_s_close!("gram[2]", gram[2], S::zero());

    // Should be orthogonal to existing basis vectors
    for block in &basis {
        for bcol in 0..2 {
            for qcol in 0..2 {
                let dot = dot_conj(block.col(bcol), q_block.col(qcol));
                assert_s_close!("basis dot", dot, S::zero());
            }
        }
    }

    // R block should be upper triangular with positive diagonal
    assert!(output.r_block[0].real() > R::default());
    assert!(output.r_block[3].real() > R::default());
    assert!(output.r_block[2].abs() < ATOL);
}
