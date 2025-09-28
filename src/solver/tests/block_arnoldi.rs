use crate::context::ksp_context::Workspace;
use crate::parallel::{NoComm, UniverseComm};
use crate::solver::block::arnoldi::{ArnoldiOutput, block_arnoldi_step};
use crate::solver::block::block_vec::BlockVec;
use crate::solver::block::kernels::{gram_pxp, tall_t_times_block};

fn setup_basis() -> Vec<BlockVec> {
    let mut v0 = BlockVec::new(4, 2);
    v0.col_mut(0).copy_from_slice(&[1.0, 0.0, 0.0, 0.0]);
    v0.col_mut(1).copy_from_slice(&[0.0, 1.0, 0.0, 0.0]);
    vec![v0]
}

fn build_candidate() -> (BlockVec, BlockVec) {
    let mut w = BlockVec::new(4, 2);
    w.col_mut(0).copy_from_slice(&[0.5, -0.2, 1.0, 0.1]);
    w.col_mut(1).copy_from_slice(&[0.25, 0.4, -0.1, 0.9]);
    (w.clone(), w)
}

fn apply_step(basis: Vec<BlockVec>, mut w: BlockVec) -> (ArnoldiOutput, BlockVec) {
    let comm = UniverseComm::NoComm(NoComm);
    let mut workspace = Workspace::new(4);
    let original = w.clone();
    let output = block_arnoldi_step(&basis, &mut w, &comm, &mut workspace, 1e8)
        .expect("block Arnoldi step should succeed");
    // Verify Arnoldi identity: basis^T * original should match coeffs
    let mut columns: Vec<&[f64]> = Vec::new();
    for block in &basis {
        for col in 0..2 {
            columns.push(block.col(col));
        }
    }
    let mut actual = vec![0.0; columns.len() * 2];
    tall_t_times_block(&columns, &original, &mut actual);
    for (idx, block_coeffs) in output.coeffs.chunks(4).enumerate() {
        for row in 0..2 {
            for col in 0..2 {
                let expected = actual[(idx * 2 + row) * 2 + col];
                assert!((block_coeffs[row * 2 + col] - expected).abs() < 1e-12);
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
    let mut gram = vec![0.0; 4];
    gram_pxp(&q_block, &q_block, &mut gram);
    let tol = 2e-3;
    assert!((gram[0] - 1.0).abs() < tol);
    assert!((gram[3] - 1.0).abs() < tol);
    assert!(gram[1].abs() < tol);
    assert!(gram[2].abs() < tol);

    // Should be orthogonal to existing basis vectors
    for block in &basis {
        for bcol in 0..2 {
            for qcol in 0..2 {
                let mut dot = 0.0;
                for (&bi, &qi) in block.col(bcol).iter().zip(q_block.col(qcol).iter()) {
                    dot += bi * qi;
                }
                assert!(dot.abs() < 1e-12);
            }
        }
    }

    // R block should be upper triangular with positive diagonal
    assert!(output.r_block[0] > 0.0);
    assert!(output.r_block[3] > 0.0);
    assert!(output.r_block[2].abs() < 1e-12);
}
