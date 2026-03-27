#![allow(deprecated)]

use crate::algebra::prelude::S;
use crate::algebra::scalar::KrystScalar;
use crate::config::options::PcOptions;
use crate::context::pc_context::{PcFactory, PcType};
use crate::matrix::op::CsrOp;
use crate::matrix::sparse::CsrMatrix;
use crate::preconditioner::approxinv::ApproxInv;
use crate::preconditioner::approxinv_csr::{ApproxInvKind, ApproxInvParams, SpaiCsr};
use crate::preconditioner::{PcDistributedSupport, PcSide, Preconditioner, SparsityPattern};
use std::sync::Arc;

fn poisson_1d_matrix() -> CsrMatrix<f64> {
    let row_ptr = vec![0, 2, 5, 8, 10];
    let col_idx = vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3];
    let values = vec![2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0];
    CsrMatrix::from_csr(4, 4, row_ptr, col_idx, values)
}

#[cfg(not(feature = "complex"))]
#[test]
fn legacy_approxinv_matches_csr_spai_apply() {
    let csr = Arc::new(poisson_1d_matrix());
    let op = CsrOp::new(csr.clone());

    let mut legacy = ApproxInv::<CsrOp<f64>, Vec<f64>, f64>::new(
        SparsityPattern::Auto,
        1e-12,
        10,
        1,
        100,
        8,
        1,
        0,
        false,
        false,
    );
    crate::preconditioner::Preconditioner::setup(&mut legacy, &op).expect("legacy setup");

    let mut csr_spai = SpaiCsr::new_with_params(ApproxInvParams {
        kind: ApproxInvKind::SPAI,
        // Legacy ApproxInv::Auto currently falls back to dense columns for CsrOp,
        // so mirror that behavior for parity in this migration test.
        levels: csr.nrows(),
        max_per_col: csr.nrows(),
        drop_tol: 0.0,
        reg: 1e-12,
        max_cond: 1e12,
        parallel: false,
    });
    csr_spai.setup(&op).expect("csr spai setup");

    let rhs = vec![
        S::from_real(1.0),
        S::from_real(-0.5),
        S::from_real(2.0),
        S::from_real(0.25),
    ];
    let mut y_legacy = vec![S::zero(); rhs.len()];
    let mut y_csr = vec![S::zero(); rhs.len()];

    crate::preconditioner::Preconditioner::apply(&legacy, PcSide::Left, &rhs, &mut y_legacy)
        .expect("legacy apply");
    csr_spai
        .apply(PcSide::Left, &rhs, &mut y_csr)
        .expect("csr apply");

    for (l, c) in y_legacy.iter().zip(y_csr.iter()) {
        assert!(
            (*l - *c).abs() < 1e-9,
            "legacy ({l:?}) and csr ({c:?}) diverged"
        );
    }
}

#[test]
fn pctype_approxinverse_uses_distributed_capability_for_all_modes() {
    for (kind, parallel) in [
        ("fsai", false),
        ("fsai", true),
        ("spai", false),
        ("spai", true),
    ] {
        let opts = PcOptions {
            pc_type: Some("approxinv".to_string()),
            approxinv_kind: Some(kind.to_string()),
            approxinv_parallel: Some(parallel),
            ..Default::default()
        };
        let pc = PcFactory::create_preconditioner(PcType::ApproxInverse, Some(&opts))
            .expect("approxinverse factory build");
        assert_eq!(pc.distributed_support(), PcDistributedSupport::Distributed);
    }
}
