//! FieldSplit block-system benchmark/demo.
//!
//! Demonstrates:
//! - Schur and composite split modes
//! - Per-field prefixes with sub-KSP/sub-PC options
//! - Setup reuse and repeated-apply timings

#[cfg(not(feature = "backend-faer"))]
fn main() {
    eprintln!("fieldsplit_block_system_benchmark requires the backend-faer feature.");
}

#[cfg(feature = "backend-faer")]
use kryst::KError;
#[cfg(feature = "backend-faer")]
use kryst::config::options::PcOptions;
#[cfg(feature = "backend-faer")]
use kryst::context::pc_context::PcFactory;
#[cfg(feature = "backend-faer")]
use kryst::matrix::op::CsrOp;
#[cfg(feature = "backend-faer")]
use kryst::matrix::sparse::CsrMatrix;
#[cfg(feature = "backend-faer")]
use kryst::prelude::*;
#[cfg(feature = "backend-faer")]
use std::sync::Arc;
#[cfg(feature = "backend-faer")]
use std::time::Instant;

#[cfg(feature = "backend-faer")]
fn coupled_block_system(nu: usize, np: usize) -> CsrMatrix<S> {
    let n = nu + np;
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::new();
    let mut values = Vec::new();
    row_ptr.push(0);

    for i in 0..n {
        if i < nu {
            if i > 0 {
                col_idx.push(i - 1);
                values.push(S::from_real(-1.0));
            }
            col_idx.push(i);
            values.push(S::from_real(4.0));
            if i + 1 < nu {
                col_idx.push(i + 1);
                values.push(S::from_real(-1.0));
            }
            if np > 0 {
                col_idx.push(nu + (i % np));
                values.push(S::from_real(0.2));
            }
        } else {
            let p = i - nu;
            col_idx.push(p % nu);
            values.push(S::from_real(0.15));
            col_idx.push(i);
            values.push(S::from_real(2.0));
        }
        row_ptr.push(col_idx.len());
    }

    CsrMatrix::from_csr(n, n, row_ptr, col_idx, values)
}

#[cfg(feature = "backend-faer")]
fn run_case(
    name: &str,
    opts: &PcOptions,
    op: &CsrOp,
    rhs: &[S],
    repeats: usize,
) -> Result<(), KError> {
    let mut pc = PcFactory::create_from_options(opts)?;

    let setup_t0 = Instant::now();
    pc.setup(op)?;
    let setup_ms = setup_t0.elapsed().as_secs_f64() * 1e3;

    let mut out = vec![S::zero(); rhs.len()];
    let apply_t0 = Instant::now();
    for _ in 0..repeats {
        pc.apply(PcSide::Left, rhs, &mut out)?;
    }
    let apply_ms = apply_t0.elapsed().as_secs_f64() * 1e3;

    // Trigger setup-reuse path.
    let reuse_t0 = Instant::now();
    pc.setup(op)?;
    let reuse_ms = reuse_t0.elapsed().as_secs_f64() * 1e3;

    println!(
        "{name:>28}: setup={setup_ms:8.3} ms  apply({repeats})={apply_ms:8.3} ms  setup_reuse={reuse_ms:8.3} ms"
    );
    Ok(())
}

#[cfg(feature = "backend-faer")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (nu, np) = (160, 80);
    let a = coupled_block_system(nu, np);
    let op = CsrOp::new(Arc::new(a));
    let rhs: Vec<S> = (0..(nu + np))
        .map(|i| S::from_real(1.0 + (i % 7) as f64))
        .collect();

    println!("FieldSplit block benchmark (n_u={nu}, n_p={np})");

    let composite = PcOptions {
        pc_type: Some("fieldsplit".into()),
        pc_fieldsplit_block_sizes: Some(vec![nu, np]),
        pc_fieldsplit_type: Some("composite_multiplicative".into()),
        pc_fieldsplit_child_pc_type: Some("jacobi".into()),
        pc_fieldsplit_extraction: Some("cached".into()),
        ..Default::default()
    };

    let schur_args = vec![
        "-pc_type".to_string(),
        "fieldsplit".to_string(),
        "-pc_fieldsplit_block_sizes".to_string(),
        format!("{nu},{np}"),
        "-pc_fieldsplit_type".to_string(),
        "schur".to_string(),
        "-pc_fieldsplit_schur_fact_type".to_string(),
        "full".to_string(),
        "-pc_fieldsplit_schur_precondition".to_string(),
        "full_matfree".to_string(),
        "-pc_fieldsplit_prefixes".to_string(),
        "pc_fieldsplit_0_,pc_fieldsplit_1_".to_string(),
        "-pc_fieldsplit_0_pc_type".to_string(),
        "ksp".to_string(),
        "-pc_fieldsplit_0_pc_ksp_ksp_type".to_string(),
        "gmres".to_string(),
        "-pc_fieldsplit_0_pc_ksp_pc_type".to_string(),
        "jacobi".to_string(),
        "-pc_fieldsplit_1_pc_type".to_string(),
        "jacobi".to_string(),
    ];
    let schur_refs = schur_args.iter().map(String::as_str).collect::<Vec<_>>();
    let schur = PcOptions::from_args(&schur_refs)?;

    run_case("composite_multiplicative", &composite, &op, &rhs, 64)?;
    run_case("schur(full/full_matfree)", &schur, &op, &rhs, 64)?;

    Ok(())
}
