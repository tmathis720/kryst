#![cfg(feature = "backend-faer")]
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::*};

use kryst::algebra::prelude::*;

pub struct CountingAlloc;
static ALLOCS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let _ = layout;
        ALLOCS.fetch_add(1, AcqRel);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let _ = (ptr, layout);
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static A: CountingAlloc = CountingAlloc;

fn allocs() -> usize {
    ALLOCS.load(SeqCst)
}

#[cfg(not(feature = "complex"))]
#[test]
fn ilu_apply_has_no_allocations() {
    use kryst::preconditioner::PcSide;
    use kryst::preconditioner::ilu::{IluBuilder, IluType, TriSolveType};
    use kryst::preconditioner::legacy::Preconditioner;

    let n = 32;
    let a = faer::Mat::from_fn(n, n, |i, j| {
        if i == j {
            R::from(4.0)
        } else if (i as i32 - j as i32).abs() == 1 {
            R::from(-1.0)
        } else {
            R::default()
        }
    });

    let mut ilu = IluBuilder::new()
        .ilu_type(IluType::ILU0)
        .triangular_solve(TriSolveType::Exact)
        .build()
        .unwrap();
    ilu.setup(&a).unwrap();

    let x = vec![R::from(1.0); n];
    let mut y = vec![R::default(); n];

    let before = allocs();
    ilu.apply(PcSide::Left, &x, &mut y).unwrap();
    let after = allocs();
    assert_eq!(before, after, "apply() performed heap allocations");
}

#[cfg(all(feature = "complex", feature = "complex_ilu"))]
#[test]
fn ilu_csr_complex_native_apply_mut_has_no_allocations() {
    use kryst::matrix::sparse::CsrMatrix;
    use kryst::preconditioner::ilu_csr::{IluCsr, IluCsrConfig, IluKind};
    use kryst::preconditioner::{PcSide, Preconditioner};

    let n = 32;
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::with_capacity(3 * n - 2);
    let mut values = Vec::with_capacity(3 * n - 2);
    row_ptr.push(0);
    for i in 0..n {
        if i > 0 {
            col_idx.push(i - 1);
            values.push(S::from_parts(-1.0, 0.25));
        }
        col_idx.push(i);
        values.push(S::from_parts(4.0, 0.5));
        if i + 1 < n {
            col_idx.push(i + 1);
            values.push(S::from_parts(-1.0, -0.25));
        }
        row_ptr.push(col_idx.len());
    }
    let a = CsrMatrix::from_csr(n, n, row_ptr, col_idx, values);

    let mut cfg = IluCsrConfig::default();
    cfg.kind = IluKind::Ilu0;
    cfg.level_sched = false;
    let mut ilu = IluCsr::new_with_config(cfg);
    ilu.setup(&a).unwrap();

    let x = vec![S::from_parts(1.0, -0.5); n];
    let mut y = vec![S::zero(); n];

    // Exercise any one-time runtime initialization before measuring steady-state apply.
    ilu.apply_mut(PcSide::Left, &x, &mut y).unwrap();

    let before = allocs();
    ilu.apply_mut(PcSide::Left, &x, &mut y).unwrap();
    let after = allocs();
    assert_eq!(
        before, after,
        "complex apply_mut() performed heap allocations"
    );
}

#[cfg(all(feature = "complex", feature = "complex_ilu"))]
#[test]
fn ilu_csr_complex_degraded_apply_mut_has_no_allocations() {
    use kryst::matrix::sparse::CsrMatrix;
    use kryst::preconditioner::ilu_csr::{IluCsr, IluCsrConfig, IluKind};
    use kryst::preconditioner::{PcSide, Preconditioner};

    let n = 32;
    let row_ptr: Vec<_> = (0..=n).collect();
    let col_idx: Vec<_> = (0..n).collect();
    let values = (0..n)
        .map(|i| S::from_parts(2.0 + i as f64 * 0.01, 0.5))
        .collect();
    let a = CsrMatrix::from_csr(n, n, row_ptr, col_idx, values);

    let mut cfg = IluCsrConfig::default();
    cfg.kind = IluKind::Ilu0;
    cfg.level_sched = false;
    let mut ilu = IluCsr::new_with_config(cfg);
    ilu.set_complex_force_degraded(true);
    ilu.setup(&a).unwrap();

    let x = vec![S::from_parts(1.0, -0.5); n];
    let mut y = vec![S::zero(); n];

    // Exercise any one-time runtime initialization before measuring steady-state apply.
    ilu.apply_mut(PcSide::Left, &x, &mut y).unwrap();

    let before = allocs();
    ilu.apply_mut(PcSide::Left, &x, &mut y).unwrap();
    let after = allocs();
    assert_eq!(
        before, after,
        "complex degraded apply_mut() performed heap allocations"
    );
}
