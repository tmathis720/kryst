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
