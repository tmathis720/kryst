use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::*};

pub struct CountingAlloc;
static ALLOCS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let _ = layout; // ignore size details
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

fn allocs() -> usize { ALLOCS.load(SeqCst) }

#[test]
fn apply_has_no_allocations() {
    use kryst::preconditioner::{Preconditioner, PcSide};
    let n = 128;
    let a = crate::fixtures::csr_poisson_1d(n);
    let mut pc = kryst::preconditioner::jacobi::Jacobi::new();
    pc.setup(&a).unwrap();
    let x = vec![1.0; n];
    let mut y = vec![0.0; n];
    let before = allocs();
    pc.apply(PcSide::Left, &x, &mut y).unwrap();
    let after = allocs();
    assert_eq!(before, after, "apply() performed heap allocations");
}

mod fixtures;

