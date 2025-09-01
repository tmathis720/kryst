
#[derive(Debug, Clone)]
pub struct Workspace {
    pub tmp1: Vec<f64>,
    pub tmp2: Vec<f64>,
    // Legacy buffers for solvers not yet migrated
    pub q: Vec<Vec<f64>>,
    pub z: Vec<Vec<f64>>,
    pub h: Vec<Vec<f64>>,
    pub v_mem: Vec<f64>,
    pub z_mem: Vec<f64>,
    // Column-major Hessenberg storage for GMRES/FGMRES
    pub h_mem: Vec<f64>,
    pub cs: Vec<f64>,
    pub sn: Vec<f64>,
    pub g: Vec<f64>,
    pub blk_scratch: Vec<f64>,
    n: usize,
    m: usize,
    need_z: bool,
}

/// Specification for sizing GMRES/FGMRES workspaces.
#[derive(Debug, Clone, Copy)]
pub struct GmresSpec {
    pub n: usize,
    pub m: usize,
    pub need_z: bool,
    pub block_s: usize,
}

impl Default for Workspace {
    fn default() -> Self {
        Self {
            tmp1: Vec::new(),
            tmp2: Vec::new(),
            q: Vec::new(),
            z: Vec::new(),
            h: Vec::new(),
            v_mem: Vec::new(),
            z_mem: Vec::new(),
            h_mem: Vec::new(),
            cs: Vec::new(),
            sn: Vec::new(),
            g: Vec::new(),
            blk_scratch: Vec::new(),
            n: 0,
            m: 0,
            need_z: false,
        }
    }
}

impl Workspace {
    pub fn new(n: usize) -> Self {
        let mut ws = Self::default();
        ws.tmp1.resize(n, 0.0);
        ws.tmp2.resize(n, 0.0);
        ws.n = n;
        ws
    }

    #[inline]
    pub fn n(&self) -> usize { self.n }
    #[inline]
    pub fn m(&self) -> usize { self.m }
    #[inline]
    pub fn has_z(&self) -> bool { self.need_z }

    /// Ensure capacity for a (F)GMRES run. Idempotent and allocation-friendly.
    pub fn acquire_gmres(&mut self, spec: GmresSpec) {
        // Remember shape for indexers
        self.n = spec.n;
        self.m = spec.m;
        self.need_z = spec.need_z;

        let n = spec.n;
        let m = spec.m;

        let v_len = (m + 1).checked_mul(n).expect("v_len overflow");
        let z_len = if spec.need_z {
            m.checked_mul(n).expect("z_len overflow")
        } else {
            0
        };
        let h_len = (m + 1).checked_mul(m).expect("h_len overflow");
        let g_len = m + 1;

        ensure_len(&mut self.tmp1, n);
        ensure_len(&mut self.tmp2, n);
        ensure_len(&mut self.v_mem, v_len);
        if spec.need_z {
            ensure_len(&mut self.z_mem, z_len);
        } else {
            self.z_mem.clear();
            self.z_mem.shrink_to_fit();
        }
        ensure_len(&mut self.h_mem, h_len);
        ensure_len(&mut self.cs, m);
        ensure_len(&mut self.sn, m);
        ensure_len(&mut self.g, g_len);

        if spec.block_s > 0 {
            ensure_len(&mut self.blk_scratch, n * spec.block_s);
        } else {
            self.blk_scratch.clear();
        }
    }

    #[inline]
    pub fn v_col(&mut self, j: usize) -> &mut [f64] {
        debug_assert!(j <= self.m);
        let n = self.n;
        let off = j.checked_mul(n).expect("v offset overflow");
        &mut self.v_mem[off .. off + n]
    }

    #[inline]
    pub fn z_col(&mut self, j: usize) -> &mut [f64] {
        debug_assert!(self.need_z && j < self.m);
        let n = self.n;
        let off = j.checked_mul(n).expect("z offset overflow");
        &mut self.z_mem[off .. off + n]
    }

    #[inline]
    pub fn h_at(&self, i: usize, j: usize) -> f64 {
        debug_assert!(i <= self.m && j < self.m);
        self.h_mem[j * (self.m + 1) + i]
    }
    #[inline]
    pub fn h_at_mut(&mut self, i: usize, j: usize) -> &mut f64 {
        debug_assert!(i <= self.m && j < self.m);
        let idx = j * (self.m + 1) + i;
        &mut self.h_mem[idx]
    }

    pub fn v_cols2(&mut self, a: usize, b: usize) -> (&mut [f64], &mut [f64]) {
        debug_assert!(a <= self.m && b <= self.m && a != b);
        let n = self.n;
        let (lo, hi) = if a < b { (a, b) } else { (b, a) };
        let lo_off = lo * n;
        let hi_off = hi * n;
        let (lo_part, rest) = self.v_mem.split_at_mut(hi_off);
        let (_, lo_slice) = lo_part.split_at_mut(lo_off);
        let (hi_slice, _) = rest.split_at_mut(n);
        if a < b { (&mut lo_slice[..n], hi_slice) } else { (hi_slice, &mut lo_slice[..n]) }
    }

    pub fn z_cols2(&mut self, a: usize, b: usize) -> (&mut [f64], &mut [f64]) {
        debug_assert!(self.need_z && a < self.m && b < self.m && a != b);
        let n = self.n;
        let (lo, hi) = if a < b { (a, b) } else { (b, a) };
        let lo_off = lo * n;
        let hi_off = hi * n;
        let (lo_part, rest) = self.z_mem.split_at_mut(hi_off);
        let (_, lo_slice) = lo_part.split_at_mut(lo_off);
        let (hi_slice, _) = rest.split_at_mut(n);
        if a < b { (&mut lo_slice[..n], hi_slice) } else { (hi_slice, &mut lo_slice[..n]) }
    }
}

/// Grow vector to `need` length without zeroing. Never shrinks silently.
#[inline]
fn ensure_len(v: &mut Vec<f64>, need: usize) {
    if v.len() != need {
        if v.capacity() < need {
            v.reserve_exact(need - v.capacity());
        }
        unsafe { v.set_len(need); }
    }
}

