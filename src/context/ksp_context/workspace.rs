
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
    pub h_mem: Vec<f64>,
    pub cs: Vec<f64>,
    pub sn: Vec<f64>,
    pub g: Vec<f64>,
    pub blk_scratch: Vec<f64>,
    n: usize,
    m: usize,
    need_z: bool,
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

    pub fn ensure_size(&mut self, n: usize, m: usize, need_z: bool) {
        self.n = n;
        self.m = m;
        self.need_z = need_z;
        if self.tmp1.len() != n { self.tmp1.resize(n, 0.0); }
        if self.tmp2.len() != n { self.tmp2.resize(n, 0.0); }
        let v_len = (m + 1).checked_mul(n).expect("v_mem overflow");
        if self.v_mem.len() != v_len { self.v_mem.resize(v_len, 0.0); }
        if need_z {
            let z_len = m.checked_mul(n).expect("z_mem overflow");
            if self.z_mem.len() != z_len { self.z_mem.resize(z_len, 0.0); }
        } else {
            self.z_mem.clear();
        }
        let h_len = (m + 1).checked_mul(m).expect("h overflow");
        if self.h_mem.len() != h_len { self.h_mem.resize(h_len, 0.0); }
        if self.cs.len() != m { self.cs.resize(m, 0.0); }
        if self.sn.len() != m { self.sn.resize(m, 0.0); }
        if self.g.len() != m + 1 { self.g.resize(m + 1, 0.0); }
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

