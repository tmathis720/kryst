#[derive(Debug, Clone)]
#[derive(Default)]
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
    // Shared communication arenas
    pub send_arena: crate::utils::buffer_pool::BufferPool<u8>,
    pub recv_arena: crate::utils::buffer_pool::BufferPool<u8>,
    pub packet_arena: crate::utils::buffer_pool::BufferPool<u8>,
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


impl Workspace {
    pub fn new(n: usize) -> Self {
        let mut ws = Self::default();
        ws.tmp1.resize(n, 0.0);
        ws.tmp2.resize(n, 0.0);
        ws.n = n;
        ws
    }

    /// Ensure communication buffers have enough bytes for upcoming operations.
    pub fn ensure_comm_bytes(&mut self, max_send: usize, max_recv: usize) {
        self.send_arena.ensure_len(max_send);
        self.recv_arena.ensure_len(max_recv);
    }

    #[inline]
    pub fn n(&self) -> usize {
        self.n
    }
    #[inline]
    pub fn m(&self) -> usize {
        self.m
    }
    #[inline]
    pub fn has_z(&self) -> bool {
        self.need_z
    }

    #[inline]
    pub fn ld_h(&self) -> usize {
        self.m + 1
    }

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
        &mut self.v_mem[off..off + n]
    }

    #[inline]
    pub fn z_col(&mut self, j: usize) -> &mut [f64] {
        debug_assert!(self.need_z && j < self.m);
        let n = self.n;
        let off = j.checked_mul(n).expect("z offset overflow");
        &mut self.z_mem[off..off + n]
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
        if a < b {
            (&mut lo_slice[..n], hi_slice)
        } else {
            (hi_slice, &mut lo_slice[..n])
        }
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
        if a < b {
            (&mut lo_slice[..n], hi_slice)
        } else {
            (hi_slice, &mut lo_slice[..n])
        }
    }

    // --- Composite view helpers -------------------------------------------------
    #[inline]
    pub fn v_and_z_mut(&mut self, j: usize) -> (&[f64], &mut [f64]) {
        debug_assert!(self.need_z && j < self.m);
        let n = self.n;
        let off = j * n;
        let vj: &[f64] = &self.v_mem[off..off + n];
        let zj: &mut [f64] = &mut self.z_mem[off..off + n];
        (vj, zj)
    }

    #[inline]
    pub fn tmp1_and_z_mut(&mut self, j: usize) -> (&[f64], &mut [f64]) {
        debug_assert!(self.need_z && j < self.m);
        let n = self.n;
        let tmp: &[f64] = &self.tmp1[..n];
        let z: &mut [f64] = &mut self.z_mem[j * n..(j + 1) * n];
        (tmp, z)
    }

    #[inline]
    pub fn tmp2_and_z_mut(&mut self, j: usize) -> (&[f64], &mut [f64]) {
        debug_assert!(self.need_z && j < self.m);
        let n = self.n;
        let tmp: &[f64] = &self.tmp2[..n];
        let z: &mut [f64] = &mut self.z_mem[j * n..(j + 1) * n];
        (tmp, z)
    }

    #[inline]
    pub fn z_and_tmp2_mut(&mut self, j: usize) -> (&[f64], &mut [f64]) {
        debug_assert!(self.need_z && j < self.m);
        let n = self.n;
        let z: &[f64] = &self.z_mem[j * n..(j + 1) * n];
        let tmp: &mut [f64] = &mut self.tmp2[..n];
        (z, tmp)
    }

    // --- Copy helpers -----------------------------------------------------------
    #[inline]
    pub fn copy_tmp2_into_vcol(&mut self, j: usize) {
        let n = self.n;
        let dst = &mut self.v_mem[j * n..(j + 1) * n];
        let src = &self.tmp2[..n];
        dst.copy_from_slice(src);
    }

    #[inline]
    pub fn copy_tmp1_into_vcol(&mut self, j: usize) {
        let n = self.n;
        let dst = &mut self.v_mem[j * n..(j + 1) * n];
        let src = &self.tmp1[..n];
        dst.copy_from_slice(src);
    }

    #[inline]
    pub fn copy_vcol_into_zcol(&mut self, j: usize) {
        debug_assert!(self.need_z && j < self.m);
        let n = self.n;
        let src = &self.v_mem[j * n..(j + 1) * n];
        let dst = &mut self.z_mem[j * n..(j + 1) * n];
        dst.copy_from_slice(src);
    }

    #[inline]
    pub fn copy_vcol_into_tmp1(&mut self, j: usize) {
        let n = self.n;
        let src = &self.v_mem[j * n..(j + 1) * n];
        self.tmp1[..n].copy_from_slice(src);
    }

    // --- Hessenberg helpers -----------------------------------------------------
    #[inline]
    pub fn h2_mut(&mut self, i: usize, j: usize) -> (&mut f64, &mut f64) {
        debug_assert!(i < self.m && j < self.m);
        let ld = self.ld_h();
        let base = j * ld + i;
        let (left, right) = self.h_mem.split_at_mut(base + 1);
        let hij = &mut left[base];
        let hij1 = &mut right[0];
        (hij, hij1)
    }

    #[inline]
    pub fn apply_prev_givens_to_col(&mut self, j: usize, upto: usize) {
        for i in 0..upto {
            let c = self.cs[i];
            let s = self.sn[i];
            let (hij, hij1) = self.h2_mut(i, j);
            let t = c * *hij + s * *hij1;
            *hij1 = -s * *hij + c * *hij1;
            *hij = t;
        }
    }

    #[inline]
    pub fn apply_final_givens_and_update_g(&mut self, j: usize) {
        let ld = self.ld_h();
        let hkk = self.h_mem[j * ld + j];
        let hk1k = self.h_mem[j * ld + j + 1];
        let (c, s) = if hk1k == 0.0 {
            (1.0, 0.0)
        } else {
            let r = (hkk * hkk + hk1k * hk1k).sqrt();
            (hkk / r, hk1k / r)
        };
        self.cs[j] = c;
        self.sn[j] = s;
        let (hjj, hj1j) = self.h2_mut(j, j);
        let t = c * *hjj + s * *hj1j;
        *hj1j = -s * *hjj + c * *hj1j;
        *hjj = t;
        let t = c * self.g[j] + s * self.g[j + 1];
        self.g[j + 1] = -s * self.g[j] + c * self.g[j + 1];
        self.g[j] = t;
    }
}

/// Grow vector to `need` length without zeroing. Never shrinks silently.
#[inline]
fn ensure_len(v: &mut Vec<f64>, need: usize) {
    if v.len() != need {
        if v.capacity() < need {
            v.reserve_exact(need - v.capacity());
        }
        unsafe {
            v.set_len(need);
        }
    }
}
