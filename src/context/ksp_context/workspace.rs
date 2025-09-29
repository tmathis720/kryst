use crate::solver::gmres::AugmentationPolicy;

#[derive(Debug, Clone, Default)]
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
    pub block_buf: Option<BlockVec>,
    pub tsqr: Option<TsqrWorkspace>,
    pub pipelined_w: Vec<f64>,
    pub pipelined_wtmp: Vec<f64>,
    pub pipelined_payload: Vec<f64>,
    pub gmres_sstep: Option<GmresSStepWorkspace>,
    pub gmres_recycle: RecyclingSpace,
    pub reduction: crate::utils::reduction::ReductOptions,
    // Shared communication arenas
    pub send_arena: crate::utils::buffer_pool::BufferPool<u8>,
    pub recv_arena: crate::utils::buffer_pool::BufferPool<u8>,
    pub packet_arena: crate::utils::buffer_pool::BufferPool<u8>,
    n: usize,
    m: usize,
    need_z: bool,
}

#[derive(Debug, Clone)]
pub struct RecyclingSpace {
    u: Vec<f64>,
    au: Vec<f64>,
    n: usize,
    rmax: usize,
    cols: usize,
    policy: AugmentationPolicy,
}

impl Default for RecyclingSpace {
    fn default() -> Self {
        Self {
            u: Vec::new(),
            au: Vec::new(),
            n: 0,
            rmax: 0,
            cols: 0,
            policy: AugmentationPolicy::None,
        }
    }
}

impl RecyclingSpace {
    pub fn configure(&mut self, n: usize, rmax: usize, policy: AugmentationPolicy) {
        if self.n != n || self.rmax != rmax {
            self.u.resize(n.saturating_mul(rmax), 0.0);
            self.au.resize(n.saturating_mul(rmax), 0.0);
            self.n = n;
            self.rmax = rmax;
            self.cols = 0;
        }
        self.policy = policy;
    }

    #[inline]
    pub fn policy(&self) -> AugmentationPolicy {
        self.policy.clone()
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.rmax
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn clear(&mut self) {
        self.cols = 0;
    }

    pub fn col(&self, j: usize) -> &[f64] {
        let n = self.n;
        &self.u[j * n..(j + 1) * n]
    }

    pub fn col_mut(&mut self, j: usize) -> &mut [f64] {
        let n = self.n;
        &mut self.u[j * n..(j + 1) * n]
    }

    pub fn a_col(&self, j: usize) -> &[f64] {
        let n = self.n;
        &self.au[j * n..(j + 1) * n]
    }

    pub fn a_col_mut(&mut self, j: usize) -> &mut [f64] {
        let n = self.n;
        &mut self.au[j * n..(j + 1) * n]
    }

    pub fn push_from(&mut self, u: &[f64], au: &[f64]) {
        if self.cols >= self.rmax {
            return;
        }
        let n = self.n;
        let dst_u = &mut self.u[self.cols * n..(self.cols + 1) * n];
        let dst_au = &mut self.au[self.cols * n..(self.cols + 1) * n];
        dst_u.copy_from_slice(u);
        dst_au.copy_from_slice(au);
        self.cols += 1;
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReorthPolicy {
    Never,
    IfNeeded,
    Always,
}

impl Default for ReorthPolicy {
    fn default() -> Self {
        ReorthPolicy::IfNeeded
    }
}

/// Specification for sizing GMRES/FGMRES workspaces.
#[derive(Debug, Clone, Copy)]
pub struct GmresSpec {
    pub n: usize,
    pub m: usize,
    pub need_z: bool,
    pub block_s: usize,
}

/// Column-major storage reused for block Krylov vectors.
#[derive(Debug, Clone)]
pub struct BlockVec {
    data: Vec<f64>,
    n: usize,
    p: usize,
}

impl BlockVec {
    pub fn new(n: usize, p: usize) -> Self {
        Self {
            data: vec![0.0; n.saturating_mul(p)],
            n,
            p,
        }
    }

    pub fn resize(&mut self, n: usize, p: usize) {
        if self.n != n || self.p != p {
            self.data.resize(n.saturating_mul(p), 0.0);
            self.n = n;
            self.p = p;
        } else {
            let need = n.saturating_mul(p);
            if self.data.len() != need {
                self.data.resize(need, 0.0);
            }
        }
    }

    #[inline]
    pub fn nrows(&self) -> usize {
        self.n
    }

    #[inline]
    pub fn ncols(&self) -> usize {
        self.p
    }

    #[inline]
    pub fn col(&self, j: usize) -> &[f64] {
        let offset = j * self.n;
        &self.data[offset..offset + self.n]
    }

    #[inline]
    pub fn col_mut(&mut self, j: usize) -> &mut [f64] {
        let offset = j * self.n;
        &mut self.data[offset..offset + self.n]
    }

    #[inline]
    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        &mut self.data
    }
}

#[derive(Debug, Clone)]
pub struct GmresSStepWorkspace {
    pub w: BlockVec,
    pub q: BlockVec,
    pub aq: BlockVec,
    pub gram: Vec<f64>,
    pub c_prev: Vec<f64>,
    pub payload: Vec<f64>,
    pub r: Vec<f64>,
}

impl GmresSStepWorkspace {
    pub fn new(n: usize, s: usize, m: usize) -> Self {
        let mut ws = Self {
            w: BlockVec::new(n, s),
            q: BlockVec::new(n, s),
            aq: BlockVec::new(n, s),
            gram: vec![0.0; s.saturating_mul(s)],
            c_prev: vec![0.0; m.saturating_mul(s)],
            payload: vec![0.0; s.saturating_mul(s + 1) / 2 + m.saturating_mul(s)],
            r: vec![0.0; s.saturating_mul(s)],
        };
        ws.ensure(n, s, m);
        ws
    }

    pub fn ensure(&mut self, n: usize, s: usize, m: usize) {
        self.w.resize(n, s);
        self.q.resize(n, s);
        self.aq.resize(n, s);
        ensure_len(&mut self.gram, s.saturating_mul(s));
        ensure_len(&mut self.c_prev, m.saturating_mul(s));
        let payload_len = s.saturating_mul(s + 1) / 2 + m.saturating_mul(s);
        ensure_len(&mut self.payload, payload_len);
        ensure_len(&mut self.r, s.saturating_mul(s));
    }
}

/// Scratch buffers for TSQR factorizations.
#[derive(Debug, Clone)]
pub struct TsqrWorkspace {
    pub taus: Vec<f64>,
    pub rmat: Vec<f64>,
    pub w_max: usize,
}

impl TsqrWorkspace {
    pub fn with_width(w_max: usize) -> Self {
        Self {
            taus: vec![0.0; w_max],
            rmat: vec![0.0; w_max.saturating_mul(w_max)],
            w_max,
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

    /// Ensure communication buffers have enough bytes for upcoming operations.
    pub fn ensure_comm_bytes(&mut self, max_send: usize, max_recv: usize) {
        self.send_arena.ensure_len(max_send);
        self.recv_arena.ensure_len(max_recv);
    }

    /// Ensure the reusable block vector has capacity `n x p`.
    pub fn ensure_block(&mut self, n: usize, p: usize) {
        if p == 0 {
            self.block_buf = None;
            return;
        }
        let replace = match self.block_buf {
            Some(ref buf) if buf.nrows() == n && buf.ncols() >= p => false,
            _ => true,
        };
        if replace {
            self.block_buf = Some(BlockVec::new(n, p));
        }
    }

    /// Ensure the TSQR workspace supports panels up to width `w_max`.
    pub fn ensure_tsqr(&mut self, w_max: usize) {
        if w_max == 0 {
            self.tsqr = None;
            return;
        }
        let replace = match self.tsqr {
            Some(ref tsqr) if tsqr.w_max >= w_max => false,
            _ => true,
        };
        if replace {
            self.tsqr = Some(TsqrWorkspace::with_width(w_max));
        }
    }

    pub fn ensure_sstep(&mut self, n: usize, s: usize, m: usize) {
        if s == 0 {
            self.gmres_sstep = None;
            return;
        }
        let need_new = match self.gmres_sstep {
            Some(ref buf) => {
                buf.w.nrows() != n || buf.w.ncols() < s || buf.c_prev.len() < m.saturating_mul(s)
            }
            None => true,
        };
        if need_new {
            self.gmres_sstep = Some(GmresSStepWorkspace::new(n, s, m));
        } else if let Some(ref mut buf) = self.gmres_sstep {
            buf.ensure(n, s, m);
        }
    }

    #[inline]
    pub fn sstep_mut(&mut self) -> Option<&mut GmresSStepWorkspace> {
        self.gmres_sstep.as_mut()
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
        ensure_len(&mut self.pipelined_w, n);
        ensure_len(&mut self.pipelined_wtmp, n);
        ensure_len(&mut self.pipelined_payload, m + 2);

        if spec.block_s > 0 {
            ensure_len(&mut self.blk_scratch, n * spec.block_s);
        } else {
            self.blk_scratch.clear();
        }

        self.ensure_sstep(n, spec.block_s, m);
    }

    pub fn set_reduction_options(&mut self, opt: crate::utils::reduction::ReductOptions) {
        self.reduction = opt;
    }

    pub fn set_reduction_mode(&mut self, mode: crate::utils::reduction::ReductMode) {
        self.reduction.mode = mode;
    }

    pub fn reduction_options(&self) -> &crate::utils::reduction::ReductOptions {
        &self.reduction
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

    pub fn pipelined_arnoldi_step(
        &mut self,
        k: usize,
        n: usize,
        comm: &crate::parallel::UniverseComm,
        policy: ReorthPolicy,
        tol: f64,
    ) -> Result<usize, crate::error::KError> {
        debug_assert!(k < self.m);

        let w = &self.pipelined_w[..n];
        let payload_len = k + 2;
        let send = {
            let payload = &mut self.pipelined_payload[..payload_len];
            for i in 0..=k {
                let vi = &self.v_mem[i * n..(i + 1) * n];
                payload[i] = vi.iter().zip(w).map(|(a, b)| a * b).sum();
            }
            payload[k + 1] = w.iter().map(|val| val * val).sum();
            payload.to_vec()
        };
        let opt = self.reduction.clone();
        let (handle, _) = <crate::parallel::UniverseComm as crate::utils::reduction::AllreduceOps>::
            allreduce_n_async(comm, send, &opt)?;
        let glob =
            <crate::parallel::UniverseComm as crate::utils::reduction::AllreduceOps>::wait_vec(
                handle,
            );

        let mut reductions = 1usize;

        self.pipelined_wtmp[..n].copy_from_slice(w);

        let mut sum_h2 = 0.0;
        for i in 0..=k {
            let hij = glob[i];
            sum_h2 += hij * hij;
            let vi = &self.v_mem[i * n..(i + 1) * n];
            for idx in 0..n {
                self.pipelined_wtmp[idx] -= hij * vi[idx];
            }
            *self.h_at_mut(i, k) = hij;
        }

        let total_norm_sq = glob[k + 1];
        let mut hnext_sq = (total_norm_sq - sum_h2).max(0.0);
        if !hnext_sq.is_finite() {
            hnext_sq = 0.0;
        }

        let tol = tol.max(0.0);
        let tol_sq = tol * tol;
        let trigger_reorth = match policy {
            ReorthPolicy::Never => false,
            ReorthPolicy::Always => true,
            ReorthPolicy::IfNeeded => total_norm_sq > 0.0 && hnext_sq < tol_sq * total_norm_sq,
        };

        if trigger_reorth {
            reductions += 1;

            let send = {
                let payload = &mut self.pipelined_payload[..payload_len];
                for i in 0..=k {
                    let vi = &self.v_mem[i * n..(i + 1) * n];
                    payload[i] = vi
                        .iter()
                        .zip(&self.pipelined_wtmp[..n])
                        .map(|(a, b)| a * b)
                        .sum();
                }
                payload[k + 1] = self.pipelined_wtmp[..n].iter().map(|val| val * val).sum();
                payload.to_vec()
            };
            let (handle, _) =
                <crate::parallel::UniverseComm as crate::utils::reduction::AllreduceOps>::
                    allreduce_n_async(comm, send, &opt)?;
            let corr =
                <crate::parallel::UniverseComm as crate::utils::reduction::AllreduceOps>::wait_vec(
                    handle,
                );

            let mut delta_norm_sq = 0.0;
            for i in 0..=k {
                let delta = corr[i];
                delta_norm_sq += delta * delta;
                let vi = &self.v_mem[i * n..(i + 1) * n];
                for idx in 0..n {
                    self.pipelined_wtmp[idx] -= delta * vi[idx];
                }
                let hij = *self.h_at_mut(i, k) + delta;
                *self.h_at_mut(i, k) = hij;
            }

            sum_h2 = 0.0;
            for i in 0..=k {
                let hij = *self.h_at_mut(i, k);
                sum_h2 += hij * hij;
            }

            let wtmp_norm_sq = corr[k + 1];
            hnext_sq = (wtmp_norm_sq - delta_norm_sq).max(0.0);
            if !hnext_sq.is_finite() {
                hnext_sq = 0.0;
            }
        }

        let hnext = hnext_sq.sqrt();
        *self.h_at_mut(k + 1, k) = hnext;

        let base = (k + 1) * n;
        if hnext > 0.0 {
            let inv = 1.0 / hnext;
            for idx in 0..n {
                self.v_mem[base + idx] = self.pipelined_wtmp[idx] * inv;
            }
        } else {
            for idx in 0..n {
                self.v_mem[base + idx] = 0.0;
            }
        }

        Ok(reductions)
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
