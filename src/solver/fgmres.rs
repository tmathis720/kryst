#![allow(clippy::needless_range_loop, unused_assignments, dead_code, clippy::type_complexity, clippy::too_many_arguments)]
//! Flexible GMRES (Saad §9.4)

use crate::preconditioner::FlexiblePreconditioner;
use crate::utils::convergence::{Convergence, SolveStats};
use crate::error::KError;
use crate::core::traits::{MatVec, InnerProduct};

pub enum Orthog { Classical, Modified }

#[allow(clippy::type_complexity)]
pub struct FgmresSolver<T> {
    pub conv: Convergence<T>,
    pub restart: usize,
    pub delta_allocate: usize,
    pub preallocate: bool,
    pub orthog: Orthog,
    pub haptol: T,
    pub modify_pc: Option<Box<dyn FnMut(usize, usize, T) -> Result<(), KError>>>,
    pub monitor: Option<Box<dyn FnMut(usize, T)>>,
    pub residual_history: Vec<T>,
}

impl<T: num_traits::Float> FgmresSolver<T> {
    pub fn new(tol: T, max_iters: usize, restart: usize) -> Self {
        Self {
            conv: Convergence { tol, max_iters },
            restart,
            delta_allocate: 10,
            preallocate: false,
            orthog: Orthog::Classical,
            haptol: T::from(1e-12).unwrap(),
            modify_pc: None,
            monitor: None,
            residual_history: Vec::new(),
        }
    }
    pub fn with_orthog(mut self, orthog: Orthog) -> Self {
        self.orthog = orthog;
        self
    }
    pub fn with_preallocate(mut self, preallocate: bool) -> Self {
        self.preallocate = preallocate;
        self
    }
    pub fn with_delta_allocate(mut self, delta: usize) -> Self {
        self.delta_allocate = delta;
        self
    }
    pub fn with_haptol(mut self, haptol: T) -> Self {
        self.haptol = haptol;
        self
    }
    pub fn with_modify_pc<F>(mut self, f: F) -> Self
    where F: FnMut(usize, usize, T) -> Result<(), KError> + 'static {
        self.modify_pc = Some(Box::new(f));
        self
    }
    pub fn with_monitor<F>(mut self, f: F) -> Self
    where F: FnMut(usize, T) + 'static {
        self.monitor = Some(Box::new(f));
        self
    }
    pub fn clear_history(&mut self) {
        self.residual_history.clear();
    }

    /// Flexible GMRES solve (Saad §9.4)
    pub fn solve_flex<M, V>(
        &mut self,
        a: &M,
        pc: Option<&mut dyn FlexiblePreconditioner<M, V>>,
        b: &V,
        x: &mut V,
    ) -> Result<SolveStats<T>, KError>
    where
        M: MatVec<V>,
        (): InnerProduct<V, Scalar = T>,
        V: From<Vec<T>> + AsRef<[T]> + AsMut<[T]> + Clone,
    {
        let n = b.as_ref().len();
        let ip = ();
        let restart = self.restart;
        let max_iters = self.conv.max_iters;
        let tol = self.conv.tol;

        // x0 is input x
        let mut r = b.clone();
        let mut tmp = V::from(vec![T::zero(); n]);
        a.matvec(x, &mut tmp);
        for (ri, ai) in r.as_mut().iter_mut().zip(tmp.as_ref()) {
            *ri = *ri - *ai;
        }
        let mut beta = ip.norm(&r);
        if beta == T::zero() {
            return Ok(SolveStats { iterations: 0, final_residual: T::zero(), converged: true });
        }
        // Allocate basis and Hessenberg storage
        let (mut v_basis, mut z_basis, mut h, mut cs, mut sn, mut s) = if self.preallocate {
            (
                vec![V::from(vec![T::zero(); n]); max_iters + 1],
                vec![V::from(vec![T::zero(); n]); max_iters],
                vec![vec![T::zero(); max_iters]; max_iters + 1],
                vec![T::zero(); max_iters],
                vec![T::zero(); max_iters],
                vec![T::zero(); max_iters + 1],
            )
        } else {
            (
                vec![V::from(vec![T::zero(); n]); restart + 1],
                vec![V::from(vec![T::zero(); n]); restart],
                vec![vec![T::zero(); restart]; restart + 1],
                vec![T::zero(); restart],
                vec![T::zero(); restart],
                vec![T::zero(); restart + 1],
            )
        };
        s[0] = beta;
        for (vi, ri) in v_basis[0].as_mut().iter_mut().zip(r.as_ref()) {
            *vi = *ri / beta;
        }
        let mut total_iters = 0;
        let res_norm = beta;
        let mut stats = SolveStats { iterations: 0, final_residual: res_norm, converged: false };
        let mut pc_mut = pc;
        'outer: while total_iters < max_iters {
            let m = if self.preallocate { max_iters.min(restart) } else { restart.min(max_iters - total_iters) };
            // If not enough storage, grow by delta_allocate
            if !self.preallocate && v_basis.len() < m + 1 {
                let grow = self.delta_allocate.max(m + 1 - v_basis.len());
                v_basis.extend((0..grow).map(|_| V::from(vec![T::zero(); n])));
                z_basis.extend((0..grow).map(|_| V::from(vec![T::zero(); n])));
                h.extend((0..grow).map(|_| vec![T::zero(); restart]));
                for row in &mut h {
                    if row.len() < restart { row.extend(vec![T::zero(); restart - row.len()]); }
                }
                cs.extend(vec![T::zero(); grow]);
                sn.extend(vec![T::zero(); grow]);
                s.extend(vec![T::zero(); grow]);
            }
            // If not enough storage, grow by delta_allocate
            if !self.preallocate && z_basis.len() < m {
                let grow = self.delta_allocate.max(m - z_basis.len());
                z_basis.extend((0..grow).map(|_| V::from(vec![T::zero(); n])));
            }
            // If not enough storage, grow by delta_allocate
            if !self.preallocate && h.len() < m + 1 {
                let grow = self.delta_allocate.max(m + 1 - h.len());
                h.extend((0..grow).map(|_| vec![T::zero(); restart]));
                for row in &mut h {
                    if row.len() < restart { row.extend(vec![T::zero(); restart - row.len()]); }
                }
            }
            // Perform one FGMRES cycle: Arnoldi, Givens, and monitoring.
            let m = if self.preallocate { max_iters.min(restart) } else { restart.min(max_iters - total_iters) };
            let mut res_norm = s[0].abs();
            let mut converged = false;
            let mut arnoldi_steps = m;
            for j in 0..m {
                // (a) Precondition: z_basis[j] = M.apply(v_basis[j])
                z_basis[j] = v_basis[j].clone();
                if let Some(ref mut pc) = pc_mut {
                    pc.apply(&v_basis[j], &mut z_basis[j])?;
                }
                // (b) w = A * z_basis[j]
                let mut w = V::from(vec![T::zero(); n]);
                a.matvec(&z_basis[j], &mut w);
                // (c) Arnoldi orthonormalization (two-phase to avoid borrow conflicts)
                let mut h_col = vec![T::zero(); j+2];
                match self.orthog {
                    Orthog::Classical => {
                        for i in 0..=j {
                            h_col[i] = ip.dot(&w, &v_basis[i]);
                        }
                        for i in 0..=j {
                            for (wi, vi) in w.as_mut().iter_mut().zip(v_basis[i].as_ref()) {
                                *wi = *wi - h_col[i] * *vi;
                            }
                        }
                    }
                    Orthog::Modified => {
                        for i in 0..=j {
                            h_col[i] = ip.dot(&w, &v_basis[i]);
                        }
                        for i in 0..=j {
                            for (wi, vi) in w.as_mut().iter_mut().zip(v_basis[i].as_ref()) {
                                *wi = *wi - h_col[i] * *vi;
                            }
                        }
                        // Iterative refinement: re-orthogonalize if needed
                        for i in 0..=j {
                            let corr = ip.dot(&w, &v_basis[i]);
                            if corr.abs() > T::from(1e-10).unwrap() {
                                for (wi, vi) in w.as_mut().iter_mut().zip(v_basis[i].as_ref()) {
                                    *wi = *wi - corr * *vi;
                                }
                            }
                        }
                    }
                }
                h[j+1][j] = ip.norm(&w);
                for i in 0..=j { h[i][j] = h_col[i]; }
                // Happy breakdown logic
                let hapbnd = self.haptol * s[j].abs();
                let happy_breakdown = h[j+1][j].abs() < hapbnd;
                if !happy_breakdown {
                    let w_norm = h[j+1][j];
                    let w_vec: Vec<T> = w.as_ref().iter().map(|&wi| wi / w_norm).collect();
                    v_basis[j+1] = V::from(w_vec);
                } else {
                    for vj in v_basis[j+1].as_mut() { *vj = T::zero(); }
                }
                // (d) Apply previous Givens rotations
                for i in 0..j {
                    let temp = cs[i] * h[i][j] + sn[i] * h[i+1][j];
                    h[i+1][j] = -sn[i] * h[i][j] + cs[i] * h[i+1][j];
                    h[i][j] = temp;
                }
                // (e) Compute new rotation
                let (c, s_) = {
                    let h1 = h[j][j];
                    let h2 = h[j+1][j];
                    let denom = (h1*h1 + h2*h2).sqrt();
                    if denom == T::zero() {
                        (T::one(), T::zero())
                    } else {
                        (h1/denom, h2/denom)
                    }
                };
                cs[j] = c;
                sn[j] = s_;
                let temp = c * s[j] + s_ * s[j+1];
                s[j+1] = -s_ * s[j] + c * s[j+1];
                s[j] = temp;
                h[j][j] = c * h[j][j] + s_ * h[j+1][j];
                h[j+1][j] = T::zero();
                res_norm = s[j+1].abs();
                total_iters += 1;
                // Per-iteration monitor and history
                if let Some(ref mut monitor) = self.monitor {
                    monitor(total_iters, res_norm);
                }
                self.residual_history.push(res_norm);
                let (stop, s_stats) = self.conv.check(res_norm, s[0], total_iters);
                stats = s_stats;
                if stop {
                    stats.final_residual = res_norm;
                    stats.iterations = total_iters;
                    arnoldi_steps = j + 1; // Only j+1 Arnoldi steps performed
                    converged = true;
                    break 'outer;
                }
            }
            // (4) Back-substitute and update x
            let k = arnoldi_steps;
            // Solve upper-triangular system (H[0..k][0..k], s[0..k])
            let mut y = vec![T::zero(); k];
            for i in (0..k).rev() {
                let mut sum = s[i];
                #[allow(clippy::needless_range_loop)]
                for l in (i+1)..k {
                    sum = sum - h[i][l] * y[l];
                }
                y[i] = sum / h[i][i];
            }
            self.build_solution(x, &y, &z_basis);
            // Compute new residual
            let mut r_new = b.clone();
            let mut tmp = V::from(vec![T::zero(); n]);
            a.matvec(x, &mut tmp);
            for (ri, ai) in r_new.as_mut().iter_mut().zip(tmp.as_ref()) {
                *ri = *ri - *ai;
            }
            let res_norm = ip.norm(&r_new);
            if res_norm < tol || converged {
                stats.final_residual = res_norm;
                stats.iterations = total_iters;
                stats.converged = true;
                break;
            }
            // Restart: set v_basis[0] = r_new / ||r_new||
            beta = res_norm;
            for (vi, ri) in v_basis[0].as_mut().iter_mut().zip(r_new.as_ref()) {
                *vi = *ri / beta;
            }
            s.clear();
            s.resize(restart + 1, T::zero());
            s[0] = beta;
        }
        stats.final_residual = res_norm;
        stats.iterations = total_iters;
        Ok(stats)
    }

    /// Build the solution x = x0 + sum y[i] * z_basis[i] from Arnoldi/QR results
    fn build_solution<V: AsRef<[T]> + AsMut<[T]> + Clone + From<Vec<T>>>(
        &self,
        x: &mut V,
        y: &[T],
        z_basis: &[V],
    ) {
        for (i, yi) in y.iter().enumerate() {
            for (xi, zi) in x.as_mut().iter_mut().zip(z_basis[i].as_ref()) {
                *xi = *xi + *yi * *zi;
            }
        }
    }

    fn cycle<M, V>(
        &mut self,
        a: &M,
        mut pc: Option<&mut dyn FlexiblePreconditioner<M, V>>,
        b: &V,
        #[allow(unused_variables)] _x: &mut V,
        v_basis: &mut [V],
        z_basis: &mut [V],
        h: &mut [Vec<T>],
        cs: &mut [T],
        sn: &mut [T],
        s: &mut [T],
        total_iters: &mut usize,
        stats: &mut SolveStats<T>,
    ) -> Result<(usize, bool, T), KError>
    where
        M: MatVec<V>,
        (): InnerProduct<V, Scalar = T>,
        V: From<Vec<T>> + AsRef<[T]> + AsMut<[T]> + Clone,
    {
        #[allow(dead_code, clippy::type_complexity, clippy::too_many_arguments)]
        let n = b.as_ref().len();
        let ip = ();
        let restart = self.restart;
        let max_iters = self.conv.max_iters;
        let _tol = self.conv.tol;

        let m = if self.preallocate { max_iters.min(restart) } else { restart.min(max_iters - *total_iters) };
        #[allow(unused_assignments)]
        let mut res_norm = s[0].abs();
        #[allow(unused_assignments)]
        let mut arnoldi_steps = m;
        #[allow(unused_assignments)]
        let mut converged = false;
        for j in 0..m {
            // (a) Precondition: z_basis[j] = M.apply(v_basis[j])
            z_basis[j] = v_basis[j].clone();
            if let Some(ref mut pc) = pc {
                pc.apply(&v_basis[j], &mut z_basis[j])?;
            }
            // (b) w = A * z_basis[j]
            let mut w = V::from(vec![T::zero(); n]);
            a.matvec(&z_basis[j], &mut w);
            // (c) Arnoldi orthonormalization (two-phase to avoid borrow conflicts)
            let mut h_col = vec![T::zero(); j+2];
            match self.orthog {
                Orthog::Classical => {
                    for i in 0..=j {
                        h_col[i] = ip.dot(&w, &v_basis[i]);
                    }
                    for i in 0..=j {
                        for (wi, vi) in w.as_mut().iter_mut().zip(v_basis[i].as_ref()) {
                            *wi = *wi - h_col[i] * *vi;
                        }
                    }
                }
                Orthog::Modified => {
                    for i in 0..=j {
                        h_col[i] = ip.dot(&w, &v_basis[i]);
                    }
                    for i in 0..=j {
                        for (wi, vi) in w.as_mut().iter_mut().zip(v_basis[i].as_ref()) {
                            *wi = *wi - h_col[i] * *vi;
                        }
                    }
                    // Iterative refinement: re-orthogonalize if needed
                    for i in 0..=j {
                        let corr = ip.dot(&w, &v_basis[i]);
                        if corr.abs() > T::from(1e-10).unwrap() {
                            for (wi, vi) in w.as_mut().iter_mut().zip(v_basis[i].as_ref()) {
                                *wi = *wi - corr * *vi;
                            }
                        }
                    }
                }
            }
            h[j+1][j] = ip.norm(&w);
            for i in 0..=j { h[i][j] = h_col[i]; }
            // Happy breakdown logic
            let hapbnd = self.haptol * s[j].abs();
            let happy_breakdown = h[j+1][j].abs() < hapbnd;
            if !happy_breakdown {
                let w_norm = h[j+1][j];
                let w_vec: Vec<T> = w.as_ref().iter().map(|&wi| wi / w_norm).collect();
                v_basis[j+1] = V::from(w_vec);
            } else {
                for vj in v_basis[j+1].as_mut() { *vj = T::zero(); }
            }
            // (d) Apply previous Givens rotations
            for i in 0..j {
                let temp = cs[i] * h[i][j] + sn[i] * h[i+1][j];
                h[i+1][j] = -sn[i] * h[i][j] + cs[i] * h[i+1][j];
                h[i][j] = temp;
            }
            // (e) Compute new rotation
            let (c, s_) = {
                let h1 = h[j][j];
                let h2 = h[j+1][j];
                let denom = (h1*h1 + h2*h2).sqrt();
                if denom == T::zero() {
                    (T::one(), T::zero())
                } else {
                    (h1/denom, h2/denom)
                }
            };
            cs[j] = c;
            sn[j] = s_;
            let temp = c * s[j] + s_ * s[j+1];
            s[j+1] = -s_ * s[j] + c * s[j+1];
            s[j] = temp;
            h[j][j] = c * h[j][j] + s_ * h[j+1][j];
            h[j+1][j] = T::zero();
            res_norm = s[j+1].abs();
            *total_iters += 1;
            let (stop, s_stats) = self.conv.check(res_norm, s[0], *total_iters);
            *stats = s_stats;
            if stop {
                stats.final_residual = res_norm;
                stats.iterations = *total_iters;
                arnoldi_steps = j + 1; // Only j+1 Arnoldi steps performed
                converged = true;
                break;
            }
        }
        Ok((arnoldi_steps, converged, res_norm))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::traits::MatVec;
    use crate::preconditioner::{Preconditioner, FlexiblePreconditioner};
    use crate::error::KError;

    // Simple 2x2 system: [2 1; 1 3]
    #[derive(Clone)]
    struct Simple2;
    impl MatVec<Vec<f64>> for Simple2 {
        fn matvec(&self, x: &Vec<f64>, y: &mut Vec<f64>) {
            y[0] = 2.0 * x[0] + 1.0 * x[1];
            y[1] = 1.0 * x[0] + 3.0 * x[1];
        }
    }

    // Fixed Jacobi preconditioner
    struct Jacobi {
        inv_diag: Vec<f64>,
    }
    impl Jacobi {
        fn new() -> Self {
            Self { inv_diag: vec![0.5, 1.0/3.0] }
        }
    }
    impl Preconditioner<Simple2, Vec<f64>> for Jacobi {
        fn apply(&self, r: &Vec<f64>, z: &mut Vec<f64>) -> Result<(), KError> {
            for (zi, (ri, &d)) in z.iter_mut().zip(r.iter().zip(self.inv_diag.iter())) {
                *zi = *ri * d;
            }
            Ok(())
        }
    }
    // Flexible wrapper for fixed preconditioner
    struct FlexJacobi<'a> {
        inner: &'a Jacobi,
    }
    impl<'a> FlexiblePreconditioner<Simple2, Vec<f64>> for FlexJacobi<'a> {
        fn apply(&mut self, r: &Vec<f64>, z: &mut Vec<f64>) -> Result<(), KError> {
            self.inner.apply(r, z)
        }
    }

    #[test]
    fn fgmres_equiv_to_gmres_on_fixed_pc() {
        let a = Simple2;
        let jacobi = Jacobi::new();
        let mut flex_jacobi = FlexJacobi { inner: &jacobi };
        let x_true = vec![1.0, 2.0];
        let b = {
            let mut v = vec![0.0; 2];
            a.matvec(&x_true, &mut v);
            v
        };
        let mut x = vec![0.0; 2];
        let mut solver = FgmresSolver::new(1e-10, 100, 25);
        let stats = solver.solve_flex(&a, Some(&mut flex_jacobi), &b, &mut x).unwrap();
        let tol = 1e-6;
        for (xi, xt) in x.iter().zip(x_true.iter()) {
            assert!((xi - xt).abs() < tol, "xi={:.6}, expected {:.6}", xi, xt);
        }
        assert!(stats.converged, "FGMRES did not converge");
    }
}
