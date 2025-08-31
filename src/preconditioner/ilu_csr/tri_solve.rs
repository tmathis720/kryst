use crate::error::KError;

use super::IluCsr;

pub fn tri_solve_serial(pc: &IluCsr, x: &[f64], y: &mut [f64]) -> Result<(), KError> {
    let n = pc.n();
    let lr = pc.l_row();
    let lc = pc.l_col();
    let lv = pc.l_val();
    let ur = pc.u_row();
    let uc = pc.u_col();
    let uv = pc.u_val();
    let di = pc.u_diag_ix();

    if y.len() != n || x.len() != n {
        return Err(KError::InvalidInput("tri_solve: dimension mismatch".into()));
    }

    // forward: L y = x  (unit diagonal)
    // Reuse y as the solution vector
    for i in 0..n {
        let mut s = x[i];
        for p in lr[i]..lr[i + 1] {
            let j = lc[p]; // j < i
            s -= lv[p] * y[j];
        }
        y[i] = s;
    }

    // backward: U z = y -> y <- z (output)
    for i in (0..n).rev() {
        let mut s = y[i];
        for p in ur[i]..ur[i + 1] {
            let j = uc[p];
            if j > i {
                s -= uv[p] * y[j];
            }
        }
        let d = uv[di[i]];
        y[i] = s / d;
    }
    Ok(())
}

pub fn tri_solve_level_scheduled(pc: &IluCsr, x: &[f64], y: &mut [f64]) -> Result<(), KError> {
    // Fallback to serial if no levels computed
    if pc.buckets_fwd().is_empty() || pc.buckets_bwd().is_empty() {
        return tri_solve_serial(pc, x, y);
    }

    let n = pc.n();
    if x.len() != n || y.len() != n {
        return Err(KError::InvalidInput("tri_solve: dimension mismatch".into()));
    }

    let lr = pc.l_row();
    let lc = pc.l_col();
    let lv = pc.l_val();
    let ur = pc.u_row();
    let uc = pc.u_col();
    let uv = pc.u_val();
    let di = pc.u_diag_ix();

    // Forward: L y = x (unit diagonal). Per-level parallel, disjoint writes.
    y.fill(0.0);
    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        for bucket in pc.buckets_fwd() {
            // Each worker receives its own copy of y pointer as usize; closure captures only &slices (Sync).
            let y_addr = y.as_mut_ptr() as usize;
            bucket.par_iter().for_each_init(
                move || y_addr,
                |y_addr, &i| unsafe {
                    let y_ptr = *y_addr as *mut f64;
                    let mut s = *x.get_unchecked(i);
                    let rs = *lr.get_unchecked(i);
                    let re = *lr.get_unchecked(i + 1);
                    let lc_p = lc.as_ptr();
                    let lv_p = lv.as_ptr();
                    for p in rs..re {
                        let j = *lc_p.add(p);
                        s -= *lv_p.add(p) * *y_ptr.add(j);
                    }
                    *y_ptr.add(i) = s;
                },
            );
        }
    }
    #[cfg(not(feature = "rayon"))]
    {
        for bucket in pc.buckets_fwd() {
            for &i in bucket {
                let mut s = x[i];
                let rs = lr[i];
                let re = lr[i + 1];
                for p in rs..re {
                    let j = lc[p];
                    s -= lv[p] * y[j];
                }
                y[i] = s;
            }
        }
    }

    // Backward: U z = y → write z into y. Per-level parallel, disjoint writes.
    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        for bucket in pc.buckets_bwd() {
            let y_addr = y.as_mut_ptr() as usize;
            bucket.par_iter().for_each_init(
                move || y_addr,
                |y_addr, &i| unsafe {
                    let y_ptr = *y_addr as *mut f64;
                    let mut s = *y_ptr.add(i);
                    let rs = *ur.get_unchecked(i);
                    let re = *ur.get_unchecked(i + 1);
                    let uc_p = uc.as_ptr();
                    let uv_p = uv.as_ptr();
                    for p in rs..re {
                        let j = *uc_p.add(p);
                        if j > i {
                            s -= *uv_p.add(p) * *y_ptr.add(j);
                        }
                    }
                    let d = *uv_p.add(*di.get_unchecked(i));
                    *y_ptr.add(i) = s / d;
                },
            );
        }
    }
    #[cfg(not(feature = "rayon"))]
    {
        for bucket in pc.buckets_bwd() {
            for &i in bucket {
                let mut s = y[i];
                let rs = ur[i];
                let re = ur[i + 1];
                for p in rs..re {
                    let j = uc[p];
                    if j > i {
                        s -= uv[p] * y[j];
                    }
                }
                let d = uv[di[i]];
                y[i] = s / d;
            }
        }
    }

    Ok(())
}
