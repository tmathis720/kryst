use crate::algebra::prelude::*;

/// Build a complex Givens rotation that zeroes the second entry of the vector [a; b].
/// Returns (c, s) where `c` is real and non-negative, and `s` is the complex sine.
#[inline]
pub fn build_complex_givens(a: S, b: S) -> (R, S) {
    let absa = a.abs();
    let absb = b.abs();
    if absb == 0.0 {
        return (1.0, S::zero());
    }
    let r = absa.hypot(absb);
    let c = if r == 0.0 { 0.0 } else { absa / r };
    let s = if absa != 0.0 {
        let scale = S::from_real(c / (absa * absa));
        (a.conj() * b) * scale
    } else {
        b / S::from_real(absb)
    };
    (c, s)
}

/// Apply a complex Givens rotation to the in-place vector [h_ik; h_ip1k].
#[inline]
pub fn apply_complex_givens(h_ik: &mut S, h_ip1k: &mut S, c: R, s: S) {
    let c_s = S::from_real(c);
    let t = c_s * *h_ik + s * *h_ip1k;
    let t2 = -s.conj() * *h_ik + c_s * *h_ip1k;
    *h_ik = t;
    *h_ip1k = t2;
}

/// Apply the previously accumulated Givens rotations to a Hessenberg column.
#[inline]
pub fn apply_prev_givens_to_col(hcol: &mut [S], k: usize, cs: &[R], sn: &[S]) {
    for i in 0..k {
        let (head, tail) = hcol.split_at_mut(i + 1);
        let hij = &mut head[i];
        let hij1 = &mut tail[0];
        apply_complex_givens(hij, hij1, cs[i], sn[i]);
    }
}

/// Apply a freshly built Givens rotation to the Hessenberg column and update g.
#[inline]
pub fn apply_new_givens_and_update_g(
    hcol: &mut [S],
    k: usize,
    cs: &mut [R],
    sn: &mut [S],
    g: &mut [S],
) {
    let (c, s) = build_complex_givens(hcol[k], hcol[k + 1]);
    cs[k] = c;
    sn[k] = s;
    let (head, tail) = hcol.split_at_mut(k + 1);
    let hik = &mut head[k];
    let hip1k = &mut tail[0];
    apply_complex_givens(hik, hip1k, c, s);

    let c_s = S::from_real(c);
    let gk = g[k];
    let gk1 = g[k + 1];
    g[k] = c_s * gk + s * gk1;
    g[k + 1] = -s.conj() * gk + c_s * gk1;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complex_givens_zeroes_second_entry() {
        let a: S = S::from_real(1.2);
        #[cfg(feature = "complex")]
        let b: S = num_complex::Complex64::new(-0.7, 0.5);
        #[cfg(not(feature = "complex"))]
        let b: S = S::from_real(-0.7);

        let (c, s) = build_complex_givens(a, b);
        let mut h0 = a;
        let mut h1 = b;
        apply_complex_givens(&mut h0, &mut h1, c, s);

        let scale = (a.abs() + b.abs()).max(1.0);
        assert!(h1.abs() <= 1e-12 * scale);
        assert!(c >= 0.0);
    }
}
