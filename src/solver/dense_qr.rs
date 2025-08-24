use crate::error::KError;
use faer::linalg::solvers::{Qr, SolveCore};
use faer::{Conj, Mat, MatMut};

/// Solve a dense system using QR factorization.
pub fn solve(a: &Mat<f64>, b: &[f64], x: &mut [f64]) -> Result<(), KError> {
    if b.len() != a.nrows() || x.len() != a.ncols() {
        return Err(KError::InvalidInput(
            "dimension mismatch in dense_qr::solve".into(),
        ));
    }
    let qr = Qr::new(a.as_ref());
    x.clone_from_slice(b);
    let m = b.len();
    let x_mat = MatMut::from_column_major_slice_mut(x, m, 1);
    qr.solve_in_place_with_conj(Conj::No, x_mat);
    Ok(())
}
