use super::super::amg::prolong::classical_pattern;
use crate::matrix::sparse::CsrMatrix;
use crate::preconditioner::amg::strength::Strength;

#[test]
fn direct_pattern_basic() {
    let a = CsrMatrix::from_csr(
        4,
        4,
        vec![0, 2, 5, 8, 10],
        vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3],
        vec![2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0],
    );
    let s = Strength::from_csr(&a, 0.0, true);
    let s_sym = s.symmetrize();
    let is_c = vec![true, false, true, false];
    let (p, cf) = classical_pattern(&a, &s_sym, &is_c, false);
    assert_eq!(cf.coarse_of, vec![Some(0), None, Some(1), None]);
    let r1s = p.row_ptr[1];
    let r1e = p.row_ptr[2];
    assert_eq!(&p.col_idx[r1s..r1e], &[0, 1]);
    let r3s = p.row_ptr[3];
    let r3e = p.row_ptr[4];
    assert_eq!(&p.col_idx[r3s..r3e], &[1]);
}
