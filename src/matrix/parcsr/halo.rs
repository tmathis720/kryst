/// Communication plan for halo exchanges in distributed matrices.
#[derive(Debug, Default, Clone)]
pub struct HaloPlan {
    pub neighbors: Vec<i32>,
    pub send_ptr: Vec<usize>,
    pub send_idx: Vec<u64>,
    pub recv_ptr: Vec<usize>,
    pub recv_idx: Vec<u64>,
}
