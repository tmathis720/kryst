use std::vec::Vec;

#[derive(Clone)]
pub struct DofLayout {
    pub block_size: usize,
    pub n_nodes: usize,
    pub node_of: Vec<usize>,
    pub comp_of: Vec<usize>,
}

impl DofLayout {
    pub fn new(n_dofs: usize, block_size: usize) -> Self {
        assert!(block_size >= 1 && n_dofs % block_size == 0);
        let n_nodes = n_dofs / block_size;
        let mut node_of = vec![0usize; n_dofs];
        let mut comp_of = vec![0usize; n_dofs];
        for i in 0..n_dofs {
            node_of[i] = i / block_size;
            comp_of[i] = i % block_size;
        }
        Self {
            block_size,
            n_nodes,
            node_of,
            comp_of,
        }
    }
}
