use std::ops::Range;

#[derive(Clone, Default)]
pub struct RowRanges {
    pub spans: Vec<Range<usize>>,
}

impl RowRanges {
    pub fn from_mask(mask: &[bool], value: bool) -> Self {
        let mut spans = Vec::new();
        if mask.is_empty() {
            return Self { spans };
        }
        let mut i = 0usize;
        while i < mask.len() {
            if mask[i] == value {
                let start = i;
                i += 1;
                while i < mask.len() && mask[i] == value {
                    i += 1;
                }
                spans.push(start..i);
            } else {
                i += 1;
            }
        }
        Self { spans }
    }

    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }
}
