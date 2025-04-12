pub(super) struct SolverCache {
    pub(super) dont_look_bits: Vec<bool>,
}

impl SolverCache {
    pub(super) fn new(n: usize) -> Self {
        let dont_look_bits = (0..n).map(|_| true).collect();
        Self { dont_look_bits }
    }
}
