pub(super) struct SolverCache {
    pub(super) dont_look_bits: Vec<bool>,
}

impl SolverCache {
    pub(super) fn new(n: usize) -> Self {
        let dont_look_bits = (0..n).map(|_| true).collect();
        Self { dont_look_bits }
    }

    pub(super) fn reset(&mut self) {
        let n = self.dont_look_bits.len();
        self.dont_look_bits = (0..n).map(|_| true).collect();
    }
}
