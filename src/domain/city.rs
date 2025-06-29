#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct City(pub usize);

impl City {
    #[inline]
    pub fn id(&self) -> usize {
        self.0
    }
}
