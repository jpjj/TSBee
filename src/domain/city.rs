#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct City(pub(crate) usize);

impl City {
    #[inline]
    pub(crate) fn id(&self) -> usize {
        self.0
    }
}
