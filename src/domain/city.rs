/// Represents a city in the Traveling Salesman Problem.
///
/// Cities are identified by a unique index (0-based). This is a lightweight
/// wrapper around `usize` that provides type safety and clarity in the code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct City(pub usize);

impl City {
    /// Returns the unique identifier (index) of this city.
    ///
    /// # Returns
    ///
    /// The city's index as a `usize`
    ///
    /// # Performance
    ///
    /// This is an inline function with zero overhead.
    #[inline]
    pub fn id(&self) -> usize {
        self.0
    }
}
