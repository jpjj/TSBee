pub mod list;
use crate::problem::Problem;
pub trait Solution<P: Problem> {
    type Distance;

    fn distance(&self, p: &P) -> Self::Distance;

    fn size(&self) -> usize;
}
