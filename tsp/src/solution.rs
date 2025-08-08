pub mod list;
use crate::problem::Problem;
use std::ops::Index;

pub trait Solution<P: Problem>: Index<usize> {
    type Distance;

    fn distance(&self, p: &P) -> Self::Distance;

    fn size(&self) -> usize;
}
