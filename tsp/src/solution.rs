pub mod list;
use crate::{edge::Edge, problem::Problem};
use std::{collections::HashSet, ops::Index};

pub trait Solution<P: Problem>: Index<usize> {
    type Distance;

    fn distance(&self, p: &P) -> Self::Distance;

    fn size(&self) -> usize;

    fn get_edges(&self) -> HashSet<Edge>;
}
