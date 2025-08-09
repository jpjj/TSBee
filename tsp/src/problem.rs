pub mod distance_matrix;
pub mod points_and_function;
use crate::{
    city::City,
    problem::{
        distance_matrix::DistanceMatrix,
        points_and_function::{
            PointsAndFunction, att::ATT, ceil_2d::Ceil2d, euc_2d::Euc2d, geo::Geo,
        },
    },
};

pub trait Problem {
    type Distance;

    fn distance(&self, c1: City, c2: City) -> Self::Distance;

    fn size(&self) -> usize;
}

pub enum TspProblem {
    Euclidean(PointsAndFunction<f64, i64, Euc2d>),
    Att(PointsAndFunction<f64, i64, ATT>),
    Ceil(PointsAndFunction<f64, i64, Ceil2d>),
    Geo(PointsAndFunction<f64, i64, Geo>),
    DistanceMatrix(DistanceMatrix<i64>),
}

impl Problem for TspProblem {
    type Distance = i64;

    fn size(&self) -> usize {
        match self {
            Self::Euclidean(p) => p.size(),
            Self::Att(p) => p.size(),
            Self::Ceil(p) => p.size(),
            Self::Geo(p) => p.size(),
            Self::DistanceMatrix(p) => p.size(),
        }
    }

    fn distance(&self, i: City, j: City) -> Self::Distance {
        match self {
            Self::Euclidean(p) => p.distance(i, j),
            Self::Ceil(p) => p.distance(i, j),
            Self::Att(p) => p.distance(i, j),
            Self::Geo(p) => p.distance(i, j),
            Self::DistanceMatrix(p) => p.distance(i, j),
        }
    }
}
