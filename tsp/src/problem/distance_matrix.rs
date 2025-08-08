use crate::{city::City, problem::Problem};

pub struct DistanceMatrix<T> {
    n: usize,
    flat_dm: Vec<T>,
}

impl<T> DistanceMatrix<T> {
    pub fn from_flat(vector: Vec<T>) -> Self {
        let vector_len = vector.len();
        let n = vector_len.isqrt();
        assert_eq!(n * n, vector_len);
        Self { n, flat_dm: vector }
    }

    fn get_index(&self, from: usize, to: usize) -> usize {
        from * self.n + to
    }
}

impl<T> Problem for DistanceMatrix<T>
where
    T: Copy,
{
    type Distance = T;

    fn distance(&self, c1: City, c2: City) -> Self::Distance {
        let index = self.get_index(c1.0, c2.0);
        self.flat_dm[index]
    }

    fn size(&self) -> usize {
        self.n
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        city::City,
        problem::{Problem, distance_matrix::DistanceMatrix},
    };

    #[test]
    fn test_distance_matrix() {
        let flat_vector: Vec<i32> = (0..16).collect();
        let dm: DistanceMatrix<i32> = DistanceMatrix::from_flat(flat_vector);
        assert_eq!(dm.distance(City(0), City(1)), 1);
        assert_eq!(dm.distance(City(3), City(3)), 15);
    }

    #[test]
    #[should_panic]
    fn test_distance_matrix_should_panic() {
        let flat_vector: Vec<i32> = (0..15).collect();
        let _ = DistanceMatrix::from_flat(flat_vector);
    }
}
