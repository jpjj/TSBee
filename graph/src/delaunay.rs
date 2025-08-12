use crate::AdjacencyList;
use spade::{DelaunayTriangulation, Triangulation};
use tsp::{
    city::City,
    problem::{TspProblem, points_and_function::Point},
};

impl<'a> AdjacencyList<'a> {
    pub fn from_delaunay(problem: &'a TspProblem) -> Result<Self, Box<dyn std::error::Error>> {
        let points = match problem {
            TspProblem::Euclidean(p) => p.points(),
            TspProblem::Att(p) => p.points(),
            TspProblem::Ceil(p) => p.points(),
            TspProblem::Geo(p) => p.points(),
            TspProblem::DistanceMatrix(_) => {
                return Err("Expected node coordinates for Delaunay triangulation".into());
            }
        };

        build_delaunay_adjacency_list(points, problem)
    }
}

pub fn build_delaunay_adjacency_list<'a>(
    points: &[Point<f64>],
    problem: &'a TspProblem,
) -> Result<AdjacencyList<'a>, Box<dyn std::error::Error>> {
    let new_points: Vec<spade::Point2<f64>> = points
        .iter()
        .map(|p| spade::Point2::new(p.0, p.1))
        .collect();
    let triangulation = DelaunayTriangulation::<spade::Point2<f64>>::bulk_load_stable(new_points)
        .expect("Insertion Error During Delaunay Triangulation");

    let n = points.len();
    let mut list = vec![Vec::new(); n];

    for edge in triangulation.undirected_edges() {
        let from = edge.vertices()[0].index();
        let to = edge.vertices()[1].index();
        list[from].push(City(to));
        list[to].push(City(from));
    }

    for neighbors in &mut list {
        neighbors.sort_by_key(|c| c.0);
        neighbors.dedup();
    }

    Ok(AdjacencyList::new(problem, list))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tsp::problem::points_and_function::PointsAndFunction;
    use tsp::problem::points_and_function::euc_2d::Euc2d;

    #[test]
    fn test_delaunay_triangulation_5_points() {
        // Test points forming a square with a center point:
        //
        //  1 ---- 3
        //  |  \ / |
        //  |   4  |
        //  | /  \ |
        //  0 ---- 2
        //
        // Point 0: (0,0), Point 1: (0,2), Point 2: (2,0), Point 3: (2,2), Point 4: (1,1)

        let points = vec![
            Point(0.0, 0.0), // Point 0: bottom-left
            Point(0.0, 2.0), // Point 1: top-left
            Point(2.0, 0.0), // Point 2: bottom-right
            Point(2.0, 2.0), // Point 3: top-right
            Point(1.0, 1.0), // Point 4: center
        ];

        let euclidean_problem = PointsAndFunction::<f64, f64, Euc2d>::new(points);
        let tsp_problem = TspProblem::Euclidean(euclidean_problem);

        let adj_list = AdjacencyList::from_delaunay(&tsp_problem)
            .expect("Failed to create Delaunay triangulation");

        let expected_edges = [
            vec![City(1), City(2), City(4)],
            vec![City(0), City(3), City(4)],
            vec![City(0), City(3), City(4)],
            vec![City(1), City(2), City(4)],
            vec![City(0), City(1), City(2), City(3)],
        ];

        for (city_idx, expected_neighbors) in expected_edges.iter().enumerate() {
            let actual_neighbors: Vec<City> = adj_list.list[city_idx].clone();
            assert_eq!(
                &actual_neighbors,
                expected_neighbors,
                "{:?}",
                format!(
                    "City {city_idx} has incorrect neighbors. Expected {:?}, got {:?}",
                    expected_neighbors, actual_neighbors
                )
            );
        }
    }
}
