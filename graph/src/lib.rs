mod delaunay;

use tsp::{
    city::City,
    edge::Edge,
    problem::{Problem, TspProblem},
};

pub struct AdjacencyMatrix<'a> {
    pub problem: &'a TspProblem,
}

impl<'a> AdjacencyMatrix<'a> {
    pub fn new(problem: &'a TspProblem) -> Self {
        Self { problem }
    }
}

pub struct AdjacencyList<'a> {
    pub problem: &'a TspProblem,
    pub list: Vec<Vec<City>>,
}

impl<'a> AdjacencyList<'a> {
    pub fn new(problem: &'a TspProblem, list: Vec<Vec<City>>) -> Self {
        Self { problem, list }
    }
    pub fn from_edges(problem: &'a TspProblem, edges: Vec<Edge>) -> Self {
        let mut list = vec![vec![]; problem.size()];
        for edge in edges {
            let (u, v) = (edge.u, edge.v);
            list[u.0].push(v);
            list[v.0].push(u);
        }
        Self { problem, list }
    }
}

pub enum Graph<'a> {
    Matrix(AdjacencyMatrix<'a>),
    List(AdjacencyList<'a>),
}

impl<'a> Graph<'a> {
    pub fn problem(&self) -> &'a TspProblem {
        match self {
            Self::Matrix(am) => am.problem,
            Self::List(al) => al.problem,
        }
    }

    pub fn weight(&self, c1: City, c2: City) -> i64 {
        match self {
            Self::Matrix(am) => am.problem.distance(c1, c2),
            Self::List(al) => al.problem.distance(c1, c2),
        }
    }

    pub fn edge_weight(&self, edge: Edge) -> i64 {
        self.weight(edge.u, edge.v)
    }

    pub fn n(&self) -> usize {
        match self {
            Self::Matrix(am) => am.problem.size(),
            Self::List(al) => al.problem.size(),
        }
    }

    pub fn m(&self) -> usize {
        match self {
            Self::Matrix(am) => {
                let n = am.problem.size();
                n * (n - 1) / 2
            }
            Self::List(al) => al.list.iter().map(|x| x.len()).sum::<usize>() / 2,
        }
    }

    pub fn neighbors(&self, c: City) -> impl Iterator<Item = City> {
        let n = self.n();
        let neighbors: Vec<City> = match self {
            Self::Matrix(_) => (0..n).filter(|&i| i != c.0).map(City).collect(),
            Self::List(al) => al.list[c.0].to_vec(),
        };
        neighbors.into_iter()
    }

    pub fn edges(&self) -> impl Iterator<Item = Edge> {
        let edges: Vec<Edge> = match self {
            Self::Matrix(am) => {
                let n = am.problem.size();
                (0..n)
                    .flat_map(|i| (i + 1..n).map(move |j| Edge::new(City(i), City(j))))
                    .collect()
            }
            Self::List(al) => al
                .list
                .iter()
                .enumerate()
                .flat_map(|(i, neighbors)| {
                    neighbors.iter().filter_map(move |&neighbor| {
                        if neighbor.0 > i {
                            Some(Edge::new(City(i), neighbor))
                        } else {
                            None
                        }
                    })
                })
                .collect(),
        };
        edges.into_iter()
    }
    pub fn cities(&self) -> impl Iterator<Item = City> {
        (0..self.n()).map(City)
    }

    pub fn complete_weight(&self) -> i64 {
        self.edges().map(|e| self.weight(e.u, e.v)).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tsp::problem::{
        distance_matrix::DistanceMatrix,
        points_and_function::{Point, PointsAndFunction, euc_2d::Euc2d},
    };

    fn create_test_problem() -> TspProblem {
        let points = vec![
            Point(0.0, 0.0),
            Point(1.0, 0.0),
            Point(1.0, 1.0),
            Point(0.0, 1.0),
        ];

        let problem = PointsAndFunction::<f64, i64, Euc2d>::new(points);
        TspProblem::Euclidean(problem)
    }

    fn create_test_distance_matrix() -> TspProblem {
        let flat_matrix = vec![0, 10, 15, 20, 10, 0, 35, 25, 15, 35, 0, 30, 20, 25, 30, 0];
        TspProblem::DistanceMatrix(DistanceMatrix::from_flat(flat_matrix))
    }

    #[test]
    fn test_adjacency_matrix_new() {
        let distance_matrix = create_test_distance_matrix();
        let adj_matrix = AdjacencyMatrix::new(&distance_matrix);

        assert_eq!(adj_matrix.problem.size(), 4);
        assert_eq!(adj_matrix.problem.distance(City(0), City(1)), 10);
    }

    #[test]
    fn test_adjacency_list_new() {
        let problem = create_test_problem();
        let list = vec![
            vec![City(1), City(3)],
            vec![City(0), City(2)],
            vec![City(1), City(3)],
            vec![City(0), City(2)],
        ];

        let adj_list = AdjacencyList::new(&problem, list.clone());

        assert_eq!(adj_list.problem.size(), 4);
        assert_eq!(adj_list.list.len(), 4);
        assert_eq!(adj_list.list[0], vec![City(1), City(3)]);
    }

    #[test]
    fn test_graphs_weight_matrix() {
        let distance_matrix = create_test_distance_matrix();
        let adj_matrix = AdjacencyMatrix::new(&distance_matrix);
        let graph = Graph::Matrix(adj_matrix);

        assert_eq!(graph.weight(City(0), City(1)), 10);
        assert_eq!(graph.weight(City(1), City(2)), 35);
        assert_eq!(graph.weight(City(0), City(3)), 20);
    }

    #[test]
    fn test_graphs_weight_list() {
        let problem = create_test_problem();
        let list = vec![vec![City(1)], vec![City(0)], vec![], vec![]];
        let adj_list = AdjacencyList::new(&problem, list);
        let graph = Graph::List(adj_list);

        let weight = graph.weight(City(0), City(1));
        assert!(weight > 0);
    }

    #[test]
    fn test_graphs_n_matrix() {
        let distance_matrix = create_test_distance_matrix();
        let adj_matrix = AdjacencyMatrix::new(&distance_matrix);
        let graph = Graph::Matrix(adj_matrix);

        assert_eq!(graph.n(), 4);
    }

    #[test]
    fn test_graphs_n_list() {
        let problem = create_test_problem();
        let list = vec![vec![], vec![], vec![], vec![]];
        let adj_list = AdjacencyList::new(&problem, list);
        let graph = Graph::List(adj_list);

        assert_eq!(graph.n(), 4);
    }

    #[test]
    fn test_graphs_m_matrix() {
        let distance_matrix = create_test_distance_matrix();
        let adj_matrix = AdjacencyMatrix::new(&distance_matrix);
        let graph = Graph::Matrix(adj_matrix);

        assert_eq!(graph.m(), 6);
    }

    #[test]
    fn test_graphs_m_list() {
        let problem = create_test_problem();
        let list = vec![
            vec![City(1), City(2)],
            vec![City(0), City(3)],
            vec![City(0), City(3)],
            vec![City(1), City(2)],
        ];
        let adj_list = AdjacencyList::new(&problem, list);
        let graph = Graph::List(adj_list);

        assert_eq!(graph.m(), 4);
    }

    #[test]
    fn test_graphs_neighbors_matrix() {
        let distance_matrix = create_test_distance_matrix();
        let adj_matrix = AdjacencyMatrix::new(&distance_matrix);
        let graph = Graph::Matrix(adj_matrix);

        let neighbors: Vec<City> = graph.neighbors(City(2)).collect();
        assert_eq!(neighbors.len(), 3);
        assert!(neighbors.contains(&City(0)));
        assert!(neighbors.contains(&City(1)));
        assert!(neighbors.contains(&City(3)));
        assert!(!neighbors.contains(&City(2)));
    }

    #[test]
    fn test_graphs_neighbors_in_list() {
        let problem = create_test_problem();
        let list = vec![vec![City(1), City(2)], vec![City(2)], vec![], vec![City(2)]];
        let adj_list = AdjacencyList::new(&problem, list);
        let graph = Graph::List(adj_list);

        // we migt need to fix this in the future, since this is not aaccording to a undirected graph's definition.
        // however, deciding which node has which neighbors is practically the candidate definition.
        // we have to see later in ejection chain definition if it makes sense to have both nodes to have references
        // to eachother or only one.
        let neighbors: Vec<City> = graph.neighbors(City(2)).collect();
        assert_eq!(neighbors.len(), 0);
    }

    #[test]
    fn test_graphs_edges_matrix() {
        let distance_matrix = create_test_distance_matrix();
        let adj_matrix = AdjacencyMatrix::new(&distance_matrix);
        let graph = Graph::Matrix(adj_matrix);

        let edges: Vec<Edge> = graph.edges().collect();
        assert_eq!(edges.len(), 6);

        assert!(edges.contains(&Edge::new(City(0), City(1))));
        assert!(edges.contains(&Edge::new(City(0), City(2))));
        assert!(edges.contains(&Edge::new(City(0), City(3))));
        assert!(edges.contains(&Edge::new(City(1), City(2))));
        assert!(edges.contains(&Edge::new(City(1), City(3))));
        assert!(edges.contains(&Edge::new(City(2), City(3))));
    }

    #[test]
    fn test_graphs_edges_list() {
        let problem = create_test_problem();
        let list = vec![
            vec![City(1), City(2)],
            vec![City(0), City(2), City(3)],
            vec![City(0), City(1)],
            vec![City(1)],
        ];
        let adj_list = AdjacencyList::new(&problem, list);
        let graph = Graph::List(adj_list);

        let edges: Vec<Edge> = graph.edges().collect();

        assert!(edges.contains(&Edge::new(City(0), City(1))));
        assert!(edges.contains(&Edge::new(City(0), City(2))));
        assert!(edges.contains(&Edge::new(City(1), City(2))));
        assert!(edges.contains(&Edge::new(City(1), City(3))));
    }

    #[test]
    fn test_graphs_cities() {
        let distance_matrix = create_test_distance_matrix();
        let adj_matrix = AdjacencyMatrix::new(&distance_matrix);
        let graph = Graph::Matrix(adj_matrix);

        let cities: Vec<City> = graph.cities().collect();
        assert_eq!(cities.len(), 4);
        assert_eq!(cities, vec![City(0), City(1), City(2), City(3)]);
    }

    #[test]
    fn test_graphs_complete_weight_matrix() {
        let distance_matrix = create_test_distance_matrix();
        let adj_matrix = AdjacencyMatrix::new(&distance_matrix);
        let graph = Graph::Matrix(adj_matrix);

        let total_weight = graph.complete_weight();
        assert_eq!(total_weight, 10 + 15 + 20 + 35 + 25 + 30);
    }

    #[test]
    fn test_graphs_complete_weight_list() {
        let problem = create_test_problem();
        let list = vec![
            vec![City(1), City(2), City(3)],
            vec![City(0), City(2), City(3)],
            vec![City(0), City(1), City(3)],
            vec![City(0), City(1), City(2)],
        ];
        let adj_list = AdjacencyList::new(&problem, list);
        let graph = Graph::List(adj_list);

        let total_weight = graph.complete_weight();
        assert!(total_weight > 0);
    }
}
