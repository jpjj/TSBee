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

pub struct Graph<'a> {
    inner: GraphInner<'a>,
    pub pi: Vec<f64>,
}

enum GraphInner<'a> {
    Matrix(AdjacencyMatrix<'a>),
    List(AdjacencyList<'a>),
}

impl<'a> Graph<'a> {
    pub fn new_matrix(problem: &'a TspProblem) -> Self {
        let n = problem.size();
        Self {
            inner: GraphInner::Matrix(AdjacencyMatrix::new(problem)),
            pi: vec![0.0; n],
        }
    }

    pub fn new_list(problem: &'a TspProblem, list: Vec<Vec<City>>) -> Self {
        let n = problem.size();
        Self {
            inner: GraphInner::List(AdjacencyList::new(problem, list)),
            pi: vec![0.0; n],
        }
    }

    pub fn new_list_from_edges(problem: &'a TspProblem, edges: Vec<Edge>) -> Self {
        let n = problem.size();
        Self {
            inner: GraphInner::List(AdjacencyList::from_edges(problem, edges)),
            pi: vec![0.0; n],
        }
    }

    pub fn problem(&self) -> &'a TspProblem {
        match &self.inner {
            GraphInner::Matrix(am) => am.problem,
            GraphInner::List(al) => al.problem,
        }
    }

    pub fn weight(&self, c1: City, c2: City) -> f64 {
        let base = match &self.inner {
            GraphInner::Matrix(am) => am.problem.distance(c1, c2),
            GraphInner::List(al) => al.problem.distance(c1, c2),
        };
        if self.pi.is_empty() {
            base
        } else {
            base + self.pi[c1.0] + self.pi[c2.0]
        }
    }

    pub fn edge_weight(&self, edge: Edge) -> f64 {
        self.weight(edge.u, edge.v)
    }

    pub fn n(&self) -> usize {
        match &self.inner {
            GraphInner::Matrix(am) => am.problem.size(),
            GraphInner::List(al) => al.problem.size(),
        }
    }

    pub fn m(&self) -> usize {
        match &self.inner {
            GraphInner::Matrix(am) => {
                let n = am.problem.size();
                n * (n - 1) / 2
            }
            GraphInner::List(al) => al.list.iter().map(|x| x.len()).sum::<usize>() / 2,
        }
    }

    pub fn neighbors(&self, c: City) -> impl Iterator<Item = City> {
        let n = self.n();
        let neighbors: Vec<City> = match &self.inner {
            GraphInner::Matrix(_) => (0..n).filter(|&i| i != c.0).map(City).collect(),
            GraphInner::List(al) => al.list[c.0].to_vec(),
        };
        neighbors.into_iter()
    }

    pub fn degrees(&self) -> Vec<i32> {
        match &self.inner {
            GraphInner::Matrix(_) => vec![self.n() as i32 - 1; self.n()],
            GraphInner::List(al) => (0..self.n())
                .map(|c_idx| al.list[c_idx].len() as i32)
                .collect::<Vec<i32>>(),
        }
    }

    pub fn edges(&self) -> impl Iterator<Item = Edge> {
        let edges: Vec<Edge> = match &self.inner {
            GraphInner::Matrix(am) => {
                let n = am.problem.size();
                (0..n)
                    .flat_map(|i| (i + 1..n).map(move |j| Edge::new(City(i), City(j))))
                    .collect()
            }
            GraphInner::List(al) => al
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

    pub fn complete_weight(&self) -> f64 {
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

        let problem = PointsAndFunction::<f64, f64, Euc2d>::new(points);
        TspProblem::Euclidean(problem)
    }

    fn create_test_distance_matrix() -> TspProblem {
        let flat_matrix = vec![
            0.0, 10.0, 15.0, 20.0, 10.0, 0.0, 35.0, 25.0, 15.0, 35.0, 0.0, 30.0, 20.0, 25.0, 30.0,
            0.0,
        ];
        TspProblem::DistanceMatrix(DistanceMatrix::from_flat(flat_matrix))
    }

    #[test]
    fn test_adjacency_matrix_new() {
        let distance_matrix = create_test_distance_matrix();
        let adj_matrix = AdjacencyMatrix::new(&distance_matrix);

        assert_eq!(adj_matrix.problem.size(), 4);
        assert_eq!(adj_matrix.problem.distance(City(0), City(1)), 10.0);
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
    fn test_graph_weight_matrix() {
        let distance_matrix = create_test_distance_matrix();
        let graph = Graph::new_matrix(&distance_matrix);

        assert_eq!(graph.weight(City(0), City(1)), 10.0);
        assert_eq!(graph.weight(City(1), City(2)), 35.0);
        assert_eq!(graph.weight(City(0), City(3)), 20.0);
    }

    #[test]
    fn test_graph_weight_list() {
        let problem = create_test_problem();
        let list = vec![vec![City(1)], vec![City(0)], vec![], vec![]];
        let graph = Graph::new_list(&problem, list);

        let weight = graph.weight(City(0), City(1));
        assert!(weight > 0.0);
    }

    #[test]
    fn test_graph_n_matrix() {
        let distance_matrix = create_test_distance_matrix();
        let graph = Graph::new_matrix(&distance_matrix);

        assert_eq!(graph.n(), 4);
    }

    #[test]
    fn test_graph_n_list() {
        let problem = create_test_problem();
        let list = vec![vec![], vec![], vec![], vec![]];
        let graph = Graph::new_list(&problem, list);

        assert_eq!(graph.n(), 4);
    }

    #[test]
    fn test_graph_m_matrix() {
        let distance_matrix = create_test_distance_matrix();
        let graph = Graph::new_matrix(&distance_matrix);

        assert_eq!(graph.m(), 6);
    }

    #[test]
    fn test_graph_m_list() {
        let problem = create_test_problem();
        let list = vec![
            vec![City(1), City(2)],
            vec![City(0), City(3)],
            vec![City(0), City(3)],
            vec![City(1), City(2)],
        ];
        let graph = Graph::new_list(&problem, list);

        assert_eq!(graph.m(), 4);
    }

    #[test]
    fn test_graph_neighbors_matrix() {
        let distance_matrix = create_test_distance_matrix();
        let graph = Graph::new_matrix(&distance_matrix);

        let neighbors: Vec<City> = graph.neighbors(City(2)).collect();
        assert_eq!(neighbors.len(), 3);
        assert!(neighbors.contains(&City(0)));
        assert!(neighbors.contains(&City(1)));
        assert!(neighbors.contains(&City(3)));
        assert!(!neighbors.contains(&City(2)));
    }

    #[test]
    fn test_graph_neighbors_in_list() {
        let problem = create_test_problem();
        let list = vec![vec![City(1), City(2)], vec![City(2)], vec![], vec![City(2)]];
        let graph = Graph::new_list(&problem, list);

        // we might need to fix this in the future, since this is not aaccording to a undirected graph's definition.
        // however, deciding which node has which neighbors is practically the candidate definition.
        // we have to see later in ejection chain definition if it makes sense to have both nodes to have references
        // to eachother or only one.
        let neighbors: Vec<City> = graph.neighbors(City(2)).collect();
        assert_eq!(neighbors.len(), 0);
    }

    #[test]
    fn test_graph_edges_matrix() {
        let distance_matrix = create_test_distance_matrix();
        let graph = Graph::new_matrix(&distance_matrix);

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
        let graph = Graph::new_list(&problem, list);

        let edges: Vec<Edge> = graph.edges().collect();

        assert!(edges.contains(&Edge::new(City(0), City(1))));
        assert!(edges.contains(&Edge::new(City(0), City(2))));
        assert!(edges.contains(&Edge::new(City(1), City(2))));
        assert!(edges.contains(&Edge::new(City(1), City(3))));
    }

    #[test]
    fn test_graphs_cities() {
        let distance_matrix = create_test_distance_matrix();
        let graph = Graph::new_matrix(&distance_matrix);

        let cities: Vec<City> = graph.cities().collect();
        assert_eq!(cities.len(), 4);
        assert_eq!(cities, vec![City(0), City(1), City(2), City(3)]);
    }

    #[test]
    fn test_graphs_complete_weight_matrix() {
        let distance_matrix = create_test_distance_matrix();
        let graph = Graph::new_matrix(&distance_matrix);

        let total_weight = graph.complete_weight();
        assert_eq!(total_weight, 10.0 + 15.0 + 20.0 + 35.0 + 25.0 + 30.0);
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
        let graph = Graph::new_list(&problem, list);

        let total_weight = graph.complete_weight();
        assert!(total_weight > 0.0);
    }

    #[test]
    fn test_graph_with_pi() {
        let distance_matrix = create_test_distance_matrix();
        let mut graph = Graph::new_matrix(&distance_matrix);

        // Test standard weight calculation (pi is initialized with zeros)
        assert_eq!(graph.weight(City(0), City(1)), 10.0);
        assert_eq!(graph.weight(City(1), City(2)), 35.0);

        // Update Pi values
        graph.pi = vec![1.0, 2.0, 3.0, 4.0];

        // Test Pi-adjusted weight calculation
        assert_eq!(graph.weight(City(0), City(1)), 10.0 + 1.0 + 2.0); // 13
        assert_eq!(graph.weight(City(1), City(2)), 35.0 + 2.0 + 3.0); // 40

        // Test getting Pi values
        assert_eq!(graph.pi, vec![1.0, 2.0, 3.0, 4.0]);

        // Test removing Pi (set back to zeros)
        graph.pi = vec![0.0, 0.0, 0.0, 0.0];
        assert_eq!(graph.weight(City(0), City(1)), 10.0);
    }

    #[test]
    fn test_graph_with_pi_list() {
        let problem = create_test_problem();
        let list = vec![
            vec![City(1), City(2), City(3)],
            vec![City(0), City(2), City(3)],
            vec![City(0), City(1), City(3)],
            vec![City(0), City(1), City(2)],
        ];
        let mut graph = Graph::new_list(&problem, list);

        let base_weight = graph.weight(City(0), City(1));

        // Update Pi values
        graph.pi = vec![10.0, 20.0, 30.0, 40.0];

        // Test Pi-adjusted weight
        assert_eq!(graph.weight(City(0), City(1)), base_weight + 10.0 + 20.0);
        assert_eq!(
            graph.weight(City(2), City(3)),
            graph.problem().distance(City(2), City(3)) + 30.0 + 40.0
        );
    }
}
