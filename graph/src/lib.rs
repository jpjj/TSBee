mod delaunay;

use tsp::{
    city::City,
    edge::Edge,
    problem::{Problem, TspProblem},
};

// Trait for weight adjustment strategies
pub trait WeightAdjuster: Clone {
    fn adjust(&self, base: i64, c1: City, c2: City) -> i64;
}

// State type for graphs without Pi values
#[derive(Debug, Clone, Default)]
pub struct WithoutPi;

// State type for graphs with Pi values
#[derive(Debug, Clone)]
pub struct WithPi {
    pub pi: Vec<i64>,
}

impl WeightAdjuster for WithoutPi {
    fn adjust(&self, base: i64, _c1: City, _c2: City) -> i64 {
        base
    }
}

impl WeightAdjuster for WithPi {
    fn adjust(&self, base: i64, c1: City, c2: City) -> i64 {
        base + self.pi[c1.0] + self.pi[c2.0]
    }
}

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

pub enum Graph<'a, State = WithoutPi> {
    Matrix(AdjacencyMatrix<'a>, State),
    List(AdjacencyList<'a>, State),
}

impl<'a, State: WeightAdjuster> Graph<'a, State> {
    pub fn problem(&self) -> &'a TspProblem {
        match self {
            Self::Matrix(am, _) => am.problem,
            Self::List(al, _) => al.problem,
        }
    }

    /// Get a reference to the state
    pub fn state(&self) -> &State {
        match self {
            Self::Matrix(_, state) | Self::List(_, state) => state,
        }
    }

    pub fn weight(&self, c1: City, c2: City) -> i64 {
        let (base, state) = match self {
            Self::Matrix(am, state) => (am.problem.distance(c1, c2), state),
            Self::List(al, state) => (al.problem.distance(c1, c2), state),
        };
        state.adjust(base, c1, c2)
    }

    pub fn edge_weight(&self, edge: Edge) -> i64 {
        self.weight(edge.u, edge.v)
    }

    pub fn n(&self) -> usize {
        match self {
            Self::Matrix(am, _) => am.problem.size(),
            Self::List(al, _) => al.problem.size(),
        }
    }

    pub fn m(&self) -> usize {
        match self {
            Self::Matrix(am, _) => {
                let n = am.problem.size();
                n * (n - 1) / 2
            }
            Self::List(al, _) => al.list.iter().map(|x| x.len()).sum::<usize>() / 2,
        }
    }

    pub fn neighbors(&self, c: City) -> impl Iterator<Item = City> {
        let n = self.n();
        let neighbors: Vec<City> = match self {
            Self::Matrix(_, _) => (0..n).filter(|&i| i != c.0).map(City).collect(),
            Self::List(al, _) => al.list[c.0].to_vec(),
        };
        neighbors.into_iter()
    }

    pub fn edges(&self) -> impl Iterator<Item = Edge> {
        let edges: Vec<Edge> = match self {
            Self::Matrix(am, _) => {
                let n = am.problem.size();
                (0..n)
                    .flat_map(|i| (i + 1..n).map(move |j| Edge::new(City(i), City(j))))
                    .collect()
            }
            Self::List(al, _) => al
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

// State transition methods
impl<'a> Graph<'a, WithoutPi> {
    pub fn with_pi(self, pi: Vec<i64>) -> Graph<'a, WithPi> {
        match self {
            Graph::Matrix(am, _) => Graph::Matrix(am, WithPi { pi }),
            Graph::List(al, _) => Graph::List(al, WithPi { pi }),
        }
    }
}

impl<'a> Graph<'a, WithPi> {
    pub fn without_pi(self) -> Graph<'a, WithoutPi> {
        match self {
            Graph::Matrix(am, _) => Graph::Matrix(am, WithoutPi),
            Graph::List(al, _) => Graph::List(al, WithoutPi),
        }
    }

    pub fn get_pi(&self) -> &[i64] {
        match self {
            Graph::Matrix(_, state) | Graph::List(_, state) => &state.pi,
        }
    }

    pub fn get_pi_mut(&mut self) -> &mut [i64] {
        match self {
            Graph::Matrix(_, state) | Graph::List(_, state) => &mut state.pi,
        }
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
    fn test_graph_weight_matrix() {
        let distance_matrix = create_test_distance_matrix();
        let adj_matrix = AdjacencyMatrix::new(&distance_matrix);
        let graph = Graph::Matrix(adj_matrix, WithoutPi);

        assert_eq!(graph.weight(City(0), City(1)), 10);
        assert_eq!(graph.weight(City(1), City(2)), 35);
        assert_eq!(graph.weight(City(0), City(3)), 20);
    }

    #[test]
    fn test_graph_weight_list() {
        let problem = create_test_problem();
        let list = vec![vec![City(1)], vec![City(0)], vec![], vec![]];
        let adj_list = AdjacencyList::new(&problem, list);
        let graph = Graph::List(adj_list, WithoutPi);

        let weight = graph.weight(City(0), City(1));
        assert!(weight > 0);
    }

    #[test]
    fn test_graph_n_matrix() {
        let distance_matrix = create_test_distance_matrix();
        let adj_matrix = AdjacencyMatrix::new(&distance_matrix);
        let graph = Graph::Matrix(adj_matrix, WithoutPi);

        assert_eq!(graph.n(), 4);
    }

    #[test]
    fn test_graph_n_list() {
        let problem = create_test_problem();
        let list = vec![vec![], vec![], vec![], vec![]];
        let adj_list = AdjacencyList::new(&problem, list);
        let graph = Graph::List(adj_list, WithoutPi);

        assert_eq!(graph.n(), 4);
    }

    #[test]
    fn test_graph_m_matrix() {
        let distance_matrix = create_test_distance_matrix();
        let adj_matrix = AdjacencyMatrix::new(&distance_matrix);
        let graph = Graph::Matrix(adj_matrix, WithoutPi);

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
        let adj_list = AdjacencyList::new(&problem, list);
        let graph = Graph::List(adj_list, WithoutPi);

        assert_eq!(graph.m(), 4);
    }

    #[test]
    fn test_graph_neighbors_matrix() {
        let distance_matrix = create_test_distance_matrix();
        let adj_matrix = AdjacencyMatrix::new(&distance_matrix);
        let graph = Graph::Matrix(adj_matrix, WithoutPi);

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
        let adj_list = AdjacencyList::new(&problem, list);
        let graph = Graph::List(adj_list, WithoutPi);

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
        let adj_matrix = AdjacencyMatrix::new(&distance_matrix);
        let graph = Graph::Matrix(adj_matrix, WithoutPi);

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
        let graph = Graph::List(adj_list, WithoutPi);

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
        let graph = Graph::Matrix(adj_matrix, WithoutPi);

        let cities: Vec<City> = graph.cities().collect();
        assert_eq!(cities.len(), 4);
        assert_eq!(cities, vec![City(0), City(1), City(2), City(3)]);
    }

    #[test]
    fn test_graphs_complete_weight_matrix() {
        let distance_matrix = create_test_distance_matrix();
        let adj_matrix = AdjacencyMatrix::new(&distance_matrix);
        let graph = Graph::Matrix(adj_matrix, WithoutPi);

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
        let graph = Graph::List(adj_list, WithoutPi);

        let total_weight = graph.complete_weight();
        assert!(total_weight > 0);
    }

    #[test]
    fn test_graph_with_pi() {
        let distance_matrix = create_test_distance_matrix();
        let adj_matrix = AdjacencyMatrix::new(&distance_matrix);
        let graph = Graph::Matrix(adj_matrix, WithoutPi);

        // Test standard weight calculation
        assert_eq!(graph.weight(City(0), City(1)), 10);
        assert_eq!(graph.weight(City(1), City(2)), 35);

        // Add Pi values
        let pi_values = vec![1, 2, 3, 4];
        let pi_graph = graph.with_pi(pi_values);

        // Test Pi-adjusted weight calculation
        assert_eq!(pi_graph.weight(City(0), City(1)), 10 + 1 + 2); // 13
        assert_eq!(pi_graph.weight(City(1), City(2)), 35 + 2 + 3); // 40

        // Test getting Pi values
        assert_eq!(pi_graph.get_pi(), &[1, 2, 3, 4]);

        // Test removing Pi
        let standard_graph = pi_graph.without_pi();
        assert_eq!(standard_graph.weight(City(0), City(1)), 10);
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
        let adj_list = AdjacencyList::new(&problem, list);
        let graph = Graph::List(adj_list, WithoutPi);

        let base_weight = graph.weight(City(0), City(1));

        // Add Pi values
        let pi_values = vec![10, 20, 30, 40];
        let pi_graph = graph.with_pi(pi_values);

        // Test Pi-adjusted weight
        assert_eq!(pi_graph.weight(City(0), City(1)), base_weight + 10 + 20);
        assert_eq!(
            pi_graph.weight(City(2), City(3)),
            pi_graph.problem().distance(City(2), City(3)) + 30 + 40
        );
    }
}
