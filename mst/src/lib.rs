use graph::Graph;
use tsp::edge::Edge;

pub struct Kruskal<'a> {
    graph: &'a Graph<'a>,
}

impl<'a> Kruskal<'a> {
    pub fn new(graph: &'a Graph<'a>) -> Self {
        Kruskal { graph }
    }
    pub fn get_mst(&self) -> (Vec<Edge>, i64) {
        let mut edges: Vec<Edge> = self.graph.edges().collect();
        self.sort_edges(&mut edges);
        self.get_mst_from_sorted_edges(&edges)
    }

    pub fn sort_edges(&self, edges: &mut [Edge]) {
        dmsort::sort_by_key(edges, |e: &Edge| self.graph.weight(e.u, e.v));
    }

    pub fn get_mst_from_sorted_edges(&self, edges: &[Edge]) -> (Vec<Edge>, i64) {
        let mut uf = UnionFind::new(self.graph.n());
        let mut mst_edges = Vec::with_capacity(self.graph.n());
        let mut total_weight = 0;

        for edge in edges.iter() {
            if uf.union(edge.u.0, edge.v.0) {
                total_weight += self.graph.weight(edge.u, edge.v);
                mst_edges.push(*edge);
                if mst_edges.len() == self.graph.n() - 1 {
                    break;
                }
            }
        }

        (mst_edges, total_weight)
    }
}

pub struct UnionFind {
    pub parent: Vec<usize>,
    pub rank: Vec<usize>,
}

impl UnionFind {
    pub fn new(n: usize) -> Self {
        UnionFind {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    #[inline]
    pub fn find(&mut self, x: usize) -> usize {
        let mut y = x;
        unsafe {
            let mut p = *self.parent.get_unchecked(y);
            while y != p {
                let grandparent = *self.parent.get_unchecked(p);
                *self.parent.get_unchecked_mut(y) = grandparent;
                y = p;
                p = grandparent;
            }
        }
        y
    }

    #[inline]
    pub fn union(&mut self, x: usize, y: usize) -> bool {
        let root_x = self.find(x);
        let root_y = self.find(y);

        if root_x == root_y {
            return false;
        }
        match self.rank[root_x].cmp(&self.rank[root_y]) {
            std::cmp::Ordering::Less => {
                self.parent[root_x] = root_y;
            }
            std::cmp::Ordering::Greater => {
                self.parent[root_y] = root_x;
            }
            std::cmp::Ordering::Equal => {
                self.parent[root_y] = root_x;
                self.rank[root_x] += 1;
            }
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use graph::{AdjacencyList, AdjacencyMatrix};
    use tsp::{
        city::City,
        problem::{TspProblem, distance_matrix::DistanceMatrix},
    };

    use super::*;
    fn create_test_distance_matrix() -> DistanceMatrix<i64> {
        let flat_matrix = vec![0, 10, 15, 20, 10, 0, 35, 25, 15, 35, 0, 30, 20, 25, 30, 0];
        DistanceMatrix::from_flat(flat_matrix)
    }
    #[test]
    fn test_kruskal_mst() {
        let distance_matrix = create_test_distance_matrix();
        let graph = Graph::Matrix(AdjacencyMatrix::new(&distance_matrix));
        let kruskal = Kruskal::new(&graph);

        let (mst_edges, total_weight) = kruskal.get_mst();

        assert_eq!(mst_edges.len(), 3);
        assert_eq!(total_weight, 45);
    }

    #[test]
    fn test_disconnected_graph() {
        let distance_matrix = create_test_distance_matrix();
        let problem = TspProblem::DistanceMatrix(distance_matrix);
        let graph = Graph::List(AdjacencyList::new(
            &problem,
            vec![vec![City(1)], vec![], vec![City(3)], vec![]],
        ));
        let kruskal = Kruskal::new(&graph);

        let (mst_edges, total_weight) = kruskal.get_mst();

        assert_eq!(mst_edges.len(), 2);
        assert_eq!(total_weight, 40);
    }

    #[test]
    fn test_stepwise_kruskal() {
        let distance_matrix = create_test_distance_matrix();
        let graph = Graph::Matrix(AdjacencyMatrix::new(&distance_matrix));
        let kruskal = Kruskal::new(&graph);
        let mut edges: Vec<Edge> = graph.edges().collect();
        kruskal.sort_edges(&mut edges);
        let (mst_edges, total_weight) = kruskal.get_mst_from_sorted_edges(&edges);
        assert_eq!(mst_edges.len(), 3);
        assert_eq!(total_weight, 45);
    }
}
