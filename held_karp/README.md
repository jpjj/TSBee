This repository has one function: run(graph: &Graph, Option<f64>) -> (Vec<f64>, f64)
Here, the graph defines the graph the min-1-trees will be calculated on.
Option<f64> is an optional first upper bound for the optimal solution, that can increase convergence speed.
The return arguments, Vec<f64> will be the final Pi of the modified Graph with the largest Min-1-Tree.
f64 will be its complete cost and therefore a (hopefully tight) lower bound for the optimal tsp tour.
