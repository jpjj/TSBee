# Alpha Nearness

## Goal of this Repo
Given a Graph and an optional slice of Edges, return values for every edge (their alpha value).
If no slice of edges is given, return an alpha value for every edge of the graph.

### How to calculate Alpha values
1. Calculate the min-1-tree of the graph (and the optional edges).
2. Given the edges of the min 1-tree, calculate the alpha values as follows:
3. for any edge e incident to City(n-1):
    - Return 0 if edge belongs to min-1-tree (i.e. it is one of the two smallest edges incident to City(n-1)).
    - Return w(e) - w(e_2) where e_2 is the second smallest edge incident to City(n-1).
4. For any other edge:
  - For every city c 0, ..., =n-2:
    - Start a DFS from c on the MST of the Graph (without City(n-1) and its two edges)
    - Additionally to the stack being built during DFS, save in the heaviest weight encountered w_max on the path from c to current node v and which nodes have been visited.
    - With all that, proceed as follows:
        1. let u = stack.pop()
        2. for v in mst.neighbors_out(u):
            - If v not visited:
                - w_max(v) = max(w_max(u), weight(u,v))
                - alpha(c, v) = weight(c, v) - w_max(u)
                - stack.apped(v)
    - Proceding like this one gets all alpha values for all possible edges.

### Consideration
What if optional edges are given?

In this case, only alpha values for the edges are wanted.
To be checked in the future.
Hence, we can skip the optional edge input for now.


You can problably cut the calculation of alpha nearness at least in half using a smarter topological sorting strategy. But so far, this should suffice.
Optimization can be done later. Bottleneck might also be the allocation of n*n entries. Let's see.
