from igraph import Graph
import numpy as np

def backbone(graph, weights=None, alpha=0.05):

    if weights is None:
        weights = graph.es["weight"]

    directed = graph.is_directed()
    if not directed:
        b = disparity_filter(graph, weights, "all", alpha)
    else:
        in_filter = disparity_filter(graph, weights, "in", alpha)
        out_filter = disparity_filter(graph, weights, "out", alpha)
        b = in_filter + out_filter

    unique_edges = sorted(set(b), key=lambda e: (e.source, e.target))
    return unique_edges

def disparity_filter(G, weights, mode="all", alpha=0.05):
    d = G.degree(mode=mode)
    e = np.array(G.get_edgelist())
    e = np.column_stack((e, weights, [np.nan] * len(e)))

    if mode == "all":
        reversed_edges = np.column_stack((e[:, 1], e[:, 0], e[:, 2], e[:, 3]))
        e = np.vstack((e, reversed_edges))

    alpha_values = []
    for u in [v for v, degree in enumerate(d) if degree > 1]:
        if mode == "all":
            u_neighbors = np.where((e[:, 0] == u) | (e[:, 1] == u))[0]
        elif mode == "in":
            u_neighbors = np.where(e[:, 1] == u)[0]
        elif mode == "out":
            u_neighbors = np.where(e[:, 0] == u)[0]

        w = np.sum(e[u_neighbors, 2]) / (1 + (mode == "all"))
        k = d[u]

        for v in G.neighborhood(vertices=u, order=1)[0][1:]:
            if mode == "all":
                ij = np.where((e[:, 0] == u) & (e[:, 1] == v) | (e[:, 0] == v) & (e[:, 1] == u))[0]
            elif mode == "in":
                ij = np.where((e[:, 1] == u) & (e[:, 0] == v))[0]
            elif mode == "out":
                ij = np.where((e[:, 0] == u) & (e[:, 1] == v))[0]

            alpha_ij = (1 - e[ij, 2] / w) ** (k - 1)
            alpha_values.append(alpha_ij)

    e_alpha = e[np.isfinite(e[:, 3]), :]
    e_alpha = e_alpha[e_alpha[:, 3] < alpha]
    return e_alpha[:, :4]

# Example usage
g = Graph.Erdos_Renyi(250, 0.02, directed=False)
g.es["weight"] = [np.random.randint(1, 25) for _ in range(g.ecount())]
backbone_edges = backbone(g)
