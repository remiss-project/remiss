from igraph import Graph
import numpy as np

def compute_backbone(graph, alpha=0.05):
    return graph.es[np.where(graph.es["weight"] < alpha)]


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
