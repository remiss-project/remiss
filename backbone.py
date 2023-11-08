import numpy as np


def compute_backbone(graph, alpha=0.05, delete_vertices=True):
    # Compute alpha for all edges (1 - weight_norm)^(degree_of_source_node - 1)
    weights = np.array(graph.es['weight_norm'])
    degrees = np.array([graph.degree(e[0]) for e in graph.get_edgelist()])
    alphas = (1 - weights) ** (degrees - 1)
    good = np.nonzero(alphas > alpha)[0]
    backbone = graph.subgraph_edges(graph.es.select(good), delete_vertices=delete_vertices)

    return backbone
