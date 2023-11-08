import numpy as np


def compute_backbone(graph, alpha=0.05, delete_vertices=True):
    # Compute alpha for all edges (1 - weight_norm)^(degree_of_source_node - 1)
    degrees = np.array(graph.degree())
    weights = np.array(graph.es['weight_norm'])
    alphas = (1 - weights) ** (degrees - 1)
    good = alphas > alpha
    backbone = graph.subgraph_edges(graph.es.select(good), delete_vertices=delete_vertices)

    return backbone
