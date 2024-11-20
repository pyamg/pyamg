"""Reference implementations of graph algorithms."""
import numpy as np

def bellman_ford_reference(adjacency_matrix, cluster_centers):
    """Execute reference implementation of Bellman-Ford.

    Parameters
    ----------
    adjacency_matrix : coo sparse matrix
        n x n directed graph with positive weights

    cluster_centers : array_like
        list of cluster centers

    Returns
    -------
    cluster_index : ndarray
        cluster index

    distances : ndarray
        distance to cluster center

    See Also
    --------
    amg_core.graph.bellman_ford

    """
    n_node = adjacency_matrix.shape[0]
    distances = np.full((n_node,), np.inf)
    cluster_index = np.full((n_node,), -1.0, dtype=np.int32)

    distances[cluster_centers] = 0  # distance
    cluster_index[cluster_centers] = cluster_centers  # index

    done = False
    while not done:
        done = True
        for i, j, adjacency_value in zip(adjacency_matrix.row, adjacency_matrix.col, adjacency_matrix.data):
            if adjacency_value > 0 and distances[i] + adjacency_value < distances[j]:
                distances[j] = distances[i] + adjacency_value
                cluster_index[j] = cluster_index[i]
                done = False

    return (distances, cluster_index)
