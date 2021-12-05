"""Reference implementations of graph algorithms."""
import numpy as np


def bellman_ford_reference(A, c):
    """Reference implementation of Bellman-Ford.

    Parameters
    ---------
    A : coo sparse matrix
        n x n directed graph with positive weights

    c : array_like
        list of cluster centers

    Return
    ------
    m : ndarray
        cluster index

    d : ndarray
        distance to cluster center

    See Also
    --------
    amg_core.graph.bellman_ford

    """
    Nnode = A.shape[0]
    d = np.full((Nnode,), np.inf)
    m = np.full((Nnode,), -1.0, dtype=np.int32)

    d[c] = 0  # distance
    m[c] = c  # index

    done = False
    while not done:
        done = True
        for i, j, Aij in zip(A.row, A.col, A.data):
            if Aij > 0 and d[i] + Aij < d[j]:
                d[j] = d[i] + Aij
                m[j] = m[i]
                done = False

    return (d, m)
