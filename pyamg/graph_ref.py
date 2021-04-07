"""Reference implementations of graph algorithms."""
import numpy as np
import scipy.sparse as sparse


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
    Ncluster = len(c)
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


if __name__  == '__main__':
    Edges = np.array([[1, 4],
                      [3, 1],
                      [1, 3],
                      [0, 1],
                      [0, 2],
                      [3, 2],
                      [1, 2],
                      [4, 3]])
    w = np.array([2, 1, 2, 1, 4, 5, 3, 1], dtype=float)
    A = sparse.coo_matrix((w, (Edges[:, 0], Edges[:, 1])))
    c = np.array([0,1,2,3,4])

    print('\nreference--')
    for cc in c:
        d, m = bellman_ford_reference(A, [cc])
        print(d, m)

    print('\npyamg--')
    from pyamg.graph import bellman_ford
    for cc in c:
        d, m = bellman_ford(A, [cc])
        print(d, m)

    print('\ncsgraph.bellman_ford')
    from scipy.sparse import csgraph
    for cc in c:
        d, p = csgraph.bellman_ford(A, directed=True, indices=[cc], return_predecessors=True)
        print(d.ravel(), p.ravel())
