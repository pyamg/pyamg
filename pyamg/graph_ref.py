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
    p : ndarray
        predecessor

    See Also
    --------
    amg_core.graph.bellman_ford

    """
    nnodes = A.shape[0]
    nclusters = len(c)
    d = np.full(nnodes, np.inf)
    m = np.full(nnodes, -1.0, dtype=np.int32)
    p = np.full(nnodes, -1.0, dtype=np.int32)

    d[c] = 0  # distance
    m[c] = np.arange(len(c))  # index

    done = False
    cnt = 0;
    while not done:
        done = True
        for i, j, Aij in zip(A.row, A.col, A.data):
            if d[i] + Aij < d[j]:
                d[j] = d[i] + Aij
                m[j] = m[i]
                p[j] = i
                done = False
        cnt += 1

    return (d, m, p)


def bellman_ford_balanced_reference(A, c):
    """Reference implementation of balanced Bellman-Ford.

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
    p : ndarray
        predecessor

    See Also
    --------
    amg_core.graph.bellman_ford

    """
    nnodes = A.shape[0]
    nclusters = len(c)
    d = np.full(nnodes, np.inf)
    m = np.full(nnodes, -1.0, dtype=np.int32)
    p = np.full(nnodes, -1.0, dtype=np.int32)

    d[c] = 0  # distance
    m[c] = np.arange(len(c))  # index

    pc = np.zeros(nnodes, dtype=np.int32)
    s = np.ones(nclusters, dtype=np.int32)

    done = False
    cnt = 0;
    while not done:
        done = True
        for i, j, Aij in zip(A.row, A.col, A.data):
            if m[i] < 0:
                continue

            swap = False

            if d[i] + Aij < d[j]: # BF
                swap = True

            si = s[m[i]] if m[i] >= 0 else 0
            sj = s[m[j]] if m[j] >= 0 else 0

            if m[j] > -1:
                if abs(d[i] + Aij - d[j]) < 1e-14:
                    if sj > (si + 1):
                        if pc[j] == 0:
                            swap = True

            if swap:
                if m[j] >= 0:     # if part of a cluster
                    s[m[j]] -= 1  # update size of cluster (removing j)
                if p[j] >= 0:     # if there's a predecessor
                    pc[p[j]] -= 1 # update predecessor count (removing j)

                m[j] = m[i]
                d[j] = d[i] + Aij
                p[j] = i

                s[m[j]] += 1  # update size of cluster (adding j)
                pc[p[j]] += 1 # update predecessor count (adding j)

                done = False

        cnt += 1

    return (d, m, p)


if __name__  == '__main__':
    if 1:
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
        A = A.tocsr()
        A = A.tocoo()
        c = np.array([0,1,2,3,4])

    if 0:
        import pyamg
        data = pyamg.gallery.load_example('unit_square')

        A = data['A'].tocoo()
        A.data = np.abs(A.data)
        n = A.shape[0]
        c = np.array([0,1,2,3,4])

    print(A.toarray())

    print('\nreference--')
    print(type(A))
    for cc in c:
        d, m, p = bellman_ford_reference(A, [cc])
        print(d, m, p)

    if 0:
        print('\nreference balanced--')
        for cc in c:
            d, m, p = bellman_ford_balanced_reference(A, [cc])
            print(d, m, p)

    if 0:
        print('\npyamg--')
        from pyamg.graph import bellman_ford
        from pyamg import amg_core
        A = A.tocsr()
        for cc in c:
            #d, m, p = bellman_ford(A, [cc])
            cc = np.asarray(cc, dtype=np.int32)
            n = A.shape[0]
            d = np.full(n, np.inf, dtype=A.dtype)
            d[cc] = 0
            m = np.full(n, -1, dtype=np.int32)
            m[cc] = cc
            p = np.full(n, -1, dtype=np.int32)
            amg_core.bellman_ford(n, A.indptr, A.indices, A.data, d, m, p)
            print(d, m, p)

    print('\ncsgraph.bellman_ford')
    from scipy.sparse import csgraph
    for cc in c:
        d, p = csgraph.bellman_ford(A, directed=True, indices=[cc], return_predecessors=True)
        print(d.ravel(), p.ravel())
