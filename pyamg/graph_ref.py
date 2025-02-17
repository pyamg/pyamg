"""Reference implementations of graph algorithms."""
import numpy as np


def bellman_ford_reference(A, c):
    """Execute reference implementation of Bellman-Ford.

    Parameters
    ----------
    A : coo sparse matrix
        n x n directed graph with positive weights
    c : array_like
        list of cluster centers

    Returns
    -------
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
    A = A.tocoo()
    nnodes = A.shape[0]
    d = np.full(nnodes, np.inf)
    m = np.full(nnodes, -1.0, dtype=np.int32)
    p = np.full(nnodes, -1.0, dtype=np.int32)

    d[c] = 0  # distance
    m[c] = np.arange(len(c))  # index

    done = False
    cnt = 0
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
    """Calculate reference implementation of balanced Bellman-Ford.

    Parameters
    ----------
    A : coo_array
        n x n directed graph with positive weights
    c : array
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
    # pylint: disable=too-many-nested-blocks
    A = A.tocoo()
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
    cnt = 0

    while not done:
        done = True
        for i, j, Aij in zip(A.row, A.col, A.data):
            if m[i] < 0:
                continue

            swap = False

            if d[i] + Aij < d[j]:  # BF
                swap = True

            si = s[m[i]] if m[i] >= 0 else 0
            sj = s[m[j]] if m[j] >= 0 else 0

            if m[j] > -1:
                if abs(d[i] + Aij - d[j]) < 1e-14:
                    if sj > (si + 1):
                        if pc[j] == 0:
                            swap = True

            if swap:
                if m[j] >= 0:      # if part of a cluster
                    s[m[j]] -= 1   # update size of cluster (removing j)
                if p[j] >= 0:      # if there's a predecessor
                    pc[p[j]] -= 1  # update predecessor count (removing j)

                m[j] = m[i]
                d[j] = d[i] + Aij
                p[j] = i

                s[m[j]] += 1   # update size of cluster (adding j)
                pc[p[j]] += 1  # update predecessor count (adding j)

                done = False

        cnt += 1

    return (d, m, p)
