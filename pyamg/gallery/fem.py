"""Poisson problem with finite elements
"""
import numpy as np
from scipy import sparse


def check_mesh(V, E):
    """Check the ccw orientation of each simplex in the mesh
    """
    E01 = np.vstack((V[E[:, 1], 0] - V[E[:, 0], 0],
                     V[E[:, 1], 1] - V[E[:, 0], 1],
                     np.zeros(E.shape[0]))).T
    E12 = np.vstack((V[E[:, 2], 0] - V[E[:, 1], 0],
                     V[E[:, 2], 1] - V[E[:, 1], 1],
                     np.zeros(E.shape[0]))).T
    orientation = np.all(np.cross(E01, E12)[:, 2] > 0)

    return orientation


def generate_quadratic(V, E, return_edges=False):
    """Generate a quadratic element list by adding midpoints to each edge

    Parameters
    ----------
    V : ndarray
        nv x 2 list of coordinates

    E : ndarray
        ne x 3 list of vertices

    return_edges : bool
        indicate whether list of the refined edges is returned

    Returns
    -------
    V2 : ndarray
        nv2 x 2 list of coordinates

    E2 : ndarray
        ne2 x 6 list of vertices

    Edges : ndarray
        ned x 2 list of edges where the midpoint is generated

    Notes
    -----
        - midpoints are introduced and globally numbered at the end of the vertex list
        - the element list includes the new list beteen v0-v1, v1-v2, and v2-v0

    Examples
    --------
    >>> import numpy as np
    >>> V = np.array([[0.,0.], [1.,0.], [0.,1.], [1.,1.]])
    >>> E = np.array([[0,1,2], [2,3,1]])
    >>> import fem
    >>> V2, E2 = fem.generate_quadratic(V, E)
    array([[0. , 0. ],
           [1. , 0. ],
           [0. , 1. ],
           [1. , 1. ],
           [0.5, 0. ],
           [0.5, 0.5],
           [0. , 0.5],
           [0.5, 1. ],
           [1. , 0.5]])
    array([[0, 1, 2, 4, 5, 6],
           [2, 3, 1, 7, 8, 5]])
    """

    if not isinstance(V, np.ndarray) or not isinstance(E, np.ndarray):
        raise ValueError('V and E must be ndarray')

    if V.shape[1] != 2 or E.shape[1] != 3:
        raise ValueError('V should be nv x 2 and E should be ne x 3')

    ne = E.shape[0]

    # make a vertext-to-vertex graph
    ID = np.kron(np.arange(0, ne), np.ones((3,), dtype=int))
    G = sparse.coo_matrix((np.ones((ne*3,), dtype=int), (E.ravel(), ID,)))
    V2V = G * G.T

    # from the vertex graph, get the edges and create new midpoints
    V2Vmid = sparse.tril(V2V, -1)
    Edges = np.vstack((V2Vmid.row, V2Vmid.col)).T
    Vmid = (V[Edges[:, 0], :] + V[Edges[:, 1], :]) / 2.0
    V = np.vstack((V, Vmid))

    # enumerate the new midpoints for the edges
    # V2Vmid[i,j] will have the new number of the midpoint between i and j
    maxindex = E.max() + 1
    newID = maxindex + np.arange(Edges.shape[0])
    V2Vmid.data = newID
    V2Vmid = V2Vmid + V2Vmid.T

    # from the midpoints, extend E
    E = np.hstack((E, np.zeros((E.shape[0], 3), dtype=int)))
    E[:, 3] = V2Vmid[E[:, 0], E[:, 1]]
    E[:, 4] = V2Vmid[E[:, 1], E[:, 2]]
    E[:, 5] = V2Vmid[E[:, 2], E[:, 0]]

    if return_edges:
        return V, E, Edges

    return V, E


def diameter(V, E):
    """Compute the diameter of a mesh

    Parameters
    ----------
    V : ndarray
        nv x 2 list of coordinates

    E : ndarray
        ne x 3 list of vertices

    Returns
    -------
    h : float
        maximum diameter of a circumcircle over all elements
        longest edge

    Examples
    --------
    >>> import numpy as np
    >>> dx = 1
    >>> V = np.array([[0,0], [dx,0], [0,dx], [dx,dx]])
    >>> E = np.array([[0,1,2], [2,3,1]])
    >>> h = diameter(V, E)
    >>> print(h)
    1.4142135623730951

    """
    if not isinstance(V, np.ndarray) or not isinstance(E, np.ndarray):
        raise ValueError('V and E must be ndarray')

    if V.shape[1] != 2 or E.shape[1] != 3:
        raise ValueError('V should be nv x 2 and E should be ne x 3')

    h = 0
    I = [0, 1, 2, 0]
    for e in E:
        hs = np.sqrt(np.diff(V[e[I], 0])**2 + np.diff(V[e[I], 1])**2)
        h = max(h, hs.max())
    return h


def refine2dtri(V, E, marked_elements=None):
    r"""Refine a triangular mesh

    Parameters
    ----------
    V : ndarray
        nv x 2 list of coordinates

    E : ndarray
        ne x 3 list of vertices

    marked_elements : array
        list of marked elements for refinement.  None means uniform.

    Returns
    -------
    Vref : ndarray
        nv x 2 list of coordinates

    Eref : ndarray
        ne x 3 list of vertices

    Notes
    -----
        - Peforms quad-section in the following where n0, n1, and n2 are
          the original vertices

                   n2
                  / |
                /   |
              /     |
           n5-------n4
          / \      /|
        /    \    / |
      /       \  /  |
    n0 --------n3-- n1
    """
    Nel = E.shape[0]
    Nv = V.shape[0]

    if marked_elements is None:
        marked_elements = np.arange(0, Nel)

    marked_elements = np.ravel(marked_elements)

    # construct vertex to vertex graph
    col = E.ravel()
    row = np.kron(np.arange(0, Nel), [1, 1, 1])
    data = np.ones((Nel*3,))
    V2V = sparse.coo_matrix((data, (row, col)), shape=(Nel, Nv))
    V2V = V2V.T * V2V

    # compute interior edges list
    V2V.data = np.ones(V2V.data.shape)
    V2Vupper = sparse.triu(V2V, 1).tocoo()

    # construct EdgeList from V2V
    Nedges = len(V2Vupper.data)
    V2Vupper.data = np.arange(0, Nedges)
    EdgeList = np.vstack((V2Vupper.row, V2Vupper.col)).T
    Nedges = EdgeList.shape[0]

    # elements to edge list
    V2Vupper = V2Vupper.tocsr()
    edges = np.vstack((E[:, [0, 1]],
                       E[:, [1, 2]],
                       E[:, [2, 0]]))
    edges.sort(axis=1)
    ElementToEdge = V2Vupper[edges[:, 0], edges[:, 1]].reshape((3, Nel)).T

    marked_edges = np.zeros((Nedges,), dtype=bool)
    marked_edges[ElementToEdge[marked_elements, :].ravel()] = True

    # mark 3-2-1 triangles
    nsplit = len(np.where(marked_edges == 1)[0])
    edge_num = marked_edges[ElementToEdge].sum(axis=1)
    edges3 = np.where(edge_num >= 2)[0]
    marked_edges[ElementToEdge[edges3, :]] = True  # marked 3rd edge
    nsplit = len(np.where(marked_edges == 1)[0])

    edges1 = np.where(edge_num == 1)[0]
    # edges1 = edge_num[id]             # all 2 or 3 edge elements

    # new nodes (only edges3 elements)

    x_new = 0.5*(V[EdgeList[marked_edges, 0], 0]) \
        + 0.5*(V[EdgeList[marked_edges, 1], 0])
    y_new = 0.5*(V[EdgeList[marked_edges, 0], 1]) \
        + 0.5*(V[EdgeList[marked_edges, 1], 1])

    V_new = np.vstack((x_new, y_new)).T
    V = np.vstack((V, V_new))
    # indices of the new nodes
    new_id = np.zeros((Nedges,), dtype=int)
    # print(len(np.where(marked_edges == 1)[0]))
    # print(nsplit)
    new_id[marked_edges] = Nv + np.arange(0, nsplit)
    # New tri's in the case of refining 3 edges
    # example, 1 element
    #                n2
    #               / |
    #             /   |
    #           /     |
    #        n5-------n4
    #       / \      /|
    #     /    \    / |
    #   /       \  /  |
    # n0 --------n3-- n1
    ids = np.ones((Nel,), dtype=bool)
    ids[edges3] = False
    ids[edges1] = False

    E_new = np.delete(E, marked_elements, axis=0)  # E[id2, :]
    n0 = E[edges3, 0]
    n1 = E[edges3, 1]
    n2 = E[edges3, 2]
    n3 = new_id[ElementToEdge[edges3, 0]].ravel()
    n4 = new_id[ElementToEdge[edges3, 1]].ravel()
    n5 = new_id[ElementToEdge[edges3, 2]].ravel()

    t1 = np.vstack((n0, n3, n5)).T
    t2 = np.vstack((n3, n1, n4)).T
    t3 = np.vstack((n4, n2, n5)).T
    t4 = np.vstack((n3, n4, n5)).T

    E_new = np.vstack((E_new, t1, t2, t3, t4))
    return V, E_new


def l2norm(u, mesh):
    """Calculate the L2 norm of a funciton on mesh (V,E)

    Parameters
    ----------
    u : array
        (nv,) list of function values

    mesh : object
        mesh object

    Returns
    -------
    val : float
        the value of the L2 norm of u, ||u||_2,V

    Notes
    -----
        - modepy is used to generate the quadrature points
          q = modepy.XiaoGimbutasSimplexQuadrature(4,2)

    Examples
    --------
    >>> import numpy as np
    >>> V = np.array([[0,0], [1,0], [0,1], [1,1]])
    >>> E = np.array([[0,1,2], [2,3,1]])
    >>> X, Y = V[:, 0], V[:, 1]
    >>> import fem
    >>> I = fem.l2norm(X+Y, V, E, degree=1)
    >>> print(I)
    >>> V2, E2 = fem.generate_quadratic(V, E)
    >>> X, Y = V2[:, 0], V2[:, 1]
    >>> I = fem.l2norm(X+Y, V2, E2, degree=2)
    >>> print(I)
    >>> # actual (from sympy): 1.08012344973464
    """
    if mesh.degree == 1:
        V = mesh.V
        E = mesh.E

    if mesh.degree == 2:
        V = mesh.V2
        E = mesh.E2

    if not isinstance(u, np.ndarray):
        raise ValueError('u must be ndarray')

    if V.shape[1] != 2:
        raise ValueError('V should be nv x 2')

    if mesh.degree == 1 and E.shape[1] != 3:
        raise ValueError('E should be nv x 3')

    if mesh.degree == 2 and E.shape[1] != 6:
        raise ValueError('E should be nv x 6')

    if mesh.degree not in [1, 2]:
        raise ValueError('degree = 1 or 2 supported')

    val = 0

    # quadrature points
    ww = np.array([0.44676317935602256, 0.44676317935602256, 0.44676317935602256,
                   0.21990348731064327, 0.21990348731064327, 0.21990348731064327])
    xy = np.array([[-0.10810301816807008, -0.78379396366385990],
                   [-0.10810301816806966, -0.10810301816807061],
                   [-0.78379396366386020, -0.10810301816806944],
                   [-0.81684757298045740, -0.81684757298045920],
                   [0.63369514596091700, -0.81684757298045810],
                   [-0.81684757298045870, 0.63369514596091750]])
    xx, yy = (xy[:, 0]+1)/2, (xy[:, 1]+1)/2
    ww *= 0.5

    if mesh.degree == 1:
        I = np.arange(3)

        def basis1(x, y):
            return np.array([1-x-y,
                             x,
                             y])
        basis = basis1

    if mesh.degree == 2:
        I = np.arange(6)

        def basis2(x, y):
            return np.array([(1-x-y)*(1-2*x-2*y),
                             x*(2*x-1),
                             y*(2*y-1),
                             4*x*(1-x-y),
                             4*x*y,
                             4*y*(1-x-y)])
        basis = basis2

    for e in E:
        x = V[e, 0]
        y = V[e, 1]

        # Jacobian
        jac = np.abs((x[1]-x[0])*(y[2]-y[0]) - (x[2]-x[0])*(y[1]-y[0]))

        # add up each quadrature point
        for wv, xv, yv in zip(ww, xx, yy):
            val += (jac / 2) * wv * np.dot(u[e[I]], basis(xv, yv))**2

    # take the square root for the norm
    return np.sqrt(val)


class Mesh:
    """Simple mesh object that holds vertices and mesh functions
    """

    # pylint: disable=too-many-instance-attributes
    # This is reasonble for this class

    def __init__(self, V, E, degree=1):

        # check to see if E is numbered 0 ... nv
        ids = np.full((E.max()+1,), False)
        ids[E.ravel()] = True
        nv = np.sum(ids)
        if V.shape[0] != nv:
            print('fixing V and E')
            I = np.where(ids)[0]
            J = np.arange(E.max()+1)
            J[I] = np.arange(nv)
            E = J[E]
            V = V[I, :]

        if not check_mesh(V, E):
            raise ValueError('triangles must be counter clockwise')

        self.V = V
        self.E = E
        self.X = V[:, 0]
        self.Y = V[:, 1]
        self.degree = degree

        self.nv = nv
        self.ne = E.shape[0]

        self.h = diameter(V, E)

        self.V2 = None
        self.E2 = None
        self.Edges = None
        self.newID = None

        if degree == 2:
            self.generate_quadratic()

    def generate_quadratic(self):
        """generate a quadratic mesh
        """
        if self.V2 is None:
            self.V2, self.E2, self.Edges = generate_quadratic(self.V, self.E,
                                                              return_edges=True)
            self.X2 = self.V2[:, 0]
            self.Y2 = self.V2[:, 1]
            self.newID = self.nv + np.arange(self.Edges.shape[0])

    def refine(self, levels):
        """refine the mesh
        """
        self.V2 = None
        self.E2 = None
        self.Edges = None
        self.newID = None
        for _ in range(levels):
            self.V, self.E = refine2dtri(self.V, self.E)
        self.nv = self.V.shape[0]
        self.ne = self.E.shape[0]
        self.h = diameter(self.V, self.E)
        self.X = self.V[:, 0]
        self.Y = self.V[:, 1]

        if self.degree == 2:
            self.generate_quadratic()

    def smooth(self, maxit=10, tol=0.01):
        """Constrained Laplacian Smoothing.

        Parameters
        ----------
        maxit : int
            Iterations
        tol : float
            Convergence toleratnce measured in the maximum
            absolute distance the mesh moves (in one iteration).

        """
        nv = self.nv

        # graph Laplacian (only the adjacency)
        edge0 = self.E[:, [0, 0, 1, 1, 2, 2]].ravel()
        edge1 = self.E[:, [1, 2, 0, 2, 0, 1]].ravel()
        data = np.ones((edge0.shape[0],), dtype=int)
        G = sparse.coo_matrix((data, (edge0, edge1)), shape=(nv, nv))
        G.sum_duplicates()
        G.eliminate_zeros()

        # boundary IDs
        bid = np.where(G.data == 1)[0]
        bid = np.unique(G.row[bid])

        # set constant (alternative: edgelength)
        G.data[:] = 1
        W = np.array(G.sum(axis=1)).flatten()

        Vnew = self.V.copy()
        edgelength = (Vnew[edge0, 0] - Vnew[edge1, 0])**2 +\
                     (Vnew[edge0, 1] - Vnew[edge1, 1])**2

        maxit = 100
        for _it in range(maxit):
            Vnew = G @ Vnew
            Vnew /= W[:, None]  # scale the columns by 1/W
            Vnew[bid, :] = self.V[bid, :]
            newedgelength = np.sqrt((Vnew[edge0, 0] - Vnew[edge1, 0])**2
                                    + (Vnew[edge0, 1] - Vnew[edge1, 1])**2)
            move = np.max(np.abs(newedgelength - edgelength) / newedgelength)
            edgelength = newedgelength
            if move < tol:
                break

        self.V = Vnew
        return _it


def gradgradform(mesh, kappa=None, f=None, degree=1):
    """Finite element discretization of a Poisson problem.

    - div . kappa(x,y) grad u = f(x,y)

    Parameters
    ----------
    V : ndarray
        nv x 2 list of coordinates

    E : ndarray
        ne x 3 or 6 list of vertices

    kappa : function
        diffusion coefficient, kappa(x,y) with vector input

    fa : function
        right hand side, f(x,y) with vector input

    degree : 1 or 2
        polynomial degree of the bases (assumed to be Lagrange locally)

    Returns
    -------
    A : sparse matrix
        finite element matrix where A_ij = <kappa grad phi_i, grad phi_j>

    b : array
        finite element rhs where b_ij = <f, phi_j>

    Notes
    -----
        - modepy is used to generate the quadrature points
          q = modepy.XiaoGimbutasSimplexQuadrature(4,2)

    Example
    -------
    >>> import numpy as np
    >>> import fem
    >>> import scipy.sparse.linalg as sla
    >>> V = np.array(
        [[  0,  0],
         [  1,  0],
         [2*1,  0],
         [  0,  1],
         [  1,  1],
         [2*1,  1],
         [  0,2*1],
         [  1,2*1],
         [2*1,2*1],
        ])
    >>> E = np.array(
        [[0,1,3],
         [1,2,4],
         [1,4,3],
         [2,5,4],
         [3,4,6],
         [4,5,7],
         [4,7,6],
         [5,8,7]])
    >>> A, b = fem.poissonfem(V, E)
    >>> print(A.toarray())
    >>> print(b)
    >>> f = lambda x, y : 0*x + 1.0
    >>> g = lambda x, y : 0*x + 0.0
    >>> g1 = lambda x, y : 0*x + 1.0
    >>> tol = 1e-12
    >>> X, Y = V[:,0], V[:,1]
    >>> id1 = np.where(abs(Y) < tol)[0]
    >>> id2 = np.where(abs(Y-2) < tol)[0]
    >>> id3 = np.where(abs(X) < tol)[0]
    >>> id4 = np.where(abs(X-2) < tol)[0]
    >>> bc = [{'id': id1, 'g': g},
              {'id': id2, 'g': g},
              {'id': id3, 'g': g1},
              {'id': id4, 'g': g}]
    >>> A, b = fem.poissonfem(V, E, f=f, bc=bc)
    >>> u = sla.spsolve(A, b)
    >>> print(A.toarray())
    >>> print(b)
    >>> print(u)
    """
    if degree not in [1, 2]:
        raise ValueError('degree = 1 or 2 supported')

    if f is None:
        def f(_x, _y):
            return 0.0

    if kappa is None:
        def kappa(_x, _y):
            return 1.0

    if not callable(f) or not callable(kappa):
        raise ValueError('f, kappa must be callable functions')

    ne = mesh.ne

    if degree == 1:
        E = mesh.E
        X = mesh.X
        Y = mesh.Y

    if degree == 2:
        E = mesh.E2
        X = mesh.X2
        Y = mesh.Y2

    # allocate sparse matrix arrays
    m = 3 if degree == 1 else 6
    AA = np.zeros((ne, m**2))
    IA = np.zeros((ne, m**2), dtype=int)
    JA = np.zeros((ne, m**2), dtype=int)
    bb = np.zeros((ne, m))
    ib = np.zeros((ne, m), dtype=int)
    jb = np.zeros((ne, m), dtype=int)

    # Assemble A and b
    for ei in range(0, ne):
        # Step 1: set the vertices and indices
        K = E[ei, :]
        x0, y0 = X[K[0]], Y[K[0]]
        x1, y1 = X[K[1]], Y[K[1]]
        x2, y2 = X[K[2]], Y[K[2]]

        # Step 2: compute the Jacobian, inv, and det
        J = np.array([[x1 - x0, x2 - x0],
                      [y1 - y0, y2 - y0]])
        invJ = np.linalg.inv(J.T)
        detJ = np.linalg.det(J)

        if degree == 1:
            # Step 3, define the gradient of the basis
            dbasis = np.array([[-1, 1, 0],
                               [-1, 0, 1]])

            # Step 4
            dphi = invJ.dot(dbasis)

            # Step 5, 1-point gauss quadrature
            Aelem = kappa(X[K].mean(), Y[K].mean()) * (detJ / 2.0) * (dphi.T).dot(dphi)

            # Step 6, 1-point gauss quadrature
            belem = f(X[K].mean(), Y[K].mean()) * (detJ / 6.0) * np.ones((3,))

        if degree == 2:
            ww = np.array([0.44676317935602256, 0.44676317935602256, 0.44676317935602256,
                           0.21990348731064327, 0.21990348731064327, 0.21990348731064327])
            xy = np.array([[-0.10810301816807008, -0.78379396366385990],
                           [-0.10810301816806966, -0.10810301816807061],
                           [-0.78379396366386020, -0.10810301816806944],
                           [-0.81684757298045740, -0.81684757298045920],
                           [0.63369514596091700, -0.81684757298045810],
                           [-0.81684757298045870, 0.63369514596091750]])
            xx, yy = (xy[:, 0]+1)/2, (xy[:, 1]+1)/2
            ww *= 0.5

            Aelem = np.zeros((m, m))
            belem = np.zeros((m,))

            for w, x, y in zip(ww, xx, yy):
                # Step 3
                basis = np.array([(1-x-y)*(1-2*x-2*y),
                                  x*(2*x-1),
                                  y*(2*y-1),
                                  4*x*(1-x-y),
                                  4*x*y,
                                  4*y*(1-x-y)])

                dbasis = np.array([
                    [4*x + 4*y - 3, 4*x-1,     0, -8*x - 4*y + 4, 4*y,           -4*y],
                    [4*x + 4*y - 3,     0, 4*y-1,           -4*x, 4*x, -4*x - 8*y + 4]
                ])

                # Step 4
                dphi = invJ.dot(dbasis)

                # Step 5
                xt, yt = J.dot(np.array([x, y])) + np.array([x0, y0])
                Aelem += (detJ / 2) * w * kappa(xt, yt) * dphi.T.dot(dphi)

                # Step 6
                belem += (detJ / 2) * w * f(xt, yt) * basis

        # Step 7
        AA[ei, :] = Aelem.ravel()
        IA[ei, :] = np.repeat(K[np.arange(m)], m)
        JA[ei, :] = np.tile(K[np.arange(m)], m)
        bb[ei, :] = belem.ravel()
        ib[ei, :] = K[np.arange(m)]
        jb[ei, :] = 0

    # convert matrices
    A = sparse.coo_matrix((AA.ravel(), (IA.ravel(), JA.ravel())))
    A.sum_duplicates()
    b = sparse.coo_matrix((bb.ravel(), (ib.ravel(), jb.ravel()))).toarray().ravel()

    # A = A.tocsr()
    return A, b


def divform(mesh):
    """Calculate the (div u , p) form that arises in Stokes
       assumes P2-P1 elements
    """
    if mesh.V2 is None:
        mesh.generate_quadratic()

    X, Y = mesh.X, mesh.Y
    ne = mesh.ne
    E = mesh.E2

    m1 = 6
    m2 = 3
    DX = np.zeros((ne, m1*m2))
    DXI = np.zeros((ne, m1*m2), dtype=int)
    DXJ = np.zeros((ne, m1*m2), dtype=int)
    DY = np.zeros((ne, m1*m2))
    DYI = np.zeros((ne, m1*m2), dtype=int)
    DYJ = np.zeros((ne, m1*m2), dtype=int)

    # Assemble A and b
    for ei in range(0, ne):
        K = E[ei, :]
        x0, y0 = X[K[0]], Y[K[0]]
        x1, y1 = X[K[1]], Y[K[1]]
        x2, y2 = X[K[2]], Y[K[2]]

        J = np.array([[x1 - x0, x2 - x0],
                      [y1 - y0, y2 - y0]])
        invJ = np.linalg.inv(J.T)
        detJ = np.linalg.det(J)

        ww = np.array([0.44676317935602256, 0.44676317935602256, 0.44676317935602256,
                       0.21990348731064327, 0.21990348731064327, 0.21990348731064327])
        xy = np.array([[-0.10810301816807008, -0.78379396366385990],
                       [-0.10810301816806966, -0.10810301816807061],
                       [-0.78379396366386020, -0.10810301816806944],
                       [-0.81684757298045740, -0.81684757298045920],
                       [ 0.63369514596091700, -0.81684757298045810],   # noqa: E201
                       [-0.81684757298045870,  0.63369514596091750]])
        xx, yy = (xy[:, 0]+1)/2, (xy[:, 1]+1)/2
        ww *= 0.5

        DXelem = np.zeros((3, 6))
        DYelem = np.zeros((3, 6))

        for w, x, y in zip(ww, xx, yy):
            basis1 = np.array([1-x-y, x, y])

            # basis2 = np.array([(1-x-y)*(1-2*x-2*y),
            #                   x*(2*x-1),
            #                   y*(2*y-1),
            #                   4*x*(1-x-y),
            #                   4*x*y,
            #                   4*y*(1-x-y)])

            dbasis = np.array([
                [4*x + 4*y - 3, 4*x-1,     0, -8*x - 4*y + 4, 4*y,           -4*y],
                [4*x + 4*y - 3,     0, 4*y-1,           -4*x, 4*x, -4*x - 8*y + 4]
            ])

            dphi = invJ.dot(dbasis)

            DXelem += (detJ / 2) * w * (np.outer(basis1, dphi[0, :]))
            DYelem += (detJ / 2) * w * (np.outer(basis1, dphi[1, :]))
            dphi.T.dot(dphi)

        # Step 7
        DX[ei, :] = DXelem.ravel()
        DXI[ei, :] = np.repeat(K[np.arange(m2)], m1)
        DXJ[ei, :] = np.tile(K[np.arange(m1)], m2)
        BX = sparse.coo_matrix((DX.ravel(), (DXI.ravel(), DXJ.ravel())))
        BX.sum_duplicates()

        DY[ei, :] = DYelem.ravel()
        DYI[ei, :] = np.repeat(K[np.arange(m2)], m1)
        DYJ[ei, :] = np.tile(K[np.arange(m1)], m2)
        BY = sparse.coo_matrix((DY.ravel(), (DYI.ravel(), DYJ.ravel())))
        BY.sum_duplicates()
    return BX, BY


def applybc(A, b, mesh, bc):
    """
    bc : list
       list of boundary conditions
       bc = [bc1, bc2, ..., bck]
       where bck = {'id': id,    a list of vertices for boundary "k"
                     'g': g,     g = g(x,y) is a function for the vertices on boundary "k"
                   'var': var    the variable, given as a start in the dof list
                'degree': degree degree of the variable, either 1 or 2
                   }
    """

    for c in bc:
        if not callable(c['g']):
            raise ValueError('each bc g must be callable functions')

        if 'degree' not in c.keys():
            c['degree'] = 1

        if 'var' not in c.keys():
            c['var'] = 0

    # now extend the BC
    # for each new id, are the orignal neighboring ids in a bc?
    for c in bc:
        if c['degree'] == 2:
            idx = c['id']
            newidx = []
            for j, ed in zip(mesh.newID, mesh.Edges):
                if ed[0] in idx and ed[1] in idx:
                    newidx.append(j)
            c['id'] = np.hstack((idx, newidx))

    # set BC in the right hand side
    # set the lifting function (1 of 3)
    u0 = np.zeros((A.shape[0],))
    for c in bc:
        idx = c['var'] + c['id']
        if c['degree'] == 1:
            X = mesh.X
            Y = mesh.Y
        elif c['degree'] == 2:
            X = mesh.X2
            Y = mesh.Y2
        u0[idx] = c['g'](X[idx], Y[idx])

    # lift (2 of 3)
    b = b - A * u0

    # fix the values (3 of 3)
    for c in bc:
        idx = c['var'] + c['id']
        b[idx] = u0[idx]

    # set BC to identity in the matrix
    # collect all BC indices (1 of 2)
    Dflag = np.full((A.shape[0],), False)
    for c in bc:
        idx = c['var'] + c['id']
        Dflag[idx] = True
    # write identity (2 of 2)
    for k, (i, j) in enumerate(zip(A.row, A.col)):
        if Dflag[i] or Dflag[j]:
            if i == j:
                A.data[k] = 1.0
            else:
                A.data[k] = 0.0

    return A, b


def stokes(mesh, fu, fv):
    """Stokes Flow
    """
    mesh.generate_quadratic()
    Au, bu = gradgradform(mesh, f=fu, degree=2)
    Av, bv = gradgradform(mesh, f=fv, degree=2)
    BX, BY = divform(mesh)

    C = sparse.bmat([[Au, None, BX.T],
                     [None, Av, BY.T],
                     [BX, BY, None]])
    b = np.hstack((bu, bv, np.zeros((BX.shape[0],))))

    return C, b


def model(num=0):
    """A list of model (elliptic) problems

    Parameters
    ----------
    num : int or string
        A tag for a particular problem.  See the notes below.

    Return
    ------
    A
    b
    V
    E
    f
    kappa
    bc

    See Also
    --------
    poissonfem - build the FE matrix and right hand side
    Notes
    -----
    """
    print(num)
    raise NotImplementedError('model is unimplemented')
