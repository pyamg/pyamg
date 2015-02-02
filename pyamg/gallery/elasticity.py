"""Constructs linear elasticity problems for first-order elements in 2D and 3D
"""

__docformat__ = "restructuredtext en"

__all__ = ['linear_elasticity', 'linear_elasticity_p1']

from scipy import array, matrix, ones, zeros, arange, empty, \
    hstack, vstack, tile, ravel, mgrid, concatenate, \
    cumsum, dot, eye, asarray
from scipy.linalg import inv, det
from scipy.sparse import coo_matrix, bsr_matrix


def linear_elasticity(grid, spacing=None, E=1e5, nu=0.3, format=None):
    """Linear elasticity problem discretizes with Q1 finite elements
    on a regular rectangular grid

    Parameters
    ----------
    grid : tuple
        length 2 tuple of grid sizes, e.g. (10, 10)
    spacing : tuple
        length 2 tuple of grid spacings, e.g. (1.0, 0.1)
    E : float
        Young's modulus
    nu : float
        Poisson's ratio
    format : string
        Format of the returned sparse matrix (eg. 'csr', 'bsr', etc.)

    Returns
    -------
    A : {csr_matrix}
        FE Q1 stiffness matrix

    See Also
    --------
    linear_elasticity_p1

    Notes
    -----
        - only 2d for now

    Examples
    --------
    >>> from pyamg.gallery import linear_elasticity
    >>> A, B = linear_elasticity((4, 4))

    References
    ----------
    .. [1] J. Alberty, C. Carstensen, S. A. Funken, and R. KloseDOI
       "Matlab implementation of the finite element method in elasticity"
       Computing, Volume 69,  Issue 3  (November 2002) Pages: 239 - 263
       http://www.math.hu-berlin.de/~cc/

    """
    if len(grid) == 2:
        return q12d(grid, spacing=spacing, E=E, nu=nu, format=format)
    else:
        raise NotImplemented('no support for grid=%s' % str(grid))


def q12d(grid, spacing=None, E=1e5, nu=0.3, dirichlet_boundary=True,
         format=None):
    """Q1 elements in 2 dimensions

    See Also
    --------
    linear_elasticity
    """
    X, Y = tuple(grid)

    if X < 1 or Y < 1:
        raise ValueError('invalid grid shape')

    if dirichlet_boundary:
        X += 1
        Y += 1

    pts = mgrid[0:X+1, 0:Y+1]
    pts = hstack((pts[0].T.reshape(-1, 1) - X / 2.0,
                  pts[1].T.reshape(-1, 1) - Y / 2.0))

    if spacing is None:
        DX, DY = 1, 1
    else:
        DX, DY = tuple(spacing)
        pts = [DX, DY]

    # compute local stiffness matrix
    lame = E * nu / ((1 + nu) * (1 - 2*nu))  # Lame's first parameter
    mu = E / (2 + 2*nu)                   # shear modulus

    vertices = array([[0, 0], [DX, 0], [DX, DY], [0, DY]])
    K = q12d_local(vertices, lame, mu)

    nodes = arange((X+1)*(Y+1)).reshape(X+1, Y+1)
    LL = nodes[:-1, :-1]
    I = (2*LL).repeat(K.size).reshape(-1, 8, 8)
    J = I.copy()
    I += tile([0, 1, 2, 3, 2*X + 4, 2*X + 5, 2*X + 2, 2*X + 3], (8, 1))
    J += tile([0, 1, 2, 3, 2*X + 4, 2*X + 5, 2*X + 2, 2*X + 3], (8, 1)).T
    V = tile(K, (X*Y, 1))

    I = ravel(I)
    J = ravel(J)
    V = ravel(V)

    # sum duplicates
    A = coo_matrix((V, (I, J)), shape=(pts.size, pts.size)).tocsr()
    A = A.tobsr(blocksize=(2, 2))

    del I, J, V, LL, nodes

    B = zeros((2 * (X+1)*(Y+1), 3))
    B[0::2, 0] = 1
    B[1::2, 1] = 1
    B[0::2, 2] = -pts[:, 1]
    B[1::2, 2] = pts[:, 0]

    if dirichlet_boundary:
        mask = zeros((X+1, Y+1), dtype='bool')
        mask[1:-1, 1:-1] = True
        mask = ravel(mask)
        data = zeros(((X-1)*(Y-1), 2, 2))
        data[:, 0, 0] = 1
        data[:, 1, 1] = 1
        indices = arange((X-1)*(Y-1))
        indptr = concatenate((array([0]), cumsum(mask)))
        P = bsr_matrix((data, indices, indptr),
                       shape=(2*(X+1)*(Y+1), 2*(X-1)*(Y-1)))
        Pt = P.T
        A = P.T * A * P

        B = Pt * B

    return A.asformat(format), B


def q12d_local(vertices, lame, mu):
    """local stiffness matrix for two dimensional elasticity
       on a square element

    Parameters
    ----------
    lame : Float
        Lame's first parameter
    mu : Float
        shear modulus

    See Also
    --------
    linear_elasticity

    Notes
    -----
    Vertices should be listed in counter-clockwise order::

        [3]----[2]
         |      |
         |      |
        [0]----[1]

    Degrees of freedom are enumerated as follows::

        [x=6,y=7]----[x=4,y=5]
            |            |
            |            |
        [x=0,y=1]----[x=2,y=3]

    """

    M = lame + 2*mu  # P-wave modulus

    R_11 = matrix([[2, -2, -1,  1],
                   [-2,  2,  1, -1],
                   [-1,  1,  2, -2],
                   [1, -1, -2,  2]]) / 6.0

    R_12 = matrix([[1,  1, -1, -1],
                   [-1, -1,  1,  1],
                   [-1, -1,  1,  1],
                   [1,  1, -1, -1]]) / 4.0

    R_22 = matrix([[2,  1, -1, -2],
                   [1,  2, -2, -1],
                   [-1, -2,  2,  1],
                   [-2, -1,  1,  2]]) / 6.0

    F = inv(vstack((vertices[1] - vertices[0], vertices[3] - vertices[0])))

    K = zeros((8, 8))  # stiffness matrix

    E = F.T * matrix([[M, 0], [0, mu]]) * F
    K[0::2, 0::2] = E[0, 0] * R_11 + E[0, 1] * R_12 +\
        E[1, 0] * R_12.T + E[1, 1] * R_22

    E = F.T * matrix([[mu, 0], [0, M]]) * F
    K[1::2, 1::2] = E[0, 0] * R_11 + E[0, 1] * R_12 +\
        E[1, 0] * R_12.T + E[1, 1] * R_22

    E = F.T * matrix([[0, mu], [lame, 0]]) * F
    K[1::2, 0::2] = E[0, 0] * R_11 + E[0, 1] * R_12 +\
        E[1, 0] * R_12.T + E[1, 1] * R_22

    K[0::2, 1::2] = K[1::2, 0::2].T

    K /= det(F)

    return K


def linear_elasticity_p1(vertices, elements, E=1e5, nu=0.3, format=None):
    """P1 elements in 2 or 3 dimensions

    Parameters
    ----------
    vertices : array_like
        array of vertices of a triangle or tets
    elements : array_like
        array of vertex indices for tri or tet elements
    E : float
        Young's modulus
    nu : float
        Poisson's ratio
    format : string
        'csr', 'csc', 'coo', 'bsr'

    Returns
    -------
    A : {csr_matrix}
        FE Q1 stiffness matrix

    Notes
    -----
        - works in both 2d and in 3d

    Examples
    --------
    >>> from pyamg.gallery import linear_elasticity_p1
    >>> from numpy import array
    >>> E = array([[0, 1, 2],[1, 3, 2]])
    >>> V = array([[0.0, 0.0],[1.0, 0.0],[0.0, 1.0],[1.0, 1.0]])
    >>> A, B = linear_elasticity_p1(V, E)

    References
    ----------
    .. [1] J. Alberty, C. Carstensen, S. A. Funken, and R. KloseDOI
       "Matlab implementation of the finite element method in elasticity"
       Computing, Volume 69,  Issue 3  (November 2002) Pages: 239 - 263
       http://www.math.hu-berlin.de/~cc/

    """

    # compute local stiffness matrix
    lame = E * nu / ((1 + nu) * (1 - 2*nu))  # Lame's first parameter
    mu = E / (2 + 2*nu)                   # shear modulus

    vertices = asarray(vertices)
    elements = asarray(elements)

    D = vertices.shape[1]    # spatial dimension
    DoF = D*vertices.shape[0]  # number of degrees of freedom
    NE = elements.shape[0]    # number of elements

    if elements.shape[1] != D + 1:
        raise ValueError('dimension mismatch')

    if D == 2:
        local_K = p12d_local
    elif D == 3:
        local_K = p13d_local
    else:
        raise NotImplementedError('only dimension 2 and 3 are supported')

    row = elements.repeat(D).reshape(-1, D)
    row *= D
    row += arange(D)
    row = row.reshape(-1, D*(D+1)).repeat(D*(D+1), axis=0)
    row = row.reshape(-1, D*(D+1), D*(D+1))
    col = row.swapaxes(1, 2)

    data = empty((NE, D*(D+1), D*(D+1)), dtype=float)

    for i in xrange(NE):
        element_indices = elements[i, :]
        element_vertices = vertices[element_indices, :]

        data[i] = local_K(element_vertices, lame, mu)

    row = row.ravel()
    col = col.ravel()
    data = data.ravel()

    # sum duplicates
    A = coo_matrix((data, (row, col)), shape=(DoF, DoF)).tocsr()
    A = A.tobsr(blocksize=(D, D))

    # compute rigid body modes
    if D == 2:
        B = zeros((DoF, 3))
        B[0::2, 0] = 1              # vector field in x direction
        B[1::2, 1] = 1              # vector field in y direction

        B[0::2, 2] = -vertices[:, 1]  # rotation vector field (-y, x)
        B[1::2, 2] = vertices[:, 0]
    else:
        B = zeros((DoF, 6))
        B[0::3, 0] = 1              # vector field in x direction
        B[1::3, 1] = 1              # vector field in y direction
        B[2::3, 2] = 1              # vector field in z direction

        B[0::3, 3] = -vertices[:, 1]  # rotation vector field (-y, x, 0)
        B[1::3, 3] = vertices[:, 0]
        B[0::3, 4] = -vertices[:, 2]  # rotation vector field (-z, 0, x)
        B[2::3, 4] = vertices[:, 0]
        B[1::3, 5] = -vertices[:, 2]  # rotation vector field (0,-z, y)
        B[2::3, 5] = vertices[:, 1]

    return A.asformat(format), B


def p12d_local(vertices, lame, mu):
    """local stiffness matrix for P1 elements in 2d"""
    assert(vertices.shape == (3, 2))

    A = vstack((ones((1, 3)), vertices.T))
    PhiGrad = inv(A)[:, 1:]  # gradients of basis functions
    R = zeros((3, 6))
    R[[[0], [2]], [0, 2, 4]] = PhiGrad.T
    R[[[2], [1]], [1, 3, 5]] = PhiGrad.T
    C = mu*array([[2, 0, 0], [0, 2, 0], [0, 0, 1]]) +\
        lame*array([[1, 1, 0], [1, 1, 0], [0, 0, 0]])
    K = det(A)/2.0*dot(dot(R.T, C), R)
    return K


def p13d_local(vertices, lame, mu):
    """local stiffness matrix for P1 elements in 3d"""
    assert(vertices.shape == (4, 3))

    A = vstack((ones((1, 4)), vertices.T))
    PhiGrad = inv(A)[:, 1:]  # gradients of basis functions

    R = zeros((6, 12))
    R[[0, 3, 4], 0::3] = PhiGrad.T
    R[[3, 1, 5], 1::3] = PhiGrad.T
    R[[4, 5, 2], 2::3] = PhiGrad.T

    C = zeros((6, 6))
    C[0:3, 0:3] = lame + 2*mu*eye(3)
    C[3:6, 3:6] = mu*eye(3)

    K = det(A)/6*dot(dot(R.T, C), R)
    return K
