"""Testing for fem.py
"""
import numpy as np
from pyamg.gallery import fem
import scipy.sparse.linalg as sla

import os

test_dir = os.path.split(__file__)[0]
base_dir = os.path.split(test_dir)[0]
mesh_dir = os.path.join(base_dir, 'mesh_data')


class TestDiameter(np.testing.TestCase):
    """
    Testing for diameter """

    def test_diameter(self):
        """
        Test the longest edge for a two triangle mesh
        """
        h = 1.0
        for _ in range(5):

            V = np.array(
                [[0, 0],
                 [h, 0],
                 [0, h],
                 [h, h]])
            E = np.array(
                [[0, 1, 2],
                 [2, 3, 1]])

            np.testing.assert_almost_equal(fem.diameter(V, E), np.sqrt(h**2 + h**2))

            h = h / 2


class TestQuadratic(np.testing.TestCase):
    """
    Testing for generate_quadratic
    """

    def test_quadratic(self):
        V = np.array(
            [[0., 0.],
             [1., 0.],
             [0., 1.],
             [1., 1.]])
        E = np.array(
            [[0, 1, 2],
             [2, 3, 1]])

        V2 = np.array([[0., 0.],
                       [1., 0.],
                       [0., 1.],
                       [1., 1.],
                       [0.5, 0.],
                       [0.5, 0.5],
                       [0., 0.5],
                       [0.5, 1.],
                       [1., 0.5]])
        E2 = np.array([[0, 1, 2, 4, 5, 6],
                       [2, 3, 1, 7, 8, 5]])

        V2gen, E2gen = fem.generate_quadratic(V, E)

        np.testing.assert_almost_equal(V2gen, V2)
        np.testing.assert_almost_equal(E2gen, E2)


class TestL2Norm(np.testing.TestCase):
    """
    Testing for l2norm

    Notes:
        - testing formed with sympy
          from sympy import *
          x, y = symbols("x y")
    """
    def test_l2norm(self):
        data = np.load(os.path.join(mesh_dir, 'square_mesh.npz'))
        # import square mesh of vertices, elements
        V = data['vertices']
        E = data['elements']
        print(V, E)
        mesh = fem.Mesh(V, E)
        X, Y = V[:, 0], V[:, 1]

        # 2 = sqrt( integrate(x + 1, (x,-1,1), (y,-1,1))).evalf()
        np.testing.assert_almost_equal(fem.l2norm(np.sqrt(X+1), mesh), 2, decimal=2)

        # 2 = sqrt( integrate(x*y + 1, (x,-1,1), (y,-1,1))).evalf()
        np.testing.assert_almost_equal(fem.l2norm(np.sqrt(X*Y+1), mesh), 2, decimal=2)

        # 0.545351286587159 =
        # sqrt( integrate(sin(x)*sin(x)*sin(y)*sin(y), (x,-1,1), (y,-1,1))).evalf()
        norm1 = fem.l2norm(np.sin(X)*np.sin(Y), mesh)
        np.testing.assert_almost_equal(norm1, 0.54, decimal=2)

        # 0.288675134594813 =
        # sqrt( integrate(sin(x)*sin(x)*sin(y)*sin(y), (x,-1,1), (y,-1,1))).evalf()
        h = 1
        V = np.array(
            [[0, 0],
             [h, 0],
             [0, h]])
        E = np.array(
            [[0, 1, 2]])
        mesh = fem.Mesh(V, E)
        mesh.generate_quadratic()
        V2, _ = mesh.V2, mesh.E2
        X, Y = V2[:, 0], V2[:, 1]
        np.testing.assert_almost_equal(fem.l2norm(X, mesh), 0.2886, decimal=4)

        # 0.545351286587159
        # = sqrt( integrate(sin(x)*sin(x)*sin(y)*sin(y), (x,-1,1), (y,-1,1))).evalf()
        V = data['vertices']
        E = data['elements']
        mesh = fem.Mesh(V, E)
        mesh.generate_quadratic()
        V2, _ = mesh.V2, mesh.E2
        X, Y = V2[:, 0], V2[:, 1]
        norm1 = fem.l2norm(np.sin(X)*np.sin(Y), mesh)
        np.testing.assert_almost_equal(norm1, 0.54, decimal=2)


class TestGradGradFEM(np.testing.TestCase):
    def test_gradgradfem(self):
        # two element
        h = 1
        V = np.array(
            [[0, 0],
             [h, 0],
             [0, h],
             [h, h]])
        E = np.array(
            [[0, 1, 2],
             [1, 3, 2]])

        mesh = fem.Mesh(V, E)

        A, b = fem.gradgradform(mesh)

        AA = np.array([[ 1.,  -0.5, -0.5,  0. ],
                       [-0.5,  1.,   0.,  -0.5],
                       [-0.5,  0.,   1.,  -0.5],
                       [ 0.,  -0.5, -0.5,  1. ]])

        np.testing.assert_array_almost_equal(A.toarray(), AA)

        # 3 x 3 mesh
        h = 1
        V = np.array(
            [[  0,   0],
             [  h,   0],
             [2*h,   0],
             [  0,   h],
             [  h,   h],
             [2*h,   h],
             [  0, 2*h],
             [  h, 2*h],
             [2*h, 2*h]])
        E = np.array(
            [[0, 1, 3],
             [1, 2, 4],
             [1, 4, 3],
             [2, 5, 4],
             [3, 4, 6],
             [4, 5, 7],
             [4, 7, 6],
             [5, 8, 7]])

        mesh = fem.Mesh(V, E)
        A, b = fem.gradgradform(mesh)
        AA = np.array([[ 1. , -0.5,  0. , -0.5,  0. ,  0. ,  0. ,  0. ,  0. ],
                       [-0.5,  2. , -0.5,  0. , -1. ,  0. ,  0. ,  0. ,  0. ],
                       [ 0. , -0.5,  1. ,  0. ,  0. , -0.5,  0. ,  0. ,  0. ],
                       [-0.5,  0. ,  0. ,  2. , -1. ,  0. , -0.5,  0. ,  0. ],
                       [ 0. , -1. ,  0. , -1. ,  4. , -1. ,  0. , -1. ,  0. ],
                       [ 0. ,  0. , -0.5,  0. , -1. ,  2. ,  0. ,  0. , -0.5],
                       [ 0. ,  0. ,  0. , -0.5,  0. ,  0. ,  1. , -0.5,  0. ],
                       [ 0. ,  0. ,  0. ,  0. , -1. ,  0. , -0.5,  2. , -0.5],
                       [ 0. ,  0. ,  0. ,  0. ,  0. , -0.5,  0. , -0.5,  1. ]])

        np.testing.assert_array_almost_equal(A.toarray(), AA)

        # non zero f, all zero g
        def f(x, y):
            return 0*x + 0*y + 1.0

        def g(x, y):
            return 0*x + 0*y + 0.0

        tol = 1e-12
        X, Y = V[:, 0], V[:, 1]
        id1 = np.where(abs(Y) < tol)[0]
        id2 = np.where(abs(Y-2*h) < tol)[0]
        id3 = np.where(abs(X) < tol)[0]
        id4 = np.where(abs(X-2*h) < tol)[0]

        bc = [{'id': id1, 'g': g},
              {'id': id2, 'g': g},
              {'id': id3, 'g': g},
              {'id': id4, 'g': g}]
        mesh = fem.Mesh(V, E)
        A, b = fem.gradgradform(mesh, f=f)
        A, b = fem.applybc(A, b, mesh, bc)

        AA = np.array([[1., 0., 0., 0., 0., 0., 0., 0., 0.],
                       [0., 1., 0., 0., 0., 0., 0., 0., 0.],
                       [0., 0., 1., 0., 0., 0., 0., 0., 0.],
                       [0., 0., 0., 1., 0., 0., 0., 0., 0.],
                       [0., 0., 0., 0., 4., 0., 0., 0., 0.],
                       [0., 0., 0., 0., 0., 1., 0., 0., 0.],
                       [0., 0., 0., 0., 0., 0., 1., 0., 0.],
                       [0., 0., 0., 0., 0., 0., 0., 1., 0.],
                       [0., 0., 0., 0., 0., 0., 0., 0., 1.]])

        bb = np.array([0., 0., 0., 0., 1., 0., 0., 0., 0.])

        np.testing.assert_array_almost_equal(A.toarray(), AA)
        np.testing.assert_array_almost_equal(b, bb)

        # non zero boundary
        def f(x, y):
            return 0*x + 0*y + 1.0

        def g(x, y):
            return 0*x + 0*y + 0.0

        def g1(x, y):
            return 0*x + 0*y + 1.0

        tol = 1e-12
        X, Y = V[:, 0], V[:, 1]
        id1 = np.where(abs(Y) < tol)[0]
        id2 = np.where(abs(Y-2*h) < tol)[0]
        id3 = np.where(abs(X) < tol)[0]
        id4 = np.where(abs(X-2*h) < tol)[0]

        bc = [{'id': id1, 'g': g},
              {'id': id2, 'g': g},
              {'id': id3, 'g': g1},
              {'id': id4, 'g': g}]
        mesh = fem.Mesh(V, E)
        A, b = fem.gradgradform(mesh, f=f)
        A, b = fem.applybc(A, b, mesh, bc=bc)
        A = A.tocsr()
        u = sla.spsolve(A, b)
        np.testing.assert_array_almost_equal(u[id3], np.ones(u[id3].shape))
