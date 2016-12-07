from pyamg.gallery.mesh import regular_triangle_mesh

from numpy.testing import TestCase, assert_equal


class TestRegularTriangleMesh(TestCase):
    def test_1x1(self):
        try:
            regular_triangle_mesh(1, 0)
        except ValueError:
            pass
        else:
            raise Exception("regular_triangle_mesh(1,0) should throw an error")

    def test_2x2(self):
        Vert, E2V = regular_triangle_mesh(2, 2)

        assert_equal(Vert, [[0., 0.],
                            [1., 0.],
                            [0., 1.],
                            [1., 1.]])

        assert_equal(E2V, [[0, 3, 2],
                           [0, 1, 3]])

    def test_3x2(self):
        Vert, E2V = regular_triangle_mesh(3, 2)

        assert_equal(Vert, [[0., 0.],
                            [0.5, 0.],
                            [1., 0.],
                            [0., 1.],
                            [0.5, 1.],
                            [1., 1.]])

        assert_equal(E2V, [[0, 4, 3],
                           [1, 5, 4],
                           [0, 1, 4],
                           [1, 2, 5]])
