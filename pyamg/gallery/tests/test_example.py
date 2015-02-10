from pyamg.gallery.example import load_example

from numpy.testing import TestCase, assert_equal


class TestExample(TestCase):
    def test_load_example(self):
        knot = load_example('knot')

        A = knot['A']
        B = knot['B']
        vertices = knot['vertices']
        elements = knot['elements']

        assert_equal(A.shape, (239, 239))
        assert_equal(B.shape, (239, 1))
        assert_equal(vertices.shape, (240, 3))
        assert_equal(elements.shape, (480, 3))
