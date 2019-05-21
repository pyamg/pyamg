import warnings
from numpy.testing import TestCase


warnings.filterwarnings("ignore", message="another")


class TestWarn(TestCase):
    def test_f(self):
        warnings.warn("another warning!")
