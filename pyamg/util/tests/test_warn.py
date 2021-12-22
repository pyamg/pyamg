"""Test warnings used."""
import warnings
from numpy.testing import TestCase


class TestWarn(TestCase):
    def test_f(self):
        warnings.filterwarnings('ignore', message='another warning')
        warnings.warn('another warning!')
