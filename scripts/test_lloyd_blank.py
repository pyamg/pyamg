"""Test balanced lloyd clustering."""

import numpy as np
from numpy.testing import assert_array_equal
from scipy import sparse

import pytest

import pyamg
from pyamg import amg_core


@pytest.fixture()
def construct_1dfd_graph():
    u = np.ones(9, dtype=np.float64)
    return u

def test_balanced_lloyd_1d(construct_1dfd_graph):
    v = construct_1dfd_graph

    assert_array_equal(v, [1, 1, 1, 1, 1, 1, 1, 1, 1])
