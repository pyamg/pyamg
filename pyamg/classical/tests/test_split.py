"""Test splitting methods."""

import numpy as np
from numpy.testing import assert_array_equal
import pyamg


class TestMIS:
    def test_paper_result(self):
        # example from Figure 4.1 in
        # Reducing Complexity in Parallel Algebraic Multigrid Preconditioners
        # SIAM 2006

        S = pyamg.gallery.poisson((7, 7), type='FE', format='csr')
        w = [3.2, 5.6, 5.8, 5.6, 5.9, 5.9, 3.0,
             5.0, 8.8, 8.5, 8.6, 8.7, 8.9, 5.3,
             5.3, 8.7, 8.3, 8.4, 8.3, 8.8, 5.9,
             5.7, 8.6, 8.3, 8.8, 8.3, 8.1, 5.0,
             5.9, 8.1, 8.8, 8.9, 8.4, 8.2, 5.9,
             5.2, 8.0, 8.5, 8.2, 8.6, 8.9, 5.1,
             3.7, 5.3, 5.0, 5.9, 5.4, 5.3, 3.4]

        w = np.array(w)
        splitting = pyamg.classical.split.MIS(S, w)
        splitting_paper =\
            np.array([[0, 0, 0, 0, 0, 0, 0],
                      [0, 1, 0, 1, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 1, 0],
                      [0, 0, 0, 1, 0, 0, 0],
                      [0, 1, 0, 0, 0, 1, 0],
                      [0, 0, 0, 1, 0, 0, 0]], dtype=np.int32)
        assert_array_equal(splitting.reshape((7, 7)), splitting_paper)
