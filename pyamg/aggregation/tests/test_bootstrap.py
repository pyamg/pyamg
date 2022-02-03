"""Test bootstrap solver."""
import numpy as np
import pyamg

A = pyamg.gallery.poisson((100, 100), format='csr')
b = np.random.rand(A.shape[0])


def test_bootstrap_solve():
    ml, _ = pyamg.bootstrap_solver(A)

    res = []
    _ = ml.solve(b, residuals=res)

    print(res)
