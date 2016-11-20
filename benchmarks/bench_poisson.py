"""
Benchmark Poisson setup and solve
"""
import pyamg
import numpy as np


class TimeSuite:
    pretty_name = "Poisson Timing"

    def setup(self):
        n = 1000
        self.A = pyamg.gallery.poisson((n, n), format="csr")
        self.settings = {"B": None,
                         "BH": None,
                         "symmetry": "hermitian",
                         "strength": "symmetric",
                         "aggregate": "standard",
                         "smooth": ("jacobi", {"omega": 4.0/3.0}),
                         "presmoother": ("gauss_seidel",
                                         {"sweep": "symmetric"}),
                         "postsmoother": ("gauss_seidel",
                                          {"sweep": "symmetric"}),
                         "improve_candidates": None,
                         "max_levels": 10,
                         "max_coarse": 10,
                         "diagonal_dominance": False,
                         "keep": False}
        self.b = np.zeros(self.A.shape[0])
        self.x0 = np.random.rand(self.A.shape[0])
        self.ml = pyamg.smoothed_aggregation_solver(self.A, **self.settings)

    def time_setup(self):
        ml = pyamg.smoothed_aggregation_solver(self.A, **self.settings)

    def time_solve(self):
        x = self.ml.solve(self.b, x0=self.x0,
                          tol=1e-08, maxiter=100, cycle='V', accel=None)

class PeakMemSuite:
    pretty_name = "Poisson Memory"

    def setup_cache(self):
        n = 1000
        self.A = pyamg.gallery.poisson((n, n), format="csr")
        self.settings = {"B": None,
                         "BH": None,
                         "symmetry": "hermitian",
                         "strength": "symmetric",
                         "aggregate": "standard",
                         "smooth": ("jacobi", {"omega": 4.0/3.0}),
                         "presmoother": ("gauss_seidel",
                                         {"sweep": "symmetric"}),
                         "postsmoother": ("gauss_seidel",
                                          {"sweep": "symmetric"}),
                         "improve_candidates": None,
                         "max_levels": 10,
                         "max_coarse": 10,
                         "diagonal_dominance": False,
                         "keep": False}

    def peakmem_setup(self):
        ml = pyamg.smoothed_aggregation_solver(self.A, **self.settings)
