from pyamg.testing import *
from pyamg.krylov import gmres, fgmres, cgne, cgnr, cg, bicgstab
from helper import real_runs, real_runs_restrt, complex_runs, complex_runs_restrt, fgmres_runs
from scipy import random

class TestKrylov(TestCase):
    @decorators.skipif(True)
    def test_gmres(self):
        # Ensure repeatability of tests
        random.seed(0)
        n_max=8

        real_runs(gmres, n_max=n_max)
        real_runs_restrt(gmres, n_max=n_max)
        complex_runs(gmres, n_max=n_max)
        complex_runs_restrt(gmres, n_max=n_max)
    
    @decorators.skipif(True)
    def test_fgmres(self):
        # Ensure repeatability of tests
        random.seed(0)
        n_max=8

        real_runs(fgmres, n_max=n_max)
        real_runs_restrt(fgmres, n_max=n_max)
        complex_runs(fgmres, n_max=n_max)
        complex_runs_restrt(fgmres, n_max=n_max)
        fgmres_runs(fgmres)

    def test_cgne(self):
        # Ensure repeatability of tests
        random.seed(0)
        n_max=7

        real_runs(cgne, n_max=n_max, Weak=True)
        complex_runs(cgne, n_max=n_max, Weak=True)

    def test_cgnr(self):
        # Ensure repeatability of tests
        random.seed(0)
        n_max=7

        real_runs(cgnr, n_max=n_max, Weak=True)
        complex_runs(cgnr, n_max=n_max, Weak=True)
    
    def test_cg(self):
        # Ensure repeatability of tests
        random.seed(0)
        n_max=7

        real_runs(cg, n_max=n_max, Symmetric=True, Weak=True)
        complex_runs(cg, n_max=n_max, Symmetric=True, Weak=True)
    
    def test_bicgstab(self):
        # Ensure repeatability of tests
        random.seed(0)
        n_max=7

        real_runs(bicgstab, n_max=n_max, Weak=True)
        complex_runs(bicgstab, n_max=n_max, Weak=True)

