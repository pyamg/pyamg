"""Tests for ``pyamg.util.sparse_blas``.

Two layers:

1. **Kernel correctness** — ``spmv``, ``spgemm``, ``rap`` against scipy
   reference for CSR and BSR (specialized + runtime block sizes), real and
   complex dtypes.
2. **Solver-quality regression** — for each solver flavor, build the
   hierarchy with the threaded RAP kernel and again with the scipy
   fallback (``_rap_ok`` patched off), and verify the level structure,
   coarse operators, and residual histories agree to numeric tolerance.
"""
import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose

import pyamg
from pyamg.gallery import linear_elasticity, poisson
from pyamg.util.sparse_blas import (
    get_num_threads,
    has_openmp,
    matmat,
    matvec,
    rap,
    rap_compatible,
    serial_blas,
    set_num_threads,
    spgemm,
    spmv,
)


# ---------------------------------------------------------------------------
# Kernel correctness
# ---------------------------------------------------------------------------

class TestSpmv:
    @pytest.mark.parametrize("shape", [(50,), (30, 30), (10, 10, 10)])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex128])
    def test_csr_matches_scipy(self, shape, dtype):
        A = poisson(shape, format='csr').astype(dtype)
        rng = np.random.default_rng(0)
        x = rng.random(A.shape[1]).astype(dtype)
        if np.issubdtype(dtype, np.complexfloating):
            x = x + 1j * rng.random(A.shape[1]).astype(dtype)
        rtol = 1e-5 if dtype == np.float32 else 1e-12
        assert_allclose(spmv(A, x), A @ x, rtol=rtol)

    @pytest.mark.parametrize("bs", [1, 2, 3, 4, 6, 8, 5, 7])
    def test_bsr_matches_scipy(self, bs):
        # bs in {1,2,3,4,6,8} hits the compile-time-specialized templates;
        # 5 and 7 exercise the runtime-bs fallback.
        n_blocks = 30
        A = poisson((n_blocks * bs,), format='csr').tobsr(blocksize=(bs, bs))
        x = np.random.default_rng(0).random(A.shape[1])
        assert_allclose(spmv(A, x), A @ x, rtol=1e-12)

    def test_out_buffer_reuse(self):
        A = poisson((20, 20), format='csr')
        x = np.random.default_rng(0).random(A.shape[1])
        out = np.zeros(A.shape[0])
        y = spmv(A, x, out=out)
        assert y is out
        assert_allclose(out, A @ x, rtol=1e-12)

    def test_dim_mismatch_raises(self):
        A = poisson((10, 10), format='csr')
        with pytest.raises(ValueError):
            spmv(A, np.zeros(A.shape[1] + 1))

    def test_unsupported_format_raises(self):
        A = poisson((10, 10), format='csr').tocoo()
        with pytest.raises(ValueError):
            spmv(A, np.zeros(A.shape[1]))

    def test_dense_input_raises(self):
        with pytest.raises(TypeError):
            spmv(np.eye(5), np.zeros(5))


class TestMatvec:
    def test_csr_alias_of_spmv(self):
        A = poisson((20, 20), format='csr')
        x = np.random.default_rng(0).random(A.shape[1])
        assert_allclose(matvec(A, x), spmv(A, x), rtol=1e-12)

    def test_bsr_alias_of_spmv(self):
        A = poisson((40,), format='csr').tobsr(blocksize=(2, 2))
        x = np.random.default_rng(0).random(A.shape[1])
        assert_allclose(matvec(A, x), spmv(A, x), rtol=1e-12)


class TestSpgemm:
    @pytest.mark.parametrize("dtype", [np.float64, np.complex128])
    def test_csr_matches_scipy(self, dtype):
        A = poisson((30, 30), format='csr').astype(dtype)
        B = poisson((30, 30), format='csr').astype(dtype)
        assert_allclose(spgemm(A, B).toarray(), (A @ B).toarray(), rtol=1e-12)

    def test_sort_indices_default_true(self):
        A = poisson((30,), format='csr')
        C = spgemm(A, A)
        for i in range(C.shape[0]):
            row = C.indices[C.indptr[i]:C.indptr[i + 1]]
            assert np.all(np.diff(row) >= 0)

    def test_non_csr_raises(self):
        A = poisson((40,), format='csr').tobsr(blocksize=(2, 2))
        with pytest.raises(ValueError):
            spgemm(A, A)


class TestMatmat:
    def test_csr_uses_kernel(self):
        A = poisson((20,), format='csr')
        assert_allclose(matmat(A, A).toarray(), spgemm(A, A).toarray(), rtol=1e-12)

    @pytest.mark.parametrize("bs", [2, 3, 4])
    def test_bsr_falls_back_to_scipy(self, bs):
        # matmat dispatches BSR to scipy A @ B; verify result correctness.
        n = 12 * bs
        A = poisson((n,), format='csr').tobsr(blocksize=(bs, bs))
        assert_allclose(matmat(A, A).toarray(), (A @ A).toarray(), rtol=1e-12)


class TestRap:
    def _make_csr_p(self, n_f, n_c, dtype):
        rng = np.random.default_rng(0)
        Pd = np.zeros((n_f, n_c), dtype=dtype)
        for j in range(n_c):
            Pd[2 * j, j] = 1.0
            if 2 * j + 1 < n_f:
                Pd[2 * j + 1, j] = 0.5 + 0.1 * rng.random()
        return sp.csr_array(Pd)

    @pytest.mark.parametrize("dtype", [np.float64, np.complex128])
    def test_csr_matches_scipy(self, dtype):
        n_f, n_c = 50, 25
        A = poisson((n_f,), format='csr').astype(dtype)
        P = self._make_csr_p(n_f, n_c, dtype)
        R = sp.csr_array(P.toarray().conj().T)
        assert_allclose(rap(R, A, P).toarray(),
                        (R @ A @ P).toarray(), rtol=1e-12)

    @pytest.mark.parametrize("bs", [1, 2, 3, 4])
    def test_bsr_matches_scipy(self, bs):
        n_f_blocks, n_c_blocks = 20, 10
        n_f = n_f_blocks * bs
        A = poisson((n_f,), format='csr').tobsr(blocksize=(bs, bs))
        rng = np.random.default_rng(0)
        P_indptr = np.arange(n_f_blocks + 1, dtype=np.int32)
        P_indices = (np.arange(n_f_blocks) // 2).astype(np.int32)
        P_data = rng.random((n_f_blocks, bs, bs))
        P = sp.bsr_array((P_data, P_indices, P_indptr),
                         shape=(n_f, n_c_blocks * bs), blocksize=(bs, bs))
        R = sp.bsr_array(P.toarray().T).tobsr(blocksize=(bs, bs))
        assert_allclose(rap(R, A, P).toarray(),
                        (R @ A @ P).toarray(), rtol=1e-12)


class TestRapCompatible:
    def test_csr_compatible(self):
        A = poisson((10, 10), format='csr')
        P = sp.csr_array(np.random.default_rng(0).random((100, 50)))
        R = sp.csr_array(P.toarray().T)
        assert rap_compatible(R, A, P)

    def test_mixed_format_incompatible(self):
        A = poisson((10, 10), format='csr').tocoo()
        P = sp.csr_array(np.random.default_rng(0).random((100, 50)))
        R = sp.csr_array(P.toarray().T)
        assert not rap_compatible(R, A, P)

    def test_mismatched_blocksize_incompatible(self):
        A = poisson((40,), format='csr').tobsr(blocksize=(2, 2))
        P = sp.csr_array(np.random.default_rng(0).random((40, 20))) \
              .tobsr(blocksize=(4, 4))
        R = sp.bsr_array(P.toarray().T).tobsr(blocksize=(4, 4))
        assert not rap_compatible(R, A, P)

    def test_non_square_blocksize_incompatible(self):
        n = 24
        A = poisson((n,), format='csr')
        # build a (3, 2) blocksize BSR — non-square
        P = sp.csr_array(np.random.default_rng(0).random((n, 16))) \
              .tobsr(blocksize=(3, 2))
        assert not rap_compatible(P.T, A, P)


class TestThreadControl:
    def test_set_get_round_trip(self):
        prev = get_num_threads()
        try:
            set_num_threads(2)
            if has_openmp():
                assert get_num_threads() == 2
            set_num_threads(1)
            assert get_num_threads() == 1
        finally:
            set_num_threads(prev)

    def test_has_openmp_returns_bool(self):
        assert isinstance(has_openmp(), bool)


class TestSerialBlas:
    def test_smoke_yields_no_error(self):
        with serial_blas():
            A = poisson((20, 20), format='csr')
            x = np.random.default_rng(0).random(A.shape[1])
            _ = A @ x


# ---------------------------------------------------------------------------
# Solver-quality regression: kernel hierarchy vs scipy fallback
# ---------------------------------------------------------------------------
#
# Each solver flavor calls ``rap_compatible`` (imported as ``_rap_ok``) and
# runs the threaded RAP kernel when the operands are CSR or square-block
# BSR. Patching ``_rap_ok`` to always-False forces the scipy fallback
# (``A = R @ A @ P``). A correct kernel build should produce a level
# structure and residual history numerically equivalent to the scipy
# reference.

_RAP_OK_MODULES = (
    'pyamg.classical.classical',
    'pyamg.classical.air',
    'pyamg.aggregation.aggregation',
    'pyamg.aggregation.rootnode',
    'pyamg.aggregation.pairwise',
    'pyamg.aggregation.adaptive',
    'pyamg.aggregation.aggregate',
)


def _build_with_scipy_rap(factory):
    """Run ``factory()`` with the threaded RAP kernel disabled at every call
    site. Returns the resulting ``MultiLevel``."""
    import importlib
    saved = {}
    try:
        for modname in _RAP_OK_MODULES:
            mod = importlib.import_module(modname)
            saved[modname] = mod._rap_ok
            mod._rap_ok = lambda R, A, P: False
        return factory()
    finally:
        for modname, fn in saved.items():
            importlib.import_module(modname)._rap_ok = fn


def _build_pair(factory, seed=0):
    """Build the same hierarchy twice, once with the kernel and once with
    the scipy fallback. Resets ``np.random`` between runs so that random
    vectors used inside ``approximate_spectral_radius`` (and other
    ``np.random.rand`` callers in pyamg setup) are identical across the
    two builds — otherwise the spectral-radius-driven prolongation
    smoother yields different P matrices, which are unrelated to the
    kernel's correctness.
    """
    np.random.seed(seed)
    ml_kernel = factory()
    np.random.seed(seed)
    ml_scipy = _build_with_scipy_rap(factory)
    return ml_kernel, ml_scipy


def _residual_history(ml, b, x0, tol=1e-10, maxiter=30):
    res = []
    ml.solve(b.copy(), x0=x0.copy(), tol=tol, maxiter=maxiter, residuals=res)
    return np.asarray(res)


def _assert_hierarchies_match(ml_a, ml_b, atol=1e-10, rtol=1e-10):
    assert len(ml_a.levels) == len(ml_b.levels)
    for i, (la, lb) in enumerate(zip(ml_a.levels, ml_b.levels)):
        assert la.A.shape == lb.A.shape, f'level {i} shape mismatch'
        assert la.A.nnz   == lb.A.nnz,   f'level {i} nnz mismatch'
        # canonical form before comparing values
        Ad = la.A.tocsr().copy(); Ad.sum_duplicates(); Ad.sort_indices()
        Bd = lb.A.tocsr().copy(); Bd.sum_duplicates(); Bd.sort_indices()
        assert np.array_equal(Ad.indptr, Bd.indptr), f'level {i} indptr'
        assert np.array_equal(Ad.indices, Bd.indices), f'level {i} indices'
        assert_allclose(Ad.data, Bd.data, atol=atol, rtol=rtol,
                        err_msg=f'level {i} A.data')


@pytest.fixture
def small_csr_problem():
    A = poisson((50, 50), format='csr')
    rng = np.random.default_rng(0)
    return A, rng.random(A.shape[0]), np.zeros(A.shape[0])


@pytest.fixture
def small_bsr_problem():
    A, B = linear_elasticity((20, 20), format='bsr')
    rng = np.random.default_rng(0)
    return A, B, rng.random(A.shape[0]), np.zeros(A.shape[0])


class TestKernelVsScipyHierarchy:
    def test_rs_classical_csr(self, small_csr_problem):
        A, b, x0 = small_csr_problem
        factory = lambda: pyamg.ruge_stuben_solver(
            A, max_coarse=200, CF=('RS', {'second_pass': True}))
        ml_kernel, ml_scipy = _build_pair(factory)
        _assert_hierarchies_match(ml_kernel, ml_scipy)
        rk = _residual_history(ml_kernel, b, x0)
        rs = _residual_history(ml_scipy, b, x0)
        assert_allclose(rk, rs, atol=1e-10, rtol=1e-8)

    def test_smoothed_aggregation_csr(self, small_csr_problem):
        A, b, x0 = small_csr_problem
        factory = lambda: pyamg.smoothed_aggregation_solver(A, max_coarse=50)
        ml_kernel, ml_scipy = _build_pair(factory)
        _assert_hierarchies_match(ml_kernel, ml_scipy)
        rk = _residual_history(ml_kernel, b, x0)
        rs = _residual_history(ml_scipy, b, x0)
        assert_allclose(rk, rs, atol=1e-10, rtol=1e-8)

    def test_smoothed_aggregation_bsr(self, small_bsr_problem):
        A, B, b, x0 = small_bsr_problem
        factory = lambda: pyamg.smoothed_aggregation_solver(
            A, B=B, max_coarse=50)
        ml_kernel, ml_scipy = _build_pair(factory)
        _assert_hierarchies_match(ml_kernel, ml_scipy)
        rk = _residual_history(ml_kernel, b, x0)
        rs = _residual_history(ml_scipy, b, x0)
        assert_allclose(rk, rs, atol=1e-10, rtol=1e-8)

    def test_rootnode_csr(self, small_csr_problem):
        A, b, x0 = small_csr_problem
        factory = lambda: pyamg.rootnode_solver(A, max_coarse=50)
        ml_kernel, ml_scipy = _build_pair(factory)
        _assert_hierarchies_match(ml_kernel, ml_scipy)
        rk = _residual_history(ml_kernel, b, x0)
        rs = _residual_history(ml_scipy, b, x0)
        assert_allclose(rk, rs, atol=1e-10, rtol=1e-8)


class TestSolverConvergesReasonably:
    """Sanity floor: the kernel-built solver must reduce residual by at
    least 8 orders of magnitude in 30 V-cycle iterations on Poisson 50x50."""

    def test_rs_classical(self, small_csr_problem):
        A, b, x0 = small_csr_problem
        ml = pyamg.ruge_stuben_solver(
            A, max_coarse=200, CF=('RS', {'second_pass': True}))
        res = _residual_history(ml, b, x0)
        assert res[-1] / res[0] < 1e-8

    def test_smoothed_aggregation(self, small_csr_problem):
        A, b, x0 = small_csr_problem
        ml = pyamg.smoothed_aggregation_solver(A, max_coarse=50)
        res = _residual_history(ml, b, x0)
        assert res[-1] / res[0] < 1e-8
