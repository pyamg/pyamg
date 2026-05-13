"""Threaded sparse BLAS kernels (Python-level wrappers).

This module provides clean Python wrappers around `pyamg.amg_core.sparse_blas`
kernels: input validation, dtype dispatch, output allocation. The C++
kernels themselves live in `pyamg/amg_core/sparse_blas/`.

Avoiding thread oversubscription
--------------------------------
If your numpy/scipy is linked against a threaded BLAS (OpenBLAS, MKL), set
``OPENBLAS_NUM_THREADS=1`` / ``MKL_NUM_THREADS=1`` in the environment, or
use :mod:`threadpoolctl`, before importing pyamg. Otherwise BLAS routines
called from inside scipy (e.g. spectral-radius estimation) will spawn their
own thread pool that competes with pyamg's OpenMP for cores.
"""

import contextlib
import functools
from warnings import warn

import numpy as np
from scipy.sparse import bsr_array, csr_array, issparse, sparray, spmatrix
from scipy.sparse._sputils import upcast

from ..amg_core import sparse_blas as _core

try:
    from threadpoolctl import threadpool_limits as _threadpool_limits
    _HAS_THREADPOOLCTL = True
except ImportError:
    _threadpool_limits = None
    _HAS_THREADPOOLCTL = False


def has_openmp():
    """Return True if pyamg was built with OpenMP support."""
    return bool(_core.has_openmp())


def set_num_threads(n):
    """Set the OpenMP thread count for pyamg sparse BLAS kernels.

    Defaults to whatever ``OMP_NUM_THREADS`` (or the OpenMP runtime default)
    selects at process start. Calling this overrides for the current process.
    No-op if pyamg was built without OpenMP.
    """
    _core.set_num_threads(int(n))


def get_num_threads():
    """Return the current OpenMP thread count (1 if OpenMP unavailable)."""
    return int(_core.get_num_threads())


# ----- BLAS thread scoping -----
#
# Threaded BLAS in numpy/scipy (OpenBLAS, MKL) competes with pyamg's OpenMP
# for cores when both are active in the same process. The fix is to pin BLAS
# to one thread inside pyamg's hot paths via `threadpoolctl`. We use a
# context manager so BLAS threading is restored on exit.
#
# If threadpoolctl is not installed we warn once and run unscoped — the
# user is on their own to set OPENBLAS_NUM_THREADS / MKL_NUM_THREADS.

_warned_no_threadpoolctl = False


@contextlib.contextmanager
def serial_blas():
    """Context manager: restrict numpy/scipy BLAS to one thread.

    Use this around hot pyamg code paths (solve, setup) to prevent BLAS
    thread oversubscription when pyamg's own OpenMP is active. No-op (with
    a one-time warning) if :mod:`threadpoolctl` is unavailable.

    Examples
    --------
    >>> from pyamg.util.sparse_blas import serial_blas
    >>> with serial_blas():
    ...     pass  # any BLAS calls in here run single-threaded
    """
    global _warned_no_threadpoolctl
    if _HAS_THREADPOOLCTL:
        with _threadpool_limits(limits=1, user_api='blas'):
            yield
    else:
        if not _warned_no_threadpoolctl:
            warn(
                'threadpoolctl is not installed; BLAS may oversubscribe cores '
                'when pyamg OpenMP is active. Install threadpoolctl, or set '
                'OPENBLAS_NUM_THREADS=1 / MKL_NUM_THREADS=1 in the environment.',
                stacklevel=2,
            )
            _warned_no_threadpoolctl = True
        yield


def with_serial_blas(fn):
    """Decorator: run ``fn`` inside :func:`serial_blas`.

    Wraps the decorated function so that all BLAS calls made during its
    execution are pinned to a single thread. Used to wrap setup-phase
    entry points (solver constructors, hierarchy builders) that mix
    pyamg's OpenMP-threaded kernels with scipy/numpy operations whose
    threaded BLAS would otherwise oversubscribe cores.

    Nesting is safe (threadpoolctl handles re-entrancy).
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with serial_blas():
            return fn(*args, **kwargs)
    return wrapper


def _index_array(arr):
    """Return arr cast to int32 (the C++ kernel's index type)."""
    if arr.dtype == np.int32:
        return arr
    return arr.astype(np.int32, copy=False)


def spmv(A, x, out=None):
    """Compute y = A @ x with the threaded pyamg kernel.

    Parameters
    ----------
    A : csr_array or bsr_array
        Sparse matrix.
    x : ndarray, 1-D
        Input vector. Length must be A.shape[1]. Dtype must match A.dtype
        (or be safely castable; we don't copy).
    out : ndarray, 1-D, optional
        Output buffer of length A.shape[0] and matching dtype. Allocated if
        not given. Overwritten in place.

    Returns
    -------
    ndarray, 1-D
        ``y = A @ x`` of length A.shape[0].
    """
    if not issparse(A):
        raise TypeError(f'spmv requires a sparse matrix, got {type(A).__name__}')
    if A.format not in ('csr', 'bsr'):
        raise ValueError(f'spmv supports csr/bsr; got {A.format}')

    n_row, n_col = A.shape
    x = np.ascontiguousarray(x)
    if x.ndim != 1 or x.shape[0] != n_col:
        raise ValueError(f'x has shape {x.shape}, expected ({n_col},)')

    dtype = A.dtype
    if x.dtype != dtype:
        x = x.astype(dtype, copy=False)

    if out is None:
        out = np.empty(n_row, dtype=dtype)
    elif out.shape != (n_row,) or out.dtype != dtype:
        raise ValueError(f'out must have shape ({n_row},) and dtype {dtype}')

    Ap = _index_array(A.indptr)
    Aj = _index_array(A.indices)

    if A.format == 'csr':
        _core.csr_matvec(Ap, Aj, A.data, x, out)
    else:  # bsr
        r, c = A.blocksize
        _core.bsr_matvec(int(r), int(c), Ap, Aj, np.ravel(A.data), x, out)
    return out


def matvec(A, x):
    """Compute ``A @ x`` using the threaded kernel.

    Dispatches to :func:`spmv` for CSR/BSR sparse matrices. Falls back to
    scipy's ``A @ x`` only when our kernel cannot handle the operands —
    non-CSR/BSR formats (CSC, COO, dense, LinearOperator), multi-column
    right-hand side, or mismatched dtypes.

    The fallback path is bit-for-bit equivalent to ``A @ x``.
    """
    if not isinstance(x, np.ndarray) or x.ndim != 1:
        return A @ x
    if not (isinstance(A, (sparray, spmatrix)) and A.format in ('csr', 'bsr')):
        return A @ x
    if x.dtype != A.dtype:
        return A @ x
    return spmv(A, x)


def spgemm(A, B, sort_indices=True):
    """Compute ``C = A @ B`` for CSR sparse matrices using the threaded
    Gustavson SpGEMM kernel.

    Parameters
    ----------
    A, B : csr_array
        Both must be CSR with compatible inner dimensions
        (``A.shape[1] == B.shape[0]``). Other formats raise ValueError.
    sort_indices : bool, optional
        If True (default), each output row's column indices are sorted
        ascending — matching scipy's behavior. Set False to skip the per-row
        sort if the caller doesn't need it.

    Returns
    -------
    csr_array
        ``C = A @ B``, shape ``(A.shape[0], B.shape[1])``.
    """
    if not (issparse(A) and A.format == 'csr'):
        raise ValueError(f'spgemm A must be csr; got {getattr(A, "format", type(A).__name__)}')
    if not (issparse(B) and B.format == 'csr'):
        raise ValueError(f'spgemm B must be csr; got {getattr(B, "format", type(B).__name__)}')
    if A.shape[1] != B.shape[0]:
        raise ValueError(f'inner dim mismatch: A {A.shape} @ B {B.shape}')

    n_row, _ = A.shape
    _, n_col = B.shape

    out_dtype = np.dtype(upcast(A.dtype, B.dtype))
    Ax = A.data if A.dtype == out_dtype else A.data.astype(out_dtype, copy=False)
    Bx = B.data if B.dtype == out_dtype else B.data.astype(out_dtype, copy=False)

    Ap = _index_array(A.indptr)
    Aj = _index_array(A.indices)
    Bp = _index_array(B.indptr)
    Bj = _index_array(B.indices)

    Cp = np.empty(n_row + 1, dtype=np.int32)
    _core.csr_matmat_pass1(int(n_col), Ap, Aj, Bp, Bj, Cp)

    nnz = int(Cp[-1])
    Cj = np.empty(nnz, dtype=np.int32)
    Cx = np.empty(nnz, dtype=out_dtype)

    _core.csr_matmat_pass2(
        int(n_col), Ap, Aj, Ax, Bp, Bj, Bx, Cp, Cj, Cx, sort_indices,
    )

    return csr_array((Cx, Cj, Cp), shape=(n_row, n_col))


def matmat(A, B):
    """Compute ``A @ B`` using the threaded SpGEMM kernel.

    Dispatches to :func:`spgemm` for CSR @ CSR. Falls back to scipy's
    ``A @ B`` only when our kernel cannot handle the operands —
    non-CSR formats (BSR, CSC, COO) or non-sparse inputs.
    """
    if not (issparse(A) and issparse(B)):
        return A @ B
    if A.format != 'csr' or B.format != 'csr':
        return A @ B
    return spgemm(A, B)


def _rap_block_size(M):
    """Return block size of CSR (1) or square BSR (block dim).
    ``ValueError`` for non-square BSR or other formats.
    """
    if not issparse(M):
        raise ValueError(f'rap: expected sparse matrix, got {type(M).__name__}')
    if M.format == 'csr':
        return 1
    if M.format == 'bsr':
        br, bc = M.blocksize
        if br != bc:
            raise ValueError(f'rap: BSR with non-square blocks {M.blocksize} '
                             f'is not supported')
        return br
    raise ValueError(f'rap: unsupported format {M.format!r}; use csr or bsr')


def rap(R, A, P, sort_indices=True):
    """Fused triple-product Galerkin operator ``R @ A @ P``.

    Operands may be CSR or square-block BSR; if BSR, all three must share
    the same block size. Computed in a single call without materializing
    ``R @ A``. Calls one of six fully-unrolled kernels for block sizes in
    ``{1, 2, 3, 4, 6, 8}``; other sizes use a runtime-block-size kernel.

    Parameters
    ----------
    R, A, P : csr_array or bsr_array
        Sparse matrices with compatible inner dimensions
        (``R.shape[1] == A.shape[0]`` and ``A.shape[1] == P.shape[0]``).
        BSR operands must have square blocks of the same size.
    sort_indices : bool, optional
        If True (default), each output row's (block-)column indices are
        sorted ascending.

    Returns
    -------
    csr_array (CSR inputs) or bsr_array (BSR inputs)
        ``C = R @ A @ P``, shape ``(R.shape[0], P.shape[1])``.
    """
    bs_R = _rap_block_size(R)
    bs_A = _rap_block_size(A)
    bs_P = _rap_block_size(P)
    if not (bs_R == bs_A == bs_P):
        raise ValueError(f'rap: inconsistent block sizes R={bs_R} A={bs_A} '
                         f'P={bs_P}; all three must match')
    if R.shape[1] != A.shape[0]:
        raise ValueError(f'inner dim mismatch: R {R.shape} @ A {A.shape}')
    if A.shape[1] != P.shape[0]:
        raise ValueError(f'inner dim mismatch: A {A.shape} @ P {P.shape}')

    bs = bs_R
    # Block-row dimensions: number of *blocks* along rows/cols.
    n_block_row = R.shape[0] // bs
    n_block_inner = A.shape[1] // bs
    n_block_col = P.shape[1] // bs

    out_dtype = np.dtype(upcast(R.dtype, A.dtype, P.dtype))
    # BSR data is shape (nnz_blocks, bs, bs); flatten so the kernel reads
    # it as a row-major scalar array. (CSR data is already 1D.)
    Rx = np.ascontiguousarray(R.data, dtype=out_dtype).reshape(-1)
    Ax = np.ascontiguousarray(A.data, dtype=out_dtype).reshape(-1)
    Px = np.ascontiguousarray(P.data, dtype=out_dtype).reshape(-1)

    Rp = _index_array(R.indptr); Rj = _index_array(R.indices)
    Ap = _index_array(A.indptr); Aj = _index_array(A.indices)
    Pp = _index_array(P.indptr); Pj = _index_array(P.indices)

    Cp = np.empty(n_block_row + 1, dtype=np.int32)
    _core.rap_pass1(int(n_block_inner), int(n_block_col),
                    Rp, Rj, Ap, Aj, Pp, Pj, Cp)

    n_block_nnz = int(Cp[-1])
    Cj = np.empty(n_block_nnz, dtype=np.int32)
    Cx = np.empty(n_block_nnz * bs * bs, dtype=out_dtype)

    _core.rap_pass2(
        int(bs), int(n_block_inner), int(n_block_col),
        Rp, Rj, Rx, Ap, Aj, Ax, Pp, Pj, Px,
        Cp, Cj, Cx, sort_indices,
    )

    if bs == 1:
        return csr_array((Cx, Cj, Cp),
                         shape=(R.shape[0], P.shape[1]))
    return bsr_array((Cx.reshape(n_block_nnz, bs, bs), Cj, Cp),
                     shape=(R.shape[0], P.shape[1]),
                     blocksize=(bs, bs))


def rap_compatible(R, A, P):
    """Return True if all three operands are in formats ``rap()`` accepts:
    CSR, or square-block BSR with matching block sizes across R, A, P.
    """
    def bs(M):
        if not issparse(M):
            return None
        if M.format == 'csr':
            return 1
        if M.format == 'bsr':
            br, bc = M.blocksize
            return br if br == bc else None
        return None
    bR, bA, bP = bs(R), bs(A), bs(P)
    return bR is not None and bR == bA == bP
