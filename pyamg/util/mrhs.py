"""Support for multiple right-hand sides (MRHS).

These functions manage rhs with shape (n,), (n,1), and (n,k), including
convergence tests and other utilities.
"""

import numpy as np


def normalize_rhs(b):
    """Return a 2-D view of ``b``.

    Parameters
    ----------
    b : ndarray
        Right-hand side with shape ``(n,)``, ``(n, 1)``, or ``(n, k)``

    Returns
    -------
    B : ndarray
        2-D array with shape ``(n, k)``
    oshape : tuple
        The shape of ``b`` on entry, for use with :func:`restore_shape`.

    """
    if b.ndim == 1:
        B = np.asfortranarray(b.reshape(-1, 1))
    elif b.ndim == 2:
        B = np.asfortranarray(b)
    else:
        raise ValueError(f'expected b.ndim to be 1 or 2, got {b.ndim}')
    return B, b.shape


def restore_shape(x, oshape):
    """Reshape a 2-D vector back to the original shape.

    Parameters
    ----------
    x : ndarray
        2-D solution of shape ``(n, k)``.
    oshape : tuple
        Original shape.

    Returns
    -------
    ndarray
        ``x`` reshaped to ``oshape``.

    """
    return x.reshape(oshape)


def per_column_norms(R):
    """Return the 2-norm of each column of ``R``.

    Parameters
    ----------
    R : ndarray
        1-D or 2-D array.

    Returns
    -------
    ndarray
        For 1-D ``R``, a length-1 array containing ``||R||``.  For 2-D ``R``
        of shape ``(n, k)``, a length-``k`` arrey of column norms.

    """
    if R.ndim == 1:
        return np.array([np.linalg.norm(R)])
    return np.linalg.norm(R, axis=0)


def all_converged(normr, tol, normb):
    """Determine when each column has converged.

    Parameters
    ----------
    normr : ndarray
        Per-column residual norms, shape ``(k,)``.
    tol : float
        Relative tolerance.
    normb : ndarray
        Per-column right-hand side norms, shape ``(k,)``.

    Returns
    -------
    bool
        ``True`` if ``normr[j] < tol * normb[j]`` for every ``j``.

    """
    return bool(np.all(normr < tol * normb))
