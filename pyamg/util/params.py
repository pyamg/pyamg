"""Parameter type settings."""

import numpy as np


def set_tol(dtype):
    """Set a tolerance based on a numpy dtype char.

    Parameters
    ----------
    dtype : np.dtype
        numpy dtype

    Returns
    -------
    tol : float
        A smallish value based on precision

    Notes
    -----
    Handles both real and complex (through the .lower() case)

    See Also
    --------
    numpy.typecodes, numpy.sctypes
    """
    if dtype.char.lower() == 'f':
        tol = 1e3 * np.finfo(np.single).eps
    elif dtype.char.lower() == 'd':
        tol = 1e6 * np.finfo(np.double).eps
    elif dtype.char.lower() == 'g':
        tol = 1e6 * np.finfo(np.longdouble).eps
    else:
        raise ValueError('Attempting to set a tolerance for an unsupported precision.')

    return tol
