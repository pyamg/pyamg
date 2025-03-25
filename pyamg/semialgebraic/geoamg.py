"""Classical AMG (Ruge-Stuben AMG)."""


from warnings import warn
import numpy as np
import scipy as sp
from scipy import sparse

from pyamg.multilevel import MultilevelSolver
from pyamg.relaxation.smoothing import change_smoothers
from pyamg.classical.interpolate import direct_interpolation, classical_interpolation
from ..util.utils import asfptype

from typing import Union, Any
from numpy.typing import NDArray


def _unpack_arg(v):
    if isinstance(v, tuple):
        return v[0], v[1]
    return v, {}


def _interpolation1d(nc, nf):
    d = np.repeat([[1., 2, 1]], nc, axis=0).T
    I = np.zeros((3, nc), dtype=np.int32)
    for i in range(nc):
        I[:, i] = [2*i, 2*i+1, 2*i+2]
    J = np.repeat([np.arange(nc, dtype=np.int32)], 3, axis=0)
    P = sparse.coo_array((d.ravel(), (I.ravel(), J.ravel())), shape=(nf, nc)).tocsr()
    return 0.5 * P


def geoamg_solver(A,
                  cpts_list,
                  interpolation= 'classical',
                  C_list=None,
                  presmoother=('gauss_seidel', {'sweep': 'symmetric'}),
                  postsmoother=('gauss_seidel', {'sweep': 'symmetric'}),
                  keep=False, **kwargs):
    """Create a multilevel solver using Classical AMG components on a predefined grids.

    Parameters
    ----------
    A : csr_array
        Square matrix in CSR format
    cpts_list : list of ndarray
        A list of C-points for each level
    interpolation : str, default 'classical'
        Method for interpolation. Options include 'direct', 'classical'.
        If one provides theta as a SOC argument, SOC is recomputed for
        interpolation, so C_list is not needed and/or used.
    C_list : list of csr_array
        Optional list of strong connections for each level used in
        computing interpolation. If not provided, SOC is computed in
        interpolation routine.
    presmoother : str or dict
        Method used for presmoothing at each level.  Method-specific parameters
        may be passed in using a tuple, e.g.
        presmoother=('gauss_seidel',{'sweep':'symmetric}), the default.
    postsmoother : str or dict
        Postsmoothing method with the same usage as presmoother
    keep : bool, default False
        Flag to indicate keeping strength of connection (C) in the
        hierarchy for diagnostics.
    kwargs : dict
        Extra keywords passed to Multilevel class

    Returns
    -------
    ml : MultilevelSolver
        Multigrid hierarchy of matrices and prolongation operators

    See Also
    --------
    MultilevelSolver,

    """
    levels = [MultilevelSolver.Level()]
    nlevels = len(cpts_list)
    if C_list is None:
        C_list = [None]*nlevels
    else:
        if len(cpts_list) != len(C_list)
            raise ValueError('C_list length does not match cpts_list length')

    # convert A to csr
    if not sparse.issparse(A) or A.format != 'csr':
        try:
            A = sparse.csr_array(A)
            warn('Implicit conversion of A to CSR', sparse.SparseEfficiencyWarning)
        except Exception as e:
            raise TypeError('Argument A must have type csr_array, '
                            'or be convertible to csr_array') from e
    # preprocess A
    A = asfptype(A)
    if A.shape[0] != A.shape[1]:
        raise ValueError('expected square matrix')

    levels[-1].A = A

    for _, (cpts, C) in enumerate(zip(cpts_list, C_list)):

        A = levels[-1].A
        if C is not None:
            if not sparse.issparse(C) or C.format != 'csr':
                try:
                    C = sparse.csr_array(C)
                    warn('Implicit conversion of C to CSR', sparse.SparseEfficiencyWarning)
                except Exception as e:
                    raise TypeError('Argument C[l] must have type csr_array, '
                                    'or be convertible to csr_array') from e
            if C.shape[0] != C.shape[1] or C.shape[0] != A.shape[0]
                raise ValueError('expected square SOCNmatrix matching A')

        splitting = np.zeros(A.shape[0], dtype=np.int32)
        splitting[cpts] = True

        # Generate the interpolation matrix that maps from the coarse-grid to the fine-grid
        fn, kwargs = _unpack_arg(interpolation)
        if fn == 'classical':
            P = classical_interpolation(A, C, splitting, **kwargs)
        elif fn == 'direct':
            P = direct_interpolation(A, C, splitting, **kwargs)
        elif fn == 'geometric':
            P1d = _interpolation1d(int(np.sqrt(len(cpts))), int(np.sqrt(A.shape[0])))
            P = sparse.kron(P1d, P1d).tocsr()
        else:
            raise ValueError(f'Unknown interpolation method {interpolation}')

        # Generate the restriction matrix that maps from the fine-grid to the coarse-grid
        R = P.T.tocsr()

        # Store relevant information for this level
        if keep:
            levels[-1].C = C                           # strength of connection matrix

        levels[-1].splitting = splitting.astype(bool)  # C/F splitting
        levels[-1].P = P                               # prolongation operator
        levels[-1].R = R                               # restriction operator

        # Form next level through Galerkin product
        levels.append(MultilevelSolver.Level())
        A = R @ A @ P
        levels[-1].A = A

    ml = MultilevelSolver(levels, **kwargs)
    change_smoothers(ml, presmoother, postsmoother)
    return ml
