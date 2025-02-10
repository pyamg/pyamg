"""Classical AMG (Ruge-Stuben AMG)."""


from warnings import warn
from scipy.sparse import csr_array, issparse, SparseEfficiencyWarning
import numpy as np
import scipy as sp

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


def geoamg_solver(A: sp.sparse.base.spmatrix,
                  cpts_list: list[NDArray],
                  C_list: list[sp.sparse.base.spmatrix],
                  interpolation: Union[str, tuple[str, dict[str, Any]]] = 'classical',
                  presmoother=('gauss_seidel', {'sweep': 'symmetric'}),
                  postsmoother=('gauss_seidel', {'sweep': 'symmetric'}),
                  keep=False, **kwargs):
    """Create a multilevel solver using Classical AMG components on a predefined grids.

    Parameters
    ----------
    A : csr_array
        Square matrix in CSR format
    interpolation : str, default 'classical'
        Method for interpolation. Options include 'direct', 'classical'.
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

    # convert A to csr
    if not issparse(A) or A.format != 'csr':
        try:
            A = csr_array(A)
            warn('Implicit conversion of A to CSR', SparseEfficiencyWarning)
        except Exception as e:
            raise TypeError('Argument A must have type csr_array, '
                            'or be convertible to csr_array') from e
    # preprocess A
    A = asfptype(A)
    if A.shape[0] != A.shape[1]:
        raise ValueError('expected square matrix')

    levels[-1].A = A

    for i, (cpts, C) in enumerate(zip(cpts_list, C_list)):

        A = levels[-1].A

        splitting = np.zeros(A.shape[0], dtype=np.bool)
        splitting[cpts] = True

        # Generate the interpolation matrix that maps from the coarse-grid to the fine-grid
        fn, kwargs = _unpack_arg(interpolation)
        if fn == 'classical':
            P = classical_interpolation(A, C, splitting, **kwargs)
        elif fn == 'direct':
            P = direct_interpolation(A, C, splitting, **kwargs)
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
        return False

    ml = MultilevelSolver(levels, **kwargs)
    change_smoothers(ml, presmoother, postsmoother)
    return ml
