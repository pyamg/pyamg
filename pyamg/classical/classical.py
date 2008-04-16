"""Classical AMG (Ruge-Stuben AMG)"""

__docformat__ = "restructuredtext en"

from scipy.sparse import csr_matrix, isspmatrix_csr

from pyamg.multilevel import multilevel_solver
from pyamg import multigridtools

from interpolate import *
from pyamg.strength import *

__all__ = ['ruge_stuben_solver']


def ruge_stuben_solver(A, 
                       strength=('classical',{'theta':0.25}), 
                       CF='RS', 
                       max_levels=10, max_coarse=500, **kwargs):
    """Create a multilevel solver using Classical AMG (Ruge-Stuben AMG)

    Parameters
    ----------
    A : csr_matrix
        Square matrix in CSR format
    strength : ['symmetric', 'classical', 'ode', None]
        Method used to determine the strength of connection between unknowns
        of the linear system.  Method-specific parameters may be passed in
        using a tuple, e.g. strength=('symmetric',{'theta' : 0.25 }). If
        strength=None, all nonzero entries of the matrix are considered strong.
    CF : {string} : default 'RS'
        Method used for coarse grid selection (C/F splitting)
        Supported methods are RS, PMIS, PMISc, CLJP, and CLJPc
    max_levels: {integer} : default 10
        Maximum number of levels to be used in the multilevel solver.
    max_coarse: {integer} : default 500
        Maximum number of variables permitted on the coarse grid.

    References
    ----------
        Trottenberg, U., C. W. Oosterlee, and Anton Schuller.
        "Multigrid"
        San Diego: Academic Press, 2001.
        Appendix A

    """


    levels = [ multilevel_solver.level() ]

    levels[-1].A = csr_matrix(A)
    
    while len(levels) < max_levels  and levels[-1].A.shape[0] > max_coarse:
        extend_hierarchy(levels, strength=strength, CF=CF)

    return multilevel_solver(levels, **kwargs)



def extend_hierarchy(levels, strength, CF):
    def unpack_arg(v):
        if isinstance(v,tuple):
            return v[0],v[1]
        else:
            return v,{}
    
    A = levels[-1].A

    # strength of connection
    fn, kwargs = unpack_arg(strength)
    if fn == 'symmetric':
        C = symmetric_strength_of_connection(A,**kwargs)
    elif fn == 'classical':
        C = classical_strength_of_connection(A,**kwargs)
    elif fn == 'ode':
        raise NotImplementedError('ode method not supported for Classical AMG')
    elif fn is None:
        C = A
    else:
        raise ValueError('unrecognized strength of connection method: %s' % fn)


    if CF in [ 'RS', 'PMIS', 'PMISc', 'CLJP', 'CLJPc']:
        import split
        splitting = getattr(split, CF)(C)
    else:
        raise ValueError('unknown C/F splitting method (%s)' % CF)

    P = direct_interpolation(A, C, splitting)


    R = P.T.tocsr()

    levels[-1].C = C                  # strength of connection matrix
    levels[-1].P = P                  # prolongation operator
    levels[-1].R = R                  # restriction operator
    levels[-1].spliting = splitting   # C/F splitting

    levels.append( multilevel_solver.level() )

    A = R * A * P                     #galerkin operator
    levels[-1].A = A

