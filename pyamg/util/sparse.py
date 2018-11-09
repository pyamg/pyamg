import numpy as np
from scipy.sparse import csr_matrix, _sparsetools
from scipy.sparse.sputils import upcast_char
import pyamg.amg_core


class csr(csr_matrix):
    """
    New CSR class

    The purpose of this class is to redefine the matvec in scipy.sparse
    """
    def _mul_vector(self, other):
        """
        Identical to scipy.sparse with an in internal call to
        pyamg.amg_core.sparse.csr_matvec
        """
        M, N = self.shape

        # output array
        result = np.zeros(M, dtype=upcast_char(self.dtype.char,
                                               other.dtype.char))

        # csr_matvec or csc_matvec
        sparse.csr_matvec(M, N, self.indptr, self.indices, self.data, other, result)

        return result


csr.__doc__ += csr_matrix.__doc__
