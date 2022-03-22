import pyamg
import numpy as np
import sparse
import os

n = 32

A = pyamg.gallery.poisson((n,), format='csr')

exact = np.zeros((n,))
exact[0] = 1
exact[-1] = 1

result = np.zeros((n,))
other = np.ones((n,))
sparse.csr_matvec(n,n, A.indptr, A.indices, A.data, other, result)
