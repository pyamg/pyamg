import numpy as np
import pyamg
import relaxation

stencil = [ [-1,-1,-1],[-1,8,-1],[-1,-1,-1] ]
n = 100

class exfloat:
    def __init__(self):
        self.A = pyamg.gallery.stencil_grid(stencil, (n,n), dtype=np.float32, format='csr')
        N = self.A.shape[0]
        self.b = np.random.randn(N).astype(np.float32)
        self.x = np.random.randn(N).astype(np.float32)

class exdouble:
    def __init__(self):
        self.A = pyamg.gallery.stencil_grid(stencil, (n,n), dtype=np.float64, format='csr')
        N = self.A.shape[0]
        self.b = np.random.randn(N).astype(np.float64)
        self.x = np.random.randn(N).astype(np.float64)

class exdoublecomplex:
    def __init__(self):
        self.A = pyamg.gallery.stencil_grid(stencil, (n,n), dtype=np.complex128, format='csr')
        N = self.A.shape[0]
        self.b = np.random.randn(N).astype(np.float64) + np.random.randn(N).astype(np.float64)*1j
        self.x = np.random.randn(N).astype(np.float64) + np.random.randn(N).astype(np.float64)*1j

cases = [exdoublecomplex()]

for c in cases:
    A = c.A
    x = c.x
    b = c.b
    A, x, b = pyamg.relaxation.relaxation.make_system(A, x, b, formats=['csr'])
    N = A.shape[0]
    xnew = x.copy()
    pyamg.amg_core.gauss_seidel(A.indptr, A.indices, A.data, x, b, 0, N, 1)
    relaxation.gauss_seidel(A.indptr, A.indices, A.data, xnew, b, 0, N, 1)
    print(np.abs(x - xnew).max())
