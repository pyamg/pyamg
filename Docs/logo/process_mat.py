from scipy.io import loadmat, savemat
from pydec import *

X = loadmat('pyamg.mat')
del X['__version__']
del X['__header__']
del X['__globals__']

sc = simplicial_complex(X['vertices'],X['elements'].astype('intc'))

d = sc[0].d
M = whitney_innerproduct(sc,1)
A = d.T.tocsr() * M * d

X['A'] = A
X['B'] = ones((A.shape[0],1))

savemat('pyamg.mat',X)
