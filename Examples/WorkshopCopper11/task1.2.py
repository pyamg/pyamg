from scipy.sparse import *
# csr_matrix? # see documentation
from numpy import array
row = array([0,0,1,2,2,2])
col = array([0,2,2,0,1,2])
data = array([1,2,3,4,5,6])
B = csr_matrix( (data,(row,col)), shape=(3,3) )
# B.<tab> # see documentation
print(B.todense())
B = B.tocoo()
