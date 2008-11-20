from pyamg.testing import *

import numpy
from numpy import array, ones, arange, empty, array_split

from scipy.sparse import csr_matrix, spdiags

from pyamg.gallery import poisson, load_example
from pyamg.strength import symmetric_strength_of_connection
from pyamg.aggregation.aggregate import standard_aggregation


class TestAggregate(TestCase):
    def setUp(self):
        self.cases = []

        # random matrices
        numpy.random.seed(0)
        for N in [2,3,5]:
            self.cases.append( csr_matrix(rand(N,N)) )

        # poisson problems in 1D and 2D
        for N in [2,3,5,7,10,11,19]:
            self.cases.append( poisson( (N,), format='csr') )
        for N in [2,3,5,7,10,11]:
            self.cases.append( poisson( (N,N), format='csr') )

        for name in ['knot','airfoil','bar']:
            ex = load_example(name)
            self.cases.append( ex['A'].tocsr() )


    def test_standard_aggregation(self):
        for A in self.cases:
            S = symmetric_strength_of_connection(A)
            
            expected = reference_standard_aggregation(S)
            result   = standard_aggregation(S)

            assert_equal( (result - expected).nnz, 0 )
    
        # S is diagonal - no DoFs aggregated
        S = spdiags([[1,1,1,1]],[0],4,4,format='csr')
        result   = standard_aggregation(S)
        expected = array([[0],[0],[0],[0]])
        assert_equal(result.todense(),expected)
        

class TestComplexAggregate(TestCase):
    def setUp(self):
        self.cases = []

        # poisson problems in 2D
        for N in [2,3,5,7,10,11]:
            A = poisson( (N,N), format='csr'); A.data = A.data + 0.001j*rand(A.nnz)
            self.cases.append(A)

    def test_standard_aggregation(self):
        for A in self.cases:
            S = symmetric_strength_of_connection(A)
            
            expected = reference_standard_aggregation(S)
            result   = standard_aggregation(S)

            assert_equal( (result - expected).nnz, 0 )


################################################
##   reference implementations for unittests  ##
################################################

# note that this method only tests the current implementation, not
# all possible implementations
def reference_standard_aggregation(C):
    S = array_split(C.indices,C.indptr[1:-1])

    n = C.shape[0]

    R = set(range(n))
    j = 0

    aggregates    = empty(n,dtype=C.indices.dtype)
    aggregates[:] = -1

    # Pass #1
    for i,row in enumerate(S):
        Ni = set(row) | set([i])

        if Ni.issubset(R):
            R -= Ni
            for x in Ni:
                aggregates[x] = j
            j += 1

    # Pass #2
    Old_R = R.copy()
    for i,row in enumerate(S):
        if i not in R: continue

        for x in row:
            if x not in Old_R:
                aggregates[i] = aggregates[x]
                R.remove(i)
                break

    # Pass #3
    for i,row in enumerate(S):
        if i not in R: continue
        Ni = set(row) | set([i])

        for x in Ni:
            if x in R:
                aggregates[x] = j
            j += 1

    assert(len(R) == 0)

    Pj = aggregates
    Pp = arange(n+1)
    Px = ones(n)

    return csr_matrix((Px,Pj,Pp))


