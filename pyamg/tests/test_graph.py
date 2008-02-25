from scipy.testing import *

import numpy
from numpy import ones, eye, zeros, bincount, empty, asarray
from scipy import rand
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix

from pyamg.gallery import poisson, load_example
from pyamg.graph import *

class TestGraph(TestCase):
    def setUp(self):
        cases = []

        cases.append( zeros((1,1)) )
        cases.append( zeros((2,2)) )
        cases.append( zeros((8,8)) )
        cases.append( ones((2,2)) - eye(2) )
        cases.append( poisson( (5,) ) )
        cases.append( poisson( (5,5) ) )
        cases.append( poisson( (11,11) ) )
        cases.append( poisson( (5,5,5) ) )
        for name in ['airfoil','bar','knot']:
            cases.append( load_example(name)['A'] )

        cases = [ coo_matrix(G) for G in cases ]

        # convert to expected format
        # - remove diagonal entries
        # - all nonzero values = 1
        for G in cases:
            mask = G.row != G.col
            
            G.row     = G.row[mask]
            G.col     = G.col[mask]
            G.data    = G.data[mask]
            G.data[:] = 1

        self.cases = cases        

    def test_maximal_independent_set(self):
        # test that method works with diagonal entries
        assert_equal( maximal_independent_set(eye(2)), [1, 1] )

        for algo in ['serial','parallel']:
            for G in self.cases:
                mis = maximal_independent_set(G, algo=algo)
                
                # no MIS vertices joined by an edge
                if G.nnz > 0:
                    assert( (mis[G.row] + mis[G.col]).max() <= 1 )

                # all non-set vertices have set neighbor
                assert( (mis + G*mis).min() == 1 )
    
    def test_vertex_coloring(self):
        # test that method works with diagonal entries
        assert_equal( vertex_coloring(eye(1)), [0] )
        assert_equal( vertex_coloring(eye(3)), [0,0,0] )
        assert_equal( sorted(vertex_coloring(ones((3,3)))), [0,1,2] )

        for algo in ['serial','parallel']:
            for G in self.cases:
                c = vertex_coloring(G, algo=algo)

                # no colors joined by an edge
                assert( (c[G.row] != c[G.col]).all() )

                # all colors up to K occur at least once
                assert( (bincount(c) > 0).all() )

    def test_bellman_ford(self):
        numpy.random.seed(0)

        for G in self.cases:
            G.data = rand(G.nnz)
            N = G.shape[0]

            for n_seeds in [ N/20, N/10, N-2, N]:
                if n_seeds > G.shape[0] or n_seeds < 1: 
                    continue
    
                seeds = numpy.random.permutation(N)[:n_seeds]
                D_expected,S_expected = reference_bellman_ford(G, seeds) 
                D_result,S_result     = bellman_ford(G, seeds)
    
                assert_equal(D_result, D_expected)
                assert_equal(S_result, S_expected)

    def test_lloyd_cluster(self):
        numpy.random.seed(0)

        for G in self.cases:
            G.data = rand(G.nnz)
            
            for n_seeds in [5]:
                if n_seeds > G.shape[0]: 
                    continue
    
                distances,clusters,centers = lloyd_cluster(G, n_seeds)
    

from pyamg.graph import max_value
def reference_bellman_ford(G,seeds):
    G = G.tocoo()
    N = G.shape[0]
   
    seeds = asarray(seeds, dtype='intc')

    distances        = empty( N, dtype=G.dtype )
    distances[:]     = max_value(G.dtype)
    distances[seeds] = 0

    nearest_seed        = empty(N, dtype='intc')
    nearest_seed[:]     = -1
    nearest_seed[seeds] = seeds

    while True:
        update = False

        for (i,j,v) in zip(G.row,G.col,G.data):

            if distances[j] + v < distances[i]:
                update = True
                distances[i]    = distances[j] + v
                nearest_seed[i] = nearest_seed[j]

        if not update:
            break

    return (distances, nearest_seed) 


if __name__ == '__main__':
    nose.run(argv=['', __file__])
