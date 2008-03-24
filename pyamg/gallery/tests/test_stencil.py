from scipy.testing import *

from scipy import matrix, array

from pyamg.gallery.laplacian import poisson
from pyamg.gallery.stencil import *

class TestStencil(TestCase):
    def test_poisson1d(self):
        stencil = array([-1, 2, -1])

        cases = []
        cases.append( ((1,),matrix([[2]])) )
        cases.append( ((2,),matrix([[ 2,-1],
                                    [-1, 2]])) )
        cases.append( ((4,),matrix([[ 2,-1, 0, 0],
                                    [-1, 2,-1, 0],
                                    [ 0,-1, 2,-1],
                                    [ 0, 0,-1, 2]])) )
      
        for grid, expected in cases:
            result   = stencil_grid(stencil, grid).todense()
            assert_equal(result, expected)

    def test_poisson2d_5pt(self):
        stencil = array([[ 0,-1, 0],
                         [-1, 4,-1],
                         [ 0,-1, 0]])

        cases = []
        cases.append( ((1,1), matrix([[4]])) )
        cases.append( ((2,1), matrix([[ 4,-1],
                                      [-1, 4]])) )
        cases.append( ((1,2), matrix([[ 4,-1],
                                      [-1, 4]])) )
        cases.append( ((1,3), matrix([[ 4,-1, 0],
                                      [-1, 4,-1],
                                      [ 0,-1, 4]])) )
        cases.append( ((2,2), matrix([[ 4,-1,-1, 0],
                                      [-1, 4, 0,-1],
                                      [-1, 0, 4,-1],
                                      [ 0,-1,-1, 4]])) )
      
        for grid, expected in cases:
            result = stencil_grid(stencil, grid).todense()
            assert_equal(result, expected)
    
    def test_poisson2d_9pt(self):
        stencil = array([[-1,-1,-1],
                         [-1, 8,-1],
                         [-1,-1,-1]])

        cases = []
        cases.append( ((1,1), matrix([[8]])) )
        cases.append( ((2,1), matrix([[ 8,-1],
                                      [-1, 8]])) )
        cases.append( ((1,2), matrix([[ 8,-1],
                                      [-1, 8]])) )
        cases.append( ((1,3), matrix([[ 8,-1, 0],
                                      [-1, 8,-1],
                                      [ 0,-1, 8]])) )
        cases.append( ((2,2), matrix([[ 8,-1,-1,-1],
                                      [-1, 8,-1,-1],
                                      [-1,-1, 8,-1],
                                      [-1,-1,-1, 8]])) )
        
        for grid, expected in cases:
            result = stencil_grid(stencil, grid).todense()
            assert_equal(result, expected)

    def test_poisson3d_7pt(self):
        stencil = array([[[ 0, 0, 0],
                          [ 0,-1, 0],
                          [ 0, 0, 0]],
                         [[ 0,-1, 0],
                          [-1, 6,-1],
                          [ 0,-1, 0]],
                         [[ 0, 0, 0],  
                          [ 0,-1, 0],
                          [ 0, 0, 0]]])

        cases = []
        cases.append( ((1,1,1), matrix([[ 6]])) )
        cases.append( ((2,1,1), matrix([[ 6,-1],
                                        [-1, 6]])) )
        cases.append( ((2,2,1), matrix([[ 6,-1,-1, 0],
                                        [-1, 6, 0,-1],
                                        [-1, 0, 6,-1],
                                        [ 0,-1,-1, 6]])) )
        cases.append( ((2,2,2), matrix([[ 6,-1,-1, 0,-1, 0, 0, 0],
                                        [-1, 6, 0,-1, 0,-1, 0, 0],
                                        [-1, 0, 6,-1, 0, 0,-1, 0],
                                        [ 0,-1,-1, 6, 0, 0, 0,-1],
                                        [-1, 0, 0, 0, 6,-1,-1, 0],
                                        [ 0,-1, 0, 0,-1, 6, 0,-1],
                                        [ 0, 0,-1, 0,-1, 0, 6,-1],
                                        [ 0, 0, 0,-1, 0,-1,-1, 6]])) )
        
        for grid, expected in cases:
            result = stencil_grid(stencil, grid).todense()
            assert_equal(result, expected)

if __name__ == '__main__':
    nose.run(argv=['', __file__])
