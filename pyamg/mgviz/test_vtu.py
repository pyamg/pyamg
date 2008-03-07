""" Using the examples in the examples directory"""

#import os
#from scipy import rand, array
#from scipy.io import loadmat
#from pyamg.gallery import load_example
from scipy.testing import *
from numpy import ones, zeros, linspace, kron, uint32
from mgviz import write_vtu

class TestWriteVtu(TestCase):
    def setUp(self):
        nx=40
        ny=30
        Vert = zeros((nx*ny,2))
        X = kron(ones((1,ny)),linspace(0,1,nx))
        Y = kron(linspace(0,1,ny),ones((1,nx)))
        Vert[:,0]=X
        Vert[:,1]=Y
        Nel = (nx-1)*(ny-1)*2
        E2V = zeros((Nel,3),dtype=uint32)
        k=0
        for i in range(0,ny-1):
            for j in range(0,nx-1):
                E2V[k,:]   = [j+i*nx,j+1+(i+1)*nx,j+(i+1)*nx]
                E2V[k+1,:] = [j+i*nx,(j+1)+i*nx,(j+1)+(i+1)*nx]
                k+=2
        print Vert
        print E2V
        Cells = {'5':E2V}
        write_vtu(Vert,Cells,'test.vtu',None,None)

    def test_default(self):
        assert_equal( 1,1)
    

#d = loadmat('Airfoil_p1_ref1.mat')
#agg = loadmat('Airfoil_p1_ref1_aggs.mat')

#ex = load_example('airfoil')
#
#E2V  = ex['elements']
#Vert = ex['vertices']
#
#cdata = ({'5':random.random((E2V.shape[0],1))}, {'5':2*random.random((E2V.shape[0],1))})
#data = zeros((Vert.shape[0],1))
#data[5:10]=1
#pdata=concatenate((random.random((Vert.shape[0],1)),data),1)
#

if __name__ == '__main__':
    nose.run(argv=['', __file__])
