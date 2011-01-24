# This example uses Dolfin version 0.9.7 of the Fenics Project:
# http://www.fenicsproject.org/
# to construct a FE solution the Poisson problem on the Unit Square.
#
# PyAMG is used to solve the resulting system.  Data is not copied when
# constructing the scipy.sparse matrix


############################################################
# Part I: Setup problem with Dolfin
try:
    from dolfin import *
except ImportError:
    raise ImportError('Problem with Dolfin Installation')

parameters.linear_algebra_backend = "uBLAS"

# Define mesh, function space
mesh = UnitSquare(75, 35)
V = FunctionSpace(mesh, "CG", 1)

# Define basis and bilinear form
u = TrialFunction(V)
v = TestFunction(V)
a = dot(grad(v), grad(u))*dx
f = Expression('500.0 * exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)')
L = v*f*dx

# Define Dirichlet boundary (x = 0 or x = 1)
class DirichletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS
u0 = Constant(0.0)
bc = DirichletBC(V, u0, DirichletBoundary())

A, rhs = assemble_system(a,L,bcs=bc)
############################################################



############################################################
# Part II: Solve with PyAMG
from scipy.sparse import csr_matrix
from pyamg import smoothed_aggregation_solver
(row,col,data) = A.data()   # get sparse data
n = A.size(0)
Asp = csr_matrix( (data,col.view('intc'),row.view('intc')), shape=(n,n))
b = rhs.data()

ml = smoothed_aggregation_solver(Asp,max_coarse=10)
residuals = []
x = ml.solve(b,tol=1e-10,accel='cg',residuals=residuals)

residuals = residuals/residuals[0]
print ml
############################################################



############################################################
# Part III: plot
import pylab
pylab.figure(2)
pylab.semilogy(residuals)
pylab.show()
############################################################
