# This example uses Dolfin of the Fenics Project:
# http://www.fenics.org/wiki/DOLFIN
# to construct a FE solution the Poisson problem on the Unit Square.
#
# PyAMG is used to solve the resulting system.  Data is not copied when
# constructing the scipy.sparse matrix
#
from dolfin import *

from scipy.sparse import csr_matrix
from pyamg import smoothed_aggregation_solver
from pylab import figure, show, semilogy

############################################################
# Dolfin
dolfin_set("linear algebra backend","uBLAS")

mesh = UnitSquare(75, 75)
V = FunctionSpace(mesh, "CG", 1)

# Define Dirichlet boundary (x = 0 or x = 1)
class DirichletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS
u0 = Constant(mesh, 0.0)
bc = DirichletBC(V, u0, DirichletBoundary())

v = TestFunction(V)
u = TrialFunction(V)
f = Function(V, "500.0 * exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)")
a = dot(grad(v), grad(u))*dx
L = v*f*dx

A, rhs = assemble_system(a,L,bc,mesh)

############################################################
# PyAMG
(row,col,data) = A.data()   # get sparse data
n = A.size(0)
Asp = csr_matrix( (data,col.view('intc'),row.view('intc')), shape=(n,n))
b = rhs.data()

residuals = []

ml = smoothed_aggregation_solver(Asp,max_coarse=10)
x = ml.solve(b,tol=1e-10,accel='cg',residuals=residuals)

residuals = residuals/residuals[0]
print ml

figure(2)
semilogy(residuals)
show()
