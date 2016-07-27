"""This demo program solves Poisson's equation

    - div grad u(x, y) = f(x, y)

on the unit square with source f given by

    f(x, y) = 10*exp(-((x - 0.5)^2 + (y - 0.5)^2) / 0.02)

and boundary conditions given by

    u(x, y) = 0        for x = 0 or x = 1
du/dn(x, y) = sin(5*x) for y = 0 or y = 1
"""


from dolfin import *
import numpy as np
try:
	import mshr
except: 
	print "Cannot import mshr"

def get_poisson(n, theta=0.0, eps=1.0, mesh=None, rand=True, coords=False):

	parameters.linear_algebra_backend = "Eigen"

	# Create mesh and define function space
	if mesh is None:
		if rand:
			try:
				domain = mshr.Rectangle(dolfin.Point(0., 0.), dolfin.Point(1., 1.))
				mesh = mshr.generate_mesh(domain, n)
			except:
				print "Cannot use mshr package, using structured mesh."
				mesh = UnitSquareMesh(n, n)
		else:
			mesh = UnitSquareMesh(n, n)

	V = FunctionSpace(mesh, "Lagrange", 1)

	# Define Dirichlet boundary on all sides
	def boundary(x):
		east = x[0] > 1.0 - DOLFIN_EPS
		west = x[0] < DOLFIN_EPS
		north = x[1] > 1.0 - DOLFIN_EPS
		south = x[1] < DOLFIN_EPS
		return east or west or north or south

	# Define boundary condition
	u0 = Constant(0.0)
	bc = DirichletBC(V, u0, boundary)

	# Define variational problem
	u = TrialFunction(V)
	v = TestFunction(V)
	f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)")

	# A = Q^TDQ, for theta rotation matrix, Q, and diagonal scaling, D = diag(1,eps)
	A = as_tensor( ((np.cos(theta)**2 + eps*np.sin(theta)**2, -(1-eps)*np.sin(theta)*np.cos(theta)),\
				   	  (-(1-eps)*np.sin(theta)*np.cos(theta), eps*np.cos(theta)**2 + np.sin(theta)**2)) )

	a = inner(A*grad(u), grad(v))*dx
	L = f*v*dx
	A, rhs = assemble_system(a, L, bcs=bc)

	# Neumann boundaries? Need to not account for some Dirichlet if use this.
	# L = f*v*dx + g*v*ds 	# 
	# g = Expression("sin(5*x[0])")

	# Form A as sparse matrix and rhs as array
	A = as_backend_type(A).sparray()
	b = as_backend_type(rhs).array()
	if coords:
		return [A,b,mesh.coordinates()]
	else:
		return [A,b]
