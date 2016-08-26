from __future__ import print_function
from dolfin import *
import pyamg
import numpy as np
import time


def _build_nullspace(V, x):
    """Function to build null space for 3D elasticity"""

    # Create list of vectors for null space
    nullspace_basis = [x.copy() for i in range(6)]

    # Build translational null space basis
    V.sub(0).dofmap().set(nullspace_basis[0], 1.0)
    V.sub(1).dofmap().set(nullspace_basis[1], 1.0)
    V.sub(2).dofmap().set(nullspace_basis[2], 1.0)

    # Build rotational null space basis
    V.sub(0).dofmap().set_x(nullspace_basis[3], -1.0, 1, V.mesh())
    V.sub(1).dofmap().set_x(nullspace_basis[3],  1.0, 0, V.mesh())
    V.sub(0).dofmap().set_x(nullspace_basis[4],  1.0, 2, V.mesh())
    V.sub(2).dofmap().set_x(nullspace_basis[4], -1.0, 0, V.mesh())
    V.sub(2).dofmap().set_x(nullspace_basis[5],  1.0, 1, V.mesh())
    V.sub(1).dofmap().set_x(nullspace_basis[5], -1.0, 2, V.mesh())

    for x in nullspace_basis:
        x.apply("insert")

    # Create vector space basis and orthogonalize
    # basis = VectorSpaceBasis(nullspace_basis)
    # basis.orthonormalize()

    return nullspace_basis


def get_elasticity_bar(nx, ny, nz):

    parameters.linear_algebra_backend = "Eigen"

    # Load mesh and define function space
    corner1 = Point (0.0, 0.0, 0.0)
    corner2 = Point (0.01, 0.1, 0.01)
    mesh = BoxMesh(corner1, corner2, nx, ny, nz)

    def leftside(x, on_boundary):
        return x[1] < DOLFIN_EPS and on_boundary

    # Load increasing toward the end (y-direction) in the down direction (z)
    f = Expression(("0.0", "0.0", "-val*x[1]*x[1]/100.0"), val=1e8)

    # Elasticity parameters
    # E = 117.0e9     # young's modulus for copper (GPa)
    # nu = 0.33       # poisson ratio for copper
    E = 0.01e9      # rubber -- really hard for amg
    nu = 0.499      # rubber
    mu = E/(2.0*(1.0 + nu))
    lmbda = E*nu/((1.0 + nu)*(1.0 - 2.0*nu))

    # Stress computation
    def sigma(v):
        return 2.0*mu*sym(grad(v)) + lmbda*tr(sym(grad(v)))*Identity(len(v))

    # Create function space
    V = VectorFunctionSpace(mesh, "Lagrange", 1)

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(sigma(u), grad(v))*dx
    L = inner(f, v)*dx

    # Set up boundary condition on inner surface
    c = Constant((0.0, 0.0, 0.0))
    bc = DirichletBC(V, c, leftside)

    # Assemble system, applying boundary conditions and preserving
    # symmetry)
    A, rhs = assemble_system(a, L, bc)

    # Create solution function
    u = Function(V)
    A = as_backend_type(A).sparray()
    b = as_backend_type(rhs).array()
    
    # Form null space vectors
    null_space = _build_nullspace(V, u.vector())
    B = np.zeros((A.shape[0], len(null_space)))
    for i in range(len(null_space)):
        B[:, i] = null_space[i].array()
    x0 = np.random.rand(A.shape[0])

    A = A.tobsr(blocksize=(3, 3))

    return [A, b, B]

