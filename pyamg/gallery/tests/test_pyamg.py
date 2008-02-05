#Construct a SA hierarchy and use the multilevel hierarchy as both a solver and a preconditioner
#	for a Krylov method.  

#Import libraries
#from scipy import *
from numpy import array, ones, zeros, mat, sqrt
from scipy import rand
from scipy.sparse import csr_matrix, bsr_matrix
from scipy.splinalg import spsolve
from scipy.splinalg import cg
from pyamg.sa import smoothed_aggregation_solver
from pyamg.utils import read_coord, Coord2RBM
from scipy.io import loadmat
from pylab import semilogy, title, xlabel, ylabel, legend, show
import time

ogroup_dir = "/home/jacob/Desktop/ogroup/"

#----------------------------------------- Matrices and Nullspace Vectors-----------------------------------------------------
#---------Gallery Mat
#size = 16
#Amat = csr_matrix(poisson((size,size)))


#---------OGroup Mat's
#----------------------Diffusion Type
#Amat = csr_matrix((loadmat(ogroup_dir + 'matrices/Q1_Diffusion/RotatedPi4_Ani_961.mat')).get('A'))
#Amat = csr_matrix((loadmat(ogroup_dir + 'matrices/Q1_Diffusion/Horizontal_Weak_Ani_961.mat')).get('A'))

#Bmat = mat(ones((Amat.shape[0],1)))

#Random, just for kicks...
#Bmat = mat(rand(Amat.shape[0],2))


#----------------------Elasticity
#----------------------Must also load coordinates and then convert coordinates to rigid body modes 
#Amat = bsr_matrix((loadmat(ogroup_dir + 'matrices/elasticity/IronBar/Matrix_Bar_p1_ref0.mat')).get('A'), blocksize=(3,3))
#coords = ogroup_dir + 'matrices/elasticity/IronBar/Coords_Bar_p1_ref0.txt'

Amat = bsr_matrix((loadmat(ogroup_dir + 'matrices/elasticity/Tripod_4ptTet/Matrix_Tripod_p1_ref0.mat')).get('A'), blocksize=(3,3))
coords = ogroup_dir + 'matrices/elasticity/Tripod_4ptTet/Coords_Tripod_p1_ref0.txt'

#Amat = bsr_matrix((loadmat(ogroup_dir + 'matrices/elasticity/Tripod_10ptTet/Matrix_Tripod_p1_ref0.mat')).get('A'), blocksize=(3,3))
#coords = ogroup_dir + 'matrices/elasticity/Tripod_10ptTet/Coords_Tripod_p1_ref0.txt'

X,Y,Z = read_coord(coords)
Bmat = Coord2RBM(Amat.shape[0]/Amat.blocksize[0], Amat.blocksize[0], X, Y, Z)

#Random, just for kicks...
#Bmat = mat(rand(Amat.shape[0],3))


#------------------------------------- Solver Parameters ------------------------------------------------
RHS = mat(rand(Amat.shape[0],1))
x0 = mat(zeros((Amat.shape[0],1)))
solver_tol = 1e-6
solver_maxit = 100
solver_return_residuals=True

#------------------------------------- ODE Strength Params -----------------------------------------------
k = 2
t = 1.0
proj_type =  "l2"      # "D_A" or "l2" projection used in strength measure
epsilon = 4.0
file_output = False

#-----------------------------  Energy Minimization Routine Parameters ---------------------------------
nits = 4
isSPD = True
P_tol = 1e-8

#------------------------------------- Build MG Hierarchy ------------------------------------------------
start = time.time()
sa = smoothed_aggregation_solver(Amat, B=Bmat, max_coarse=100, 
				 strength=('ode', {'epsilon' : epsilon, 't' : t, 'k' : k, 'proj_type' : proj_type, 'file_output' : file_output}),
				 smooth=('energy_min', {'SPD' : isSPD, 'num_its' : nits, 'min_tol' : P_tol, 'file_output' : file_output}))
stop = time.time()
print "Building AMG hierarchy took " + str(stop - start) + " seconds" 


#---------------------------------------- Run AMG Test ---------------------------------------------------
start = time.time()
x, r = sa.solve(b=RHS, x0=x0, tol=solver_tol, maxiter=solver_maxit, return_residuals=solver_return_residuals)
stop = time.time()
print "AMG solve took " + str(stop - start) + " seconds." 

#plot residuals
r = array(r)
semilogy(r, 'ro-', label="AMG")

#----------------------------------- Run Krylov Solver Test ----------------------------------------------
#Run Krylov Solver of Choice

def Calc_NormResidual(x, b, A, rvec):
	#Calc 2-norm of the residual and place result in the first spot after the last nonzero in rvec
	#Interestingly enough, the norm(***, 2) function provided by Python is wildly inefficient, like an
	#	order of magnitude or more slower than just calculating the 2-norm directly.  What the hell
	#	does Python do????
	indys = rvec.nonzero()[0]
	r = b - ((A*x).reshape(b.shape))
	rvec[indys.max()+1] = sqrt(r.T*r)[0,0]

r = zeros(solver_maxit+1)
r0 = RHS - Amat*x0
r[0] = (sqrt(r0.T*r0))[0,0]
del r0
callback_fcn = lambda x:Calc_NormResidual(x, RHS, Amat, r)

# cg, gmres, bicgstab
Amat.psolve = sa.psolve
start = time.time()
x,flag = cg(Amat,RHS, x0=x0, tol=solver_tol, maxiter=solver_maxit, callback=callback_fcn)
stop = time.time()
print "Krylov solve took " + str(stop - start) + "seconds."

#plot residuals
r = r[r.nonzero()[0]][1:]
semilogy(r, 'bo-', label="Krylov")

#-------------------------------------- Generate Plots ---------------------------------------------------

t = title('Residuals')
t.set_fontsize(20)
t = xlabel('Iteration')
t.set_fontsize(18)
t = ylabel('Residual')
t.set_fontsize(18)
legend()
show()

