from numpy import array, zeros, matrix, mat
from scipy.sparse import csr_matrix, isspmatrix_csr, bsr_matrix, isspmatrix_bsr, spdiags
from scipy.linalg import svd, norm
#from pyamg.sa import spdiags
from pyamg.utils import UnAmal, BSR_Row_AddVect
from scipy.io import loadmat, savemat

########################################################################################################
#	Helper function for the energy minimization prolongator generation routine

def Satisfy_Constraints(U, rowptr, cols, B, BtBinv):
#	Input:	U:	Matrix to operate on
#		rowptr:	CSR rowptr array that corresponds to the sparsity pattern in U
#		cols:	CSR cols array that corresponds to the sparsity pattern in U
#		B:	Near nullspace vectors
#		BtBinv: Local inv(B'*B) matrices for each dof, i.  
#		
#	Output:	Updated U, so that U*B = 0.  Update is computed by orthogonall (in 2-norm)
#		projecting out the components of span(B) in U in a row-wise fashion

	Nfine = U.shape[0]
	csrflag = isspmatrix_csr(U)

	UB = U*mat(B)

	#Project out U's components in span(B) row-wise
	for i in range(Nfine):
		rowstart = rowptr[i]
		rowend = rowptr[i+1]
		length = rowend - rowstart
		colindx = cols[rowstart:rowend]
		
		if(length != 0):
			Bi = B[colindx,:]
			UBi = UB[i,:]

			update_local = (Bi*(BtBinv[i]*UBi.T)).T
			if(csrflag):
				U.data[rowstart:rowend] = U.data[rowstart:rowend] - update_local
			else:
				BSR_Row_AddVect(U, i, -update_local)

########################################################################################################


def sa_energy_min(A, T, Atilde, B, SPD=True, num_its=4, min_tol=1e-8, file_output=False):
#	Input:	A:		Current operator at this level, could be CSR or BSR
#		Atilde: 	Strength of connection matrix
#		T:		Tentative Prolongator.  If A is nxn, then T is nxm, which m<n.  Must be BSR.
#					Ideally, the row_blocksize is the same as A's block size.
#		B:		Near nullspace modes for coarse grid.  If T is nxm, then B is mxk, for k near nullspace modes.
#		SPD:		Booolean denoting symmetric positive-definiteness
#		num_its:	Number of energy minimization steps to apply to the prolongator
#		min_tol:	Minimization tolerance
#		file_output:  	Optional diagnostic file output of matrices
#
#
#	Output:	P:	Prolongator with energy minimized according to the algorithm from Mandel's
# 			Paper, "Energy Optimization of Algebraic Multigrid Bases."
	
	#====================================================================
	#Test Inputs
	if( not(isinstance(num_its,int)) ):
		raise TypeError("\nCalling sa_energy_min Incorrectly.  "
				"Number of minimization iterations applied to P is \"num_its\", where \"num_its\" must be an integer.\n\n")
	if( num_its < 0 ):
		raise TypeError("\nCalling sa_energy_min Incorrectly.  ",  
				"Number of minimization iterations applied to P is \"num_its\", where \"num_its\" >= 0.\n\n")
	if( min_tol > 1  ):
		raise TypeError("\nCalling sa_energy_min Incorrectly.  0 < \"min_tol\" < 1.\n\n")
	csrflag = isspmatrix_csr(A)
	if( not(csrflag) and (isspmatrix_bsr(A) == False)):
		raise TypeError("sa_energy_min routine requires a CSR or BSR operator.  Aborting.\n")
	if(isspmatrix_bsr(T) == False):
		raise TypeError("sa_energy_min routine requires a BSR tentative prolongator.  Aborting.\n")
	if( not(csrflag) and (T.blocksize[0] != A.blocksize[0]) ):
		print "Warning, T's row-blocksize should be the same as A's blocksize.\n"
	#====================================================================
	
	
	#====================================================================
	# Retrieve problem information
	Nfine = T.shape[0]
	Ncoarse = T.shape[1]
	NullDim = B.shape[1]
	if(csrflag):
		numPDEs = 1
	else:
		#Number of PDEs per point is defined implicitly by block size
		numPDEs = A.blocksize[0]
	#====================================================================
	

	#====================================================================
	#Optional file output for diagnostic purposes
	if(file_output == True):
		savemat('Amat', { 'Amat' : A.toarray() } ) 
		savemat('Atilde', { 'Atilde' : Atilde.toarray() } )
		savemat('P', { 'P' : T.toarray() } ) 
		savemat('ParamsEnMin', {'nPDE' : numPDEs, 'Nits' : num_its, 'SPD' : SPD } ) 
		savemat('Bone', { 'Bone' : array(B) } )
	#====================================================================
	

	#====================================================================
	# Construct Dinv and Unamalgate Atilde if (numPDEs > 1)
	D = A.diagonal();
	if(csrflag):
		Dinv    = spdiags( [1.0/D], [0], Nfine, Nfine, format = 'csr')
	else:
		Dinv    = spdiags( [1.0/D], [0], Nfine, Nfine, format = 'bsr')
		Dinv = bsr_matrix(Dinv, blocksize=A.blocksize)
	
	#If A is BSR, then Atilde needs to be "UnAmalgamated" to generate prolongator
	#	sparsity pattern.  Sparsity pattern is generated through matrix multiplication
	if(not(csrflag)):
		#UnAmal returns a BSR matrix, so the mat-mat will be a BSR mat-mat.  Unfortunately, 
		#	we need column indices for Sparsity_Pattern, so convert to CSR and store 
		#	permanently only the rowptr and col arrays.
		Sparsity_Pattern = UnAmal(Atilde, numPDEs).__abs__()*T.__abs__()
		TempMat = csr_matrix(Sparsity_Pattern)
		rowptr = TempMat.indptr
		cols = TempMat.indices
		del TempMat
		Sparsity_Pattern.data[:,:,:] = 1.0
	else:
		#Sparsity_Pattern will be CSR as Atilde is CSR.  This means T will be converted to
		#	CSR, but we need Sparsity_Pattern in CSR.  T is converted back to BSR at the end.
		blocks = T.blocksize
		T = T.tocsr()
		Sparsity_Pattern = Atilde.__abs__()*T.__abs__()
		Sparsity_Pattern.data[:] = 1.0
		
		#This isn't a deep copy...
		rowptr = Sparsity_Pattern.indptr
		cols = Sparsity_Pattern.indices
	#====================================================================
	
	
	#====================================================================
	#Construct array of inv(Bi'Bi), where Bi is B restricted to row i's sparsity pattern in 
	#	Sparsity Pattern.  This array is used multiple times in the Satisfy_Constraints routine.
	preall = zeros((NullDim,NullDim))
	BtBinv = [matrix(preall,copy=True) for i in range(Nfine)]
	del preall
	B = mat(B)
	for i in range(Nfine):
		rowstart = rowptr[i]
		rowend = rowptr[i+1]
		length = rowend - rowstart
		colindx = cols[rowstart:rowend]
	
		if(length != 0):
			Bi = B[colindx,:]
			#Calculate SVD as system may be singular
			U,Sigma,VT = svd(Bi.T*Bi)
			
			#Filter Sigma and calculate inv(Sigma)
			if(abs(Sigma[0]) < 1e-10):
				Sigma[:] = 0.0
			else:
				#Zero out "numerically" zero singular values
				#	Efficiency TODO -- would this be faster in a loop that starts from the
				#	back of Sigma and assumes Sigma is sorted?  Experiments say no.
				Sigma =  (Sigma/Sigma[0]).__abs__().__gt__(1e-8)*Sigma
					
				#Test for any zeros in Sigma
				if(Sigma[NullDim-1] == 0.0):
					#Truncate U, VT and Sigma w.r.t. zero entries in Sigma
					indys = Sigma.nonzero()[0]
					Sigma = Sigma[indys]
					U = U[:,indys]
					VT = VT[indys,:]
						
				#Invert nonzero sing values
				Sigma = 1.0/Sigma
			
			#Calculate inverse
			BtBinv[i] = mat(VT.T)*mat(Sigma.reshape(Sigma.shape[0],1)*U.T)
	#====================================================================
	

	#====================================================================
	#Prepare for Energy Minimization
	#Calculate initial residual
	if(csrflag):
		R = -A*T
	else:
		R = -A*T
	
	#Enforce constraints on R.  First the sparsity pattern, then the nullspace vectors.
	#------ A numerical sleight of hand is also needed.  R must have nonzeros at the 
	#	locations of Sparsity_Pattern.  If this is not so, then Satisfy_Constraints
	#	would be much more expensive.  Hence R += 1e-20*Sparsity_Pattern is a quick 
	#	dirty way around this.  1e-20 is below machine epsilon for float64, so this
	#	addition shouldn't affect the correctness of the algorithmic.  Its just an
	#	ugly fix.
	R = R.multiply(Sparsity_Pattern)
	R = R + 1e-20*Sparsity_Pattern
	Satisfy_Constraints(R, rowptr, cols, B, BtBinv)
	
	#Calculate max norm of the residual
	resid = max(R.data.flatten().__abs__())
	print "Energy Minimization of Prolongator --- Iteration 0 --- r = " + str(resid)
	#====================================================================
	

	#====================================================================
	#Iteratively minimize the energy of T subject to the constraints of Sparsity_Pattern
	#	and maintaining T's effect on B, i.e. T*B = (T+Update)*B, i.e. Update*B = 0 
	i = 0
	if(SPD):
		#Apply CG
		while( (i < num_its) and (resid > min_tol) ):
			#Apply diagonal preconditioner
			Z = Dinv*R
	
			#Frobenius innerproduct of (R,Z) = sum(rk.*zk)
	 		newsum = (R.multiply(Z)).sum()
				
			#P is the search direction, not the prolongator, which is T.	
			if(i == 0):
				P = Z
			else:
				beta = newsum/oldsum
				P = Z + beta*P
			oldsum = newsum
	
			#Calculate new direction and enforce constraints
			AP = A*P
			AP = AP.multiply(Sparsity_Pattern)
			Satisfy_Constraints(AP, rowptr, cols, B, BtBinv)
			
			#Frobenius innerproduct of (P, AP)
			alpha = newsum/(P.multiply(AP)).sum()
	
			#Update the prolongator, T
			T = T + alpha*P	
	
			#Update residual
			R = R - alpha*AP
			
			i += 1
			resid = max(R.data.flatten().__abs__())
			print "Energy Minimization of Prolongator --- Iteration " + str(i) + " --- r = " + str(resid)
			
	else:
		#Apply min-res to the nonsymmetric system
		while( (i < num_its) and (resid > min_tol) ):
	
			#P is the search direction, not the prolongator
			P = A*R
	
			#Enforce constraints on P
			P = P.multiply(Sparsity_Pattern)
			Satisfy_Constraints(P, rowptr, cols, B, BtBinv)
	
			#Frobenius innerproduct of (P, R)
			numer = (P.multiply(R)).sum()
			
			#Frobenius innerproduct of (P, P)
			denom = (P.multiply(P)).sum()
	
			alpha = numer/denom
	
			#Update prolongator
			T = T + alpha*R
	
			#Update residual
			R = R - alpha*P
			
			i += 1
			resid = max(R.data.flatten().__abs__())
			print "Energy Minimization of Prolongator --- Iteration " + str(i) + " --- r = " + str(resid)
	#====================================================================
	
	
	#====================================================================
	#Make sure T is in correct block format.
	if(csrflag):
		T = T.tobsr(blocksize=blocks)
	#====================================================================
	

	#====================================================================
	#Optional file output for diagnostic purposes
	if(file_output == True):
		savemat('Ppyth', { 'Ppyth' : T.toarray() } ) 
	#====================================================================
	

	return T
	
	
