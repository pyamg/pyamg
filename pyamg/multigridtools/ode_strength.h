#ifndef ODE_STRENGTH_H
#define ODE_STRENGTH_H

#include <iterator>
#include <algorithm>
#include <cmath>
#include <limits>

#include "smoothed_aggregation.h"

/*
 * Return a filtered strength-of-connection matrix by applying a drop tolerance
 *  Strength values are assumed to be "distance"-like, i.e. the smaller the 
 *  value the stronger the connection
 *
 *    An off-diagonal entry A[i,j] is a strong connection iff
 *
 *            S[i,j] <= epsilon * min( S[i,k] )   where k != i
 *  
 *   Also, set the diagonal to 1.0, as each node is perfectly close to itself
 *
 * Parameters
 * ----------
 * n_row : {int}
 *      Dimension of matrix, S
 * epsilon : {float}
 *      Drop tolerance
 * Sp : {int array}
 *      Row pointer array for CSR matrix S
 * Sj : {int array}
 *      Col index array for CSR matrix S
 * Sx : {float|complex array}
 *      Value array for CSR matrix S
 *
 * Returns
 * -------
 * Sx is modified in place such that the above dropping strategy has been applied
 * There will be explicit zero entries for each weak connection
 *
 * Notes
 * -----
 * Principle calling routines are strength of connection routines, e.g.
 * ode_strength_of_connection(...) in strength.py.  It is used to apply 
 * a drop tolerance.
 *
 * Examples
 * --------
 * >>> from scipy.sparse import csr_matrix
 * >>> from pyamg.multigridtools import apply_distance_filter
 * >>> from scipy import array
 * >>> # Graph in CSR where entries in row i represent distances from dof i
 * >>> indptr = array([0,3,6,9])
 * >>> indices = array([0,1,2,0,1,2,0,1,2])
 * >>> data = array([1.,2.,3.,4.,1.,2.,3.,9.,1.])
 * >>> S = csr_matrix( (data,indices,indptr), shape=(3,3) )
 * >>> print "Matrix BEfore Applying Filter\n" + str(S.todense())
 * >>> apply_distance_filter(3, 1.9, S.indptr, S.indices, S.data)
 * >>> print "Matrix AFter Applying Filter\n" + str(S.todense())
 */          
template<class I, class T>
void apply_distance_filter(const I n_row,
                           const T epsilon,
                           const I Sp[],    const I Sj[], T Sx[])
{
    //Loop over rows
    for(I i = 0; i < n_row; i++)
    {
        const I row_start = Sp[i];
        const I row_end   = Sp[i+1];
    
        //Find min for row i
        T min_offdiagonal = std::numeric_limits<T>::max();
        for(I jj = row_start; jj < row_end; jj++){
            if(Sj[jj] != i){
                min_offdiagonal = std::min(min_offdiagonal,Sx[jj]);
            }
        }

        //Apply drop tol to row i
        const T threshold = epsilon*min_offdiagonal;
        for(I jj = row_start; jj < row_end; jj++){
            if(Sj[jj] == i){
                Sx[jj] = 1.0;  //Set diagonal to 1.0 
            } else if(Sx[jj] >= threshold){
                Sx[jj] = 0.0;  //Set weak connection to 0.0
            }
        } //end for

    }
}

/*
 *  Given a BSR with num_blocks stored, return a linear array of length 
 *  num_blocks, which holds each block's smallest, nonzero, entry
 *  
 * Parameters
 * ----------
 * n_blocks : {int}
 *      Number of blocks in matrix
 * blocksize : {int}
 *      Size of each block
 * Sx : {float|complex array}
 *      Block data structure of BSR matrix, S
 *      Sx is (n_blocks x blocksize) in length
 * Tx : {float|complex array}
 *      modified inplace for output
 *
 * Returns
 * ------
 * Tx[i] modified in place, it holds 
 * the minimum nonzero value of block i of S
 *
 * Notes
 * -----
 * Principle calling routine is ode_strength_of_connection(...) in strength.py.  
 * In that routine, it is used to assign a strength of connection value between 
 * supernodes by setting the strength value to be the minimum nonzero in a block.
 *
 * Examples
 * --------
 * >>> from scipy.sparse import bsr_matrix, csr_matrix
 * >>> from pyamg.multigridtools import min_blocks
 * >>> from numpy import zeros, array, ravel, round
 * >>> from scipy import rand
 * >>> row  = array([0,2,4,6])
 * >>> col  = array([0,2,2,0,1,2])
 * >>> data = round(10*rand(6,2,2), decimals=1)
 * >>> S = bsr_matrix( (data,col,row), shape=(6,6) )
 * >>> T = zeros(data.shape[0])
 * >>> print "Matrix BEfore\n" + str(S.todense())
 * >>> min_blocks(6, 4, ravel(S.data), T)
 * >>> S2 = csr_matrix((T, S.indices, S.indptr), shape=(3,3))
 * >>> print "Matrix AFter\n" + str(S2.todense())
 */          
template<class I, class T>
void min_blocks(const I n_blocks, const I blocksize, 
                const T Sx[],     T Tx[])
{
    const T * block = Sx;

    //Loop over blocks
    for(I i = 0; i < n_blocks; i++)
    {
        T block_min = std::numeric_limits<T>::max();
        
        //Find smallest nonzero value in this block
        for(I j = 0; j < blocksize; j++)
        {
            const T val = block[j];
            if( val != 0.0 )
                block_min = std::min(block_min, val);
        }
        
        Tx[i] = block_min;
        
        block += blocksize;
    }    
}


/*
 * Create strength-of-connection matrix based on constrained min problem of 
 *    min( z - B*x ), such that
 *       (B*x)|_i = z|_i, i.e. they are equal at point i
 *        z = (I - (t/k) Dinv A)^k delta_i
 *   
 * Strength is defined as the relative point-wise approx. error between
 * B*x and z.  B is the near-nullspace candidates.  The constrained min problem
 * is also restricted to consider B*x and z only at the nonozeros of column i of A
 *    
 * Can use either the D_A inner product, or l2 inner-prod in the minimization 
 * problem. Using D_A gives scale invariance.  This choice is reflected in 
 * whether the parameter DB = B or diag(A)*B  
 *
 * This is a quadratic minimization problem with a linear constraint, so
 * we can build a linear system and solve it to find the critical point, i.e. minimum.
 *
 * Parameters
 * ----------
 * Sp : {int array}
 *      Row pointer array for CSR matrix S
 * Sj : {int array}
 *      Col index array for CSR matrix S
 * Sx : {float|complex array}
 *      Value array for CSR matrix S.
 *      Upon entry to the routine, S = (I - (t/k) Dinv A)^k 
 * nrows : {int}
 *      Dimension of S
 * B : {float|complex array}
 *      nrows x NullDim array of near nullspace vectors in col major form,
 *      if calling from within Python, take a transpose.
 * DB : {float|complex array}
 *      nrows x NullDim array of possibly scaled near nullspace 
 *      vectors in col major form.  If calling from within Python, take a
 *      transpose.  For a scale invariant measure, 
 *      DB = diag(A)*conjugate(B)), corresponding to the D_A inner-product
 *      Otherwise, DB = conjugate(B), corresponding to the l2-inner-product
 * b : {float|complex array}
 *      nrows x BDBCols array in row-major form.
 *      This  array is B-squared, i.e. it is each column of B
 *      multiplied against each other column of B.  For a Nx3 B,
 *      b[:,0] = conjugate(B[:,0])*B[:,0]
 *      b[:,1] = conjugate(B[:,0])*B[:,1]
 *      b[:,2] = conjugate(B[:,0])*B[:,2]
 *      b[:,3] = conjugate(B[:,1])*B[:,1]
 *      b[:,4] = conjugate(B[:,1])*B[:,2]
 *      b[:,5] = conjugate(B[:,2])*B[:,2]
 * BDBCols : {int}
 *      sum(range(NullDim+1)), i.e. number of columns in b
 * NullDim : {int}
 *      Number of nullspace vectors
 *
 * Returns
 * -------
 *   Sx is written in place and holds strength 
 *   values reflecting the above minimization problem
 *
 * Notes
 * -----
 * Upon entry to the routine, S = (I - (t/k) Dinv A)^k.  However,
 * we only need the values of S at the sparsity pattern of A.  Hence,
 * there is no need to completely calculate all of S.
 *
 * b is used to save on the computation of each local minimization problem
 *
 * Principle calling routine is ode_strength_of_connection(...) in strength.py.  
 * In that routine, it is used to calculate strength-of-connection for the case
 * of multiple near-nullspace modes.
 *
 * Examples
 * --------
 * See odes_strength_of_connection(...) in strength.py
 */
template<class I, class T, class F>
void ode_strength_helper(      T Sx[],  const I Sp[],    const I Sj[], 
                         const I nrows, const T x[],     const T y[], 
                         const T b[],   const I BDBCols, const I NullDim)
{
    //Compute maximum row length
    I max_length = 0;
    for(I i = 0; i < nrows; i++)
        max_length = std::max(max_length, Sp[i + 1] - Sp[i]);
    
    //Declare Workspace
    const I NullDimPone = NullDim + 1;
    const I work_size   = 2*NullDimPone*NullDimPone + NullDimPone;
    T * z         = new T[max_length];
    T * zhat      = new T[max_length];
    T * DBi       = new T[max_length*NullDim];
    T * Bi        = new T[max_length*NullDim];
    T * LHS       = new T[NullDimPone*NullDimPone];
    T * RHS       = new T[NullDimPone];
    T * work      = new T[work_size];
    F * sing_vals = new F[NullDimPone];

    //Rename to something more understandable
    const T * BDB = b;
    const T * B   = x;
    const T * DB  = y;
    
    //Calculate what we consider to be a "numerically" zero approximation value in z
    const F near_zero = std::numeric_limits<F>::epsilon();
    const F sqrt_near_zero = std::sqrt(near_zero);

    //Loop over rows
    for(I i = 0; i < nrows; i++)
    {
        const I rowstart = Sp[i];
        const I rowend   = Sp[i+1];
        const I length   = rowend - rowstart;
        
        if(length <= NullDim) {   
            // If B can perfectly locally approximate this row of S, 
            // then all connections are strong
            for(I kk = rowstart; kk < rowend; kk++)
            {   Sx[kk] = 1.0; }
            continue; //skip to next row
        }


        //S[i,:] ==> z
        std::copy(Sx + rowstart, Sx + rowend, z);

        //construct Bi, where B_i is B with the rows restricted only to 
        //the nonzero column indices of row i of S 
        T z_at_i = 1.0;
        for(I jj = rowstart, Bicounter = 0; jj < rowend; jj++)
        {
            const I j = Sj[jj];
            const T v = Sx[jj];

            if(i == j)
                z_at_i = v;
            
            I Bcounter = j*NullDim;
            for(I k = 0; k < NullDim; k++)
            {
                Bi[Bicounter] = B[Bcounter];
                Bicounter++;
                Bcounter++;
            }
        }
        
        //Construct Bi^H*D_A in row major,  where DB_i is DB 
        // with the rows restricted only to the nonzero column indices of row i of S
        for(I k = 0, Bicounter = 0, Bcounter = 0; k < NullDim; k++, Bcounter += nrows)
        {
            for(I jj = rowstart; jj < rowend; jj++)
            {
                DBi[Bicounter] = DB[Bcounter + Sj[jj]];
                Bicounter++; 
            }
        }
        
        //Construct B_i^H * diag(A_i) * B_i, the 1,1 block of LHS in column major ordering
        for(I kk = 0; kk < NullDimPone*NullDimPone; kk++)
        {   LHS[kk] = 0.0; }
        
        for(I jj = rowstart; jj < rowend; jj++)
        {
            const I j = Sj[jj];
            // Do work in computing Diagonal of LHS  
            I LHScounter = 0; 
            I BDBCounter = j*BDBCols;
            for(I m = 0; m < NullDim; m++)
            {
                LHS[LHScounter] += BDB[BDBCounter];
                LHScounter += NullDimPone + 1;
                BDBCounter += (NullDim - m);
            }
            // Do work in computing offdiagonals of LHS, 
            //   noting that the (1,1) block of LHS is Hermitian
            BDBCounter = j*BDBCols;
            for(I m = 0; m < NullDim; m++)
            {
                I counter = 1;
                for(I n = m+1; n < NullDim; n++)
                {
                    T elmt_bdb = BDB[BDBCounter + counter];
                    LHS[m*NullDimPone + n] += conjugate(elmt_bdb);      // entry(n,m)
                    LHS[n*NullDimPone + m] += elmt_bdb;                 // entry(m,n)
                    counter++;
                }
                BDBCounter += (NullDim - m);
            }
        }
        
        //Write last row of LHS, e_i^T*B           
        for(I j = NullDim, Bcounter = i*NullDim; j < NullDim*NullDimPone; j+= NullDimPone, Bcounter++)
        {   LHS[j] = B[Bcounter]; }
        
        //Write last column of LHS, B^H*D_A*e_i
        for(I j = NullDim*NullDimPone, Bcounter = i; j < (NullDimPone*NullDimPone - 1); j++, Bcounter += nrows)
        {   LHS[j] = DB[Bcounter]; }

        //Write first NullDim Entries of RHS
        //  Bi^H*D_A*z ==> RHS
        gemm( DBi, NullDim, length, 'F', 
                z, length,       1, 'F', 
              RHS, NullDim,      1, 'F');
        //Double the first NullDim entries in RHS
        for(I j = 0; j < NullDim; j++)
        {   RHS[j] *= 2.0; }
        //Last entry of RHS
        RHS[NullDim] = z_at_i;

        //Solve minimization problem,  pseudo_inverse(LHS)*RHS ==> prod
        svd_solve(&(LHS[0]), NullDimPone, NullDimPone, &(RHS[0]), &(sing_vals[0]), &(work[0]), work_size);

        //Find best approximation to z in span(Bi), Bi*RHS[0:NullDim] ==> zhat
        gemm(  Bi,   length, NullDim, 'F', 
              RHS,  NullDim,       1, 'F', 
             zhat,   length,       1, 'F');
        
        for(I jj = rowstart, zcounter = 0; jj < rowend; jj++, zcounter++)
        {
            //Strongly connected to self
            if(Sj[jj] == i)
            {   Sx[jj] = 1.0; }
            else
            {
                //Approximation ratio
                const T ratio = zhat[zcounter]/z[zcounter];
                
                // if zhat is numerically zero, but z is not, then weak connection
                if( mynormsq(z[zcounter]) >= near_zero &&  mynormsq(zhat[zcounter]) < near_zero )
                {   Sx[jj] = 0.0; }
                
                // if zhat[j] and z[j] have different sign, then weak connection
                else if( (signof(real(zhat[zcounter])) != signof(real(z[zcounter]))) || 
                         (signof(imag(zhat[zcounter])) != signof(imag(z[zcounter])))    )
                {   Sx[jj] = 0.0; }
                
                //Calculate approximation error as strength value
                else 
                {
                    const F error = mynormsq(-ratio + 1.0);
                    //This comparison allows for predictable handling of the "zero" error case
                    if(error < sqrt_near_zero)
                    {    Sx[jj] = 1e-4; }
                    else
                    {    Sx[jj] = std::sqrt(error); }
                }
            }
            

        } //end for

    } //end i loop

    //Clean up heap
    delete[] LHS; 
    delete[] RHS; 
    delete[] z; 
    delete[] zhat; 
    delete[] DBi; 
    delete[] Bi; 
    delete[] work; 
    delete[] sing_vals; 
}

/* Helper function for my_inner(...) where we search for the row-th entry 
 * in the current j-th column of CSC matrix, B.
 *
 * Parameters
 * ----------
 * Bj : {int array}
 *  column indices of CSC matrix, B
 *  MUst be sorted
 * Bx : {float array}
 *  values array for CSC matrix B
 * BptrLim : {int}
 *  stop index for the current column of B
 *  equal to B.indptr[j+1]
 * Bptr : {int}
 *  current index under consideration in this column of B
 *  B.indptr[j] <= Bptr < B.indptr[j+1]
 * row : {int}
 *  the row number of the entry that we are searching for
 * Aval : {float}
 *  the current entry of the right matrix in the mat-mat
 *  this is the (i,k)-th entry where k=row
 * sum : {float}
 *  cumulative variable to store the eventual value A(i,:)*B(:,j)
 *
 * Returns
 * -------
 * sum is modified in place, such that 
 *   sum += Aval*B(row,j) 
 * 
 * Bptr is modified in place such that it 
 *   is incremented to the first entry past B(row,j)
 *
 * Notes
 * -----
 * Principle calling routine is my_inner(...) in this file
 *
 * CSC matrix B must have sorted indices
 */
template<class I, class T>
inline void find_matval( const I Bj[],  const T Bx[],  const I BptrLim,
                         const I row,         I &Bptr, const T Aval,
                               T &sum )
{
    // loop over this column of B until we either find a matching entry in B, 
    // or we reach an entry in B that has a row number larger than the current column number in A
    while(Bptr < BptrLim)
    {
        if(Bj[Bptr] == row)
        {   
            sum += Aval*Bx[Bptr];
            Bptr++;
            return;
        }
        else if(Bj[Bptr] > row)
        {   
            //entry not found, do nothing
            return; 
        }
        Bptr++;
    }

    // entry not found, do nothing
}


/* For use in incomplete_matmat(...)
 * Calcuate <A_{row,:}, B_{:, col}>
 *
 * Parameters
 * ----------
 * Ap : {int array}
 *  row ptr array for CSR matrix A
 * Aj : {int array}
 *  col index array for CSR matrix A
 *  MUst be sorted
 * Ax : {float array}
 *  value array for CSR matrix A
 * Bp : {int array}
 *  col ptr array for CSC matrix A
 * Bj : {int array}
 *  row index array for CSC matrix A
 *  MUst be sorted
 * Bx : {float array}
 *  value array for CSC matrix A
 * row, col : {int}
 *  indicate which row of A and column of B to take
 *  the inner product of
 *
 * Returns
 * -------
 * returns <A_{row,:}, B_{:, col}>
 *
 * Notes
 * -----
 * Principle calling routine is incomplete_matmat in this file 
 *
 * A and B are assumed to have sorted indices
 *  
 */
template<class I, class T>
inline T my_inner( const I Ap[],  const I Aj[],    const T Ax[], 
                   const I Bp[],  const I Bj[],    const T Bx[], 
                   const I row,   const I col )
{
    // sum will be incremented by Ax[.]*Bx[.] each time an entry in 
    // this row of A matches up with an entry in this column of B
    T sum = 0.0;
    
    I Bptr = Bp[col];
    I BptrLim = Bp[col+1];
    I rowstart = Ap[row];
    I rowend = Ap[row+1];

    // Loop over row=row of A, looking for entries in column=col 
    // of B that line up for the innerproduct
    for(I colptr = rowstart; colptr < rowend; colptr++)
    {
        // Return if there are no more entries in this column of B
        if(Bptr == BptrLim)
        {   return sum;}

        //Indices are assumed to be sorted
        I Acol = Aj[colptr];
        if(Bj[Bptr] <= Acol)
        {
            //increment sum by Ax[colptr]*B(Acol,col) = A(row,Acol)*B(Acol,col)
            find_matval(Bj, Bx, BptrLim, Acol, Bptr, Ax[colptr], sum);
        }
    }
    return sum;
}


/* Calculate A*B = S, but only at the pre-exitsting sparsity
 * pattern of S, i.e. do an exact, but incomplete mat-mat mult.
 *
 * A must be in CSR, B must be in CSC and S must be in CSR
 * Indices for A, B and S must be sorted
 * A, B, and S must be square
 *
 * Parameters
 * ----------
 * Ap : {int array}
 *      Row pointer array for CSR matrix A
 * Aj : {int array}
 *      Col index array for CSR matrix A
 * Ax : {float|complex array}
 *      Value array for CSR matrix A
 * Bp : {int array}
 *      Row pointer array for CSC matrix B
 * Bj : {int array}
 *      Col index array for CSC matrix B
 * Bx : {float|complex array}
 *      Value array for CSC matrix B
 * Sp : {int array}
 *      Row pointer array for CSR matrix S
 * Sj : {int array}
 *      Col index array for CSR matrix S
 * Sx : {float|complex array}
 *      Value array for CSR matrix S
 * dimen: {int} 
 *      dimensionality of A,B and S
 *
 * Returns
 * -------
 * Sx is modified inplace to reflect S(i,j) = <A_{i,:}, B_{:,j}>
 *
 * Notes
 * -----
 * A must be in CSR, B must be in CSC and S must be in CSR.
 * Indices for A, B and S must all be sorted.
 * A, B and S must be square.
 *
 * Algorithm is naive, S(i,j) = <A_{i,:}, B_{:,j}>
 * But, the routine is written for the case when S's 
 * sparsity pattern is a subset of A*B, so this algorithm 
 * should work well.
 *
 * Principle calling routine is ode_strength_of_connection in 
 * strength.py.  Here it is used to calculate S*S only at the 
 * sparsity pattern of the original operator.  This allows for
 * BIG cost savings.
 *
 * Examples
 * --------
 * >>> from pyamg.multigridtools import incomplete_matmat
 * >>> from scipy import arange, eye, ones
 * >>> from scipy.sparse import csr_matrix, csc_matrix
 * >>>
 * >>> A = csr_matrix(arange(1,10,dtype=float).reshape(3,3))
 * >>> B = csc_matrix(ones((3,3),dtype=float))
 * >>> AB = csr_matrix(eye(3,3,dtype=float))
 * >>> A.sort_indices()
 * >>> B.sort_indices()
 * >>> AB.sort_indices()
 * >>> incomplete_matmat(A.indptr, A.indices, A.data, B.indptr, B.indices,
 *                       B.data, AB.indptr, AB.indices, AB.data, 3)
 * >>> print "Incomplete Matrix-Matrix Multiplication\n" + str(AB.todense())
 * >>> print "Complete Matrix-Matrix Multiplication\n" + str((A*B).todense())
 */
template<class I, class T, class F>
void incomplete_matmat(  const I Ap[],  const I Aj[],    const T Ax[], 
                         const I Bp[],  const I Bj[],    const T Bx[], 
                         const I Sp[],  const I Sj[],          T Sx[], const I dimen)
{
    for(I row = 0; row < dimen; row++)
    {
        I rowstart = Sp[row];
        I rowend = Sp[row+1];

        for(I colptr = rowstart; colptr < rowend; colptr++)
        {
            //calculate S(row, Sj[colptr]) = <A_{row,:}, B_{:,Sj[colptr]}>
            Sx[colptr] = my_inner(Ap, Aj, Ax, Bp, Bj, Bx, row, Sj[colptr]);
        }
    }
}



#endif

