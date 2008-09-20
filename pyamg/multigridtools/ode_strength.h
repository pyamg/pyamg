#ifndef ODE_STRENGTH_H
#define ODE_STRENGTH_H

#include <iterator>
#include <algorithm>
#include <cmath>
#include "smoothed_aggregation.h"

/*
 *
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
 * Parameters:
 *   n_row      Dimension of matrix, S
 *   epsilon    Drop tolerance
 *   Sp, Sj, Sx Define CSR matrix, S
 *
 * Returns:
 *   Sx         Such that the above dropping strategy has been applied
 *
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
 *
 *  Given a BSR matrice's data structure, return a linear array of length 
 *  num_blocks, which holds each block's smallest, nonzero, entry
 *  
 * Parameters:
 *   n_blocks       Number of blocks in matrix, S
 *   blocksize      Size of each block
 *   Sx             Block data structure of BSR matrix, S
 *
 * Returns:
 *   Tx             Tx[i] holds the minimum nonzero of block i of S
 *
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
 *
 * Given Strength of connection matrix, Atilde, calculate strength based on 
 * constrained min problem of 
 *    min( z - B*x ), such that
 *       (B*x)|_i = z|_i, i.e. they are equal at point i
 *        z = (I - (t/k) Dinv A)^k delta_i
 *   
 * Strength is defined as the relative point-wise approx. error between
 * B*x and z.  We don't use the full z in this problem, only that part of
 * z that is in the sparsity pattern of A.
 *    
 * Can use either the D-norm, and inner product, or l2-norm and inner-prod
 * to solve the constrained min problem.  Using D gives scale invariance.
 * This choice is reflected in whether the parameter DB = B or diag(A)*B  
 *
 * This is a quadratic minimization problem with a linear constraint, so
 * we can build a linear system and solve it to find the critical point,
 * i.e. minimum.
 *
 * Parameters:
 *   Sx, Sp, Sj     Define CSR raw strength of connection matrix
 *   nrows          Dimension of S
 *   B              Transpose of near nullspace vectors (nrows x NullDim)
 *   DB (transpose) Depending on calling routine, either (diag(A)*B)^T or B^T
 *                    (nrows x NullDim)
 *   b              In row-major form, this is B-squared, i.e. it 
 *                  is each column of B multiplied against each 
 *                  other column of B.  For a Nx3 B,
 *                  b[:,0] = B[:,0]*B[:,0]
 *                  b[:,1] = B[:,0]*B[:,1]
 *                  b[:,2] = B[:,0]*B[:,2]
 *                  b[:,3] = B[:,1]*B[:,1]
 *                  b[:,4] = B[:,1]*B[:,2]
 *                  b[:,5] = B[:,2]*B[:,2]
 *   BDBCols        sum(range(NullDim+1)), i.e. number of columns in b
 *   NullDim        Number of nullspace vectors
 *
 * Returns:
 *   Sx             Holds new strength values reflecting 
 *                    the above minimization problem
 */
template<class I, class T>
void ode_strength_helper(      T Sx[],  const I Sp[],    const I Sj[], 
                         const I nrows, const T x[],     const T y[], 
                         const T b[],     const I BDBCols, const I NullDim)
{
    //Compute maximum row length
    I max_length = 0;
    for(I i = 0; i < nrows; i++)
        max_length = std::max(max_length, Sp[i + 1] - Sp[i]);
    
    //Declare Workspace
    const I NullDimPone = NullDim + 1;
    const I work_size   = 5*NullDimPone + 10;
    T * z         = new T[max_length];
    T * zhat      = new T[max_length];
    T * DBi       = new T[max_length*NullDim];
    T * Bi        = new T[max_length*NullDim];
    T * LHS       = new T[NullDimPone*NullDimPone];
    T * RHS       = new T[NullDimPone];
    T * work      = new T[work_size];
    T * sing_vals = new T[NullDimPone];

    //Rename to something more understandable
    const T * BDB = b;
    const T * B   = x;
    const T * DB  = y;
    
    //Calculate what we consider to be a "numerically" zero approximation value in z
    const T near_zero = std::sqrt( std::numeric_limits<T>::epsilon() );

    //Loop over rows
    for(I i = 0; i < nrows; i++)
    {
        const I rowstart = Sp[i];
        const I rowend   = Sp[i+1];
        const I length   = rowend - rowstart;
        
        if(length <= NullDim) {   
            // If B can perfectly locally approximate this row of S, 
            // then all connections are strong
            std::fill(Sx + rowstart, Sx + rowend, static_cast<T>(1.0));
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
        
        //Construct DBi^T in row major,  where DB_i is DB 
        // with the rows restricted only to the nonzero column indices of row i of S
        for(I k = 0, Bicounter = 0, Bcounter = 0; k < NullDim; k++, Bcounter += nrows)
        {
            for(I jj = rowstart; jj < rowend; jj++)
            {
                DBi[Bicounter] = DB[Bcounter + Sj[jj]];
                Bicounter++; 
            }
        }
        
        //Construct B_i^T * diag(A_i) * B_i, the 1,1 block of LHS
        std::fill(LHS, LHS + NullDimPone * NullDimPone, static_cast<T>(0.0)); // clear LHS
        
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
            //   noting that the (1,1) block of LHS is symmetric
            BDBCounter = j*BDBCols;
            for(I m = 0; m < NullDim; m++)
            {
                I counter = 1;
                for(I n = m+1; n < NullDim; n++)
                {
                    T elmt_bdb = BDB[BDBCounter + counter];
                    LHS[m*NullDimPone + n] += elmt_bdb;
                    LHS[n*NullDimPone + m] += elmt_bdb;
                    counter++;
                }
                BDBCounter += (NullDim - m);
            }
        }
        
        //Write last row of LHS           
        for(I j = NullDim, Bcounter = i*NullDim; j < NullDim*NullDimPone; j+= NullDimPone, Bcounter++)
        {   LHS[j] = B[Bcounter]; }
        
        //Write last column of LHS
        for(I j = NullDim*NullDimPone, Bcounter = i; j < (NullDimPone*NullDimPone - 1); j++, Bcounter += nrows)
        {   LHS[j] = DB[Bcounter]; }

        //Write first NullDim Entries of RHS
        //  DBi^T*z ==> RHS
        gemm( DBi, NullDim, length, 'F', 
                z, length,       1, 'F', 
              RHS, NullDim,      1, 'F');
        //Double the first NullDim entries in RHS
        for(I j = 0; j < NullDim; j++)
        {   RHS[j] *= 2.0; }
        //Last entry of RHS
        RHS[NullDim] = z_at_i;

        //Solve minimization problem,  pseudo_inverse(LHS)*RHS ==> prod
        svd_solve(&(LHS[0]), NullDimPone, NullDimPone, &(RHS[0]), 1, &(sing_vals[0]), &(work[0]), work_size);

        //Find best approximation to z in span(Bi), Bi*RHS[0:NullDim] ==> zhat
        gemm(  Bi,   length, NullDim, 'F', 
              RHS,  NullDim,       1, 'F', 
             zhat,   length,       1, 'F');
        
        for(I jj = rowstart, zcounter = 0; jj < rowend; jj++, zcounter++)
        {
            //Perfectly connected to self
            if(Sj[jj] == i)
            {   Sx[jj] = 1.0; }
            else
            {
                //Approximation ratio
                const T ratio = zhat[zcounter]/z[zcounter];
                
                // if zhat is numerically zero, but z is not, then weak connection
                if( std::abs(z[zcounter]) >= near_zero &&  std::abs(zhat[zcounter]) < near_zero )
                {   Sx[jj] = 0.0; }
                
                // if zhat[j] and z[j] have different sign, then weak connection
                else if(ratio < 0.0)
                {   Sx[jj] = 0.0; }
                
                //Calculate approximation error as strength value
                else 
                {
                    const T error = std::abs(1.0 - ratio);
                    //This comparison allows for predictable handling of the "zero" error case
                    if(error < near_zero)
                    {    Sx[jj] = near_zero; }
                    else
                    {    Sx[jj] = error; }
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

#endif

