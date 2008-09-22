#ifndef SMOOTHED_AGGREGATION_H
#define SMOOTHED_AGGREGATION_H

#include <iostream>
#include <vector>
#include <iterator>
#include <algorithm>

#include <assert.h>
#include <cmath>

#include "linalg.h"
 
template<class I, class T>
void symmetric_strength_of_connection(const I n_row, 
                                      const T theta,
                                      const I Ap[], const I Aj[], const T Ax[],
                                            I Sp[],       I Sj[],       T Sx[])
{
    //Sp,Sj form a CSR representation where the i-th row contains
    //the indices of all the strong connections from node i
    std::vector<T> diags(n_row);

    //compute diagonal values
    for(I i = 0; i < n_row; i++){
        T diag = 0;
        for(I jj = Ap[i]; jj < Ap[i+1]; jj++){
            if(Aj[jj] == i){
                diag += Ax[jj]; //gracefully handle duplicates
            }
        }    
        diags[i] = std::abs(diag);
    }

    I nnz = 0;
    Sp[0] = 0;

    for(I i = 0; i < n_row; i++){

        T eps_Aii = theta*theta*diags[i];

        for(I jj = Ap[i]; jj < Ap[i+1]; jj++){
            const I   j = Aj[jj];
            const T Aij = Ax[jj];

            if(i == j){continue;}  //skip diagonal

            //  |A(i,j)| >= theta * sqrt(|A(i,i)|*|A(j,j)|) 
            if(Aij*Aij >= eps_Aii * diags[j]){    
                Sj[nnz] =   j;
                Sx[nnz] = Aij;
                nnz++;
            }
        }
        Sp[i+1] = nnz;
    }
}


/*
 * Compute aggregates for a matrix A stored in CSR format
 *
 * Parameters:
 *   n_row         - # of rows in A
 *   Ap[n_row + 1] - CSR row pointer
 *   Aj[nnz]       - CSR column indices
 *    x[n_row]     - aggregate numbers for each node
 *
 * Returns:
 *  The number of aggregates (== max(x[:]) + 1 )
 *
 * Notes:
 *    It is assumed that A is symmetric.
 *    A may contain diagonal entries (self loops)
 *    Unaggregated nodes are marked with a -1
 *    
 */
    
template <class I>
I standard_aggregation(const I n_row,
                       const I Ap[], 
                       const I Aj[],
                             I  x[])
{
    // Bj[n] == -1 means i-th node has not been aggregated
    std::fill(x, x + n_row, 0);

    I next_aggregate = 1; // number of aggregates + 1

    //Pass #1
    for(I i = 0; i < n_row; i++){
        if(x[i]){ continue; } //already marked

        const I row_start = Ap[i];
        const I row_end   = Ap[i+1];

        //Determine whether all neighbors of this node are free (not already aggregates)
        bool has_aggregated_neighbors = false;
        bool has_neighbors            = false;
        for(I jj = row_start; jj < row_end; jj++){
            const I j = Aj[jj];
            if( i != j ){
                has_neighbors = true;
                if( x[j] ){
                    has_aggregated_neighbors = true;
                    break;
                }
            }
        }    

        if(!has_neighbors){
            //isolated node, do not aggregate
            x[i] = -n_row;
        }
        else if (!has_aggregated_neighbors){
            //Make an aggregate out of this node and its neighbors
            x[i] = next_aggregate;
            for(I jj = row_start; jj < row_end; jj++){
                x[Aj[jj]] = next_aggregate;
            }
            next_aggregate++;
        }
    }


    //Pass #2
    // Add unaggregated nodes to any neighboring aggregate
    for(I i = 0; i < n_row; i++){
        if(x[i]){ continue; } //already marked

        for(I jj = Ap[i]; jj < Ap[i+1]; jj++){
            const I j = Aj[jj];
        
            const I xj = x[j];
            if(xj > 0){
                x[i] = -xj;
                break;
            }
        }   
    }
   
    next_aggregate--; 

    //Pass #3
    for(I i = 0; i < n_row; i++){
        const I xi = x[i];

        if(xi != 0){ 
            // node i has been aggregated
            if(xi > 0)
                x[i] = xi - 1;
            else if(xi == -n_row)
                x[i] = -1;
            else
                x[i] = -xi - 1;
            continue;
        }

        // node i has not been aggregated
        const I row_start = Ap[i];
        const I row_end   = Ap[i+1];

        x[i] = next_aggregate;

        for(I jj = row_start; jj < row_end; jj++){
            const I j = Aj[jj];

            if(x[j] == 0){ //unmarked neighbors
                x[j] = next_aggregate;
            }
        }  
        next_aggregate++;
    }

    return next_aggregate; //number of aggregates
}



template <class I, class T>
void fit_candidates(const I n_row,
                    const I n_col,
                    const I   K1,
                    const I   K2,
                    const I Ap[], 
                    const I Ai[],
                          T Ax[],
                    const T  B[],
                          T  R[],
                    const T  tol)
{
    std::fill(R, R + (n_col*K2*K2), 0);


    const I BS = K1*K2; //blocksize

    //Copy blocks into Ax
    for(I j = 0; j < n_col; j++){
        T * Ax_start = Ax + BS * Ap[j];

        for(I ii = Ap[j]; ii < Ap[j+1]; ii++){
            const T * B_start = B + BS*Ai[ii];
            const T * B_end   = B_start + BS;
            std::copy(B_start, B_end, Ax_start);
            Ax_start += BS;
        }
    }

    
    //orthonormalize columns
    for(I j = 0; j < n_col; j++){
        const I col_start  = Ap[j];
        const I col_end    = Ap[j+1];

        T * Ax_start = Ax + BS * col_start;
        T * Ax_end   = Ax + BS * col_end;
        T * R_start  = R  + j * K2 * K2;
        
        for(I bj = 0; bj < K2; bj++){
            //compute norm of block column
            T norm = 0;

            {
                T * Ax_col = Ax_start + bj;
                while(Ax_col < Ax_end){
                    norm   += (*Ax_col) * (*Ax_col);
                    Ax_col += K2;
                }
                norm = std::sqrt(norm);
            }
            
            const T threshold = tol * norm;
    
            //orthogonalize bj against previous columns
            for(I bi = 0; bi < bj; bi++){

                //compute dot product with column bi
                T dot_prod = 0;

                {
                    T * Ax_bi = Ax_start + bi;
                    T * Ax_bj = Ax_start + bj;
                    while(Ax_bi < Ax_end){
                        dot_prod += (*Ax_bi) * (*Ax_bj);
                        Ax_bi    += K2;
                        Ax_bj    += K2;
                    }
                }

                // orthogonalize against column bi
                {
                    T * Ax_bi = Ax_start + bi;
                    T * Ax_bj = Ax_start + bj;
                    while(Ax_bi < Ax_end){
                        *Ax_bj -= dot_prod * (*Ax_bi);
                        Ax_bi  += K2;
                        Ax_bj  += K2;
                    }
                }
                
                R_start[K2 * bi + bj] = dot_prod;
            } // end orthogonalize bj against previous columns


            //compute norm of column bj
            norm = 0;
            {
                T * Ax_bj = Ax_start + bj;
                while(Ax_bj < Ax_end){
                    norm  += (*Ax_bj) * (*Ax_bj);
                    Ax_bj += K2;
                }
                norm = std::sqrt(norm);
            }
           

            //normalize column bj if, after orthogonalization, its
            //euclidean norm exceeds the threshold. otherwise set 
            //column bj to 0.
            T scale;
            if(norm > threshold){
                scale = 1.0/norm;
                R_start[K2 * bj + bj] = norm;
            } else {
                scale = 0;
                R_start[K2 * bj + bj] = 0;
            }
            {
                T * Ax_bj = Ax_start + bj;
                while(Ax_bj < Ax_end){
                    *Ax_bj *= scale;
                    Ax_bj  += K2;
                }
            }

        } // end orthogonalizing block column j
    }
}


/*
 * Helper routine for satisfy_constraints routine called 
 *     by energy_prolongation_smoother(...) in smooth.py
 * This implements the python code:
 *
 *   # U is a BSR matrix, B is num_block_rows x ColsPerBlock x ColsPerBlock
 *   # UB is num_block_rows x RowsPerBlock x ColsPerBlock,  BtBinv is num_block_rows x ColsPerBlock x ColsPerBlock
 *
 *   rows = csr_matrix((U.indices,U.indices,U.indptr), shape=(U.shape[0]/RowsPerBlock,U.shape[1]/ColsPerBlock)).tocoo(copy=False).row
 *   for n,j in enumerate(U.indices):
 *      i = rows[n]
 *      Bi  = B[j]
 *      UBi = UB[i]
 *      U.data[n] -= dot(UBi,dot(BtBinv[i],Bi.T))
 *
 * Parameters:
 *  RowsPerBlock     rows per block in the BSR matrix, S
 *  ColsPerBlock     cols per block in the BSR matrix, S
 *  num_blocks       number of stored blocks in Sx
 *  num_block_rows   S.shape[0]/RowsPerBlock
 *  x                Near-nullspace vectors, B
 *  y                S*B
 *  z                BtBinv
 *  Sp,Sj,Sx         BSR matrix, S, that is the update to the prolongator
 *  
 * Returns:
 *  Sx is modified such that S*B = 0
 *
 * Notes:
 *  
 */          

template<class I, class T>
void satisfy_constraints_helper(const I RowsPerBlock,   const I ColsPerBlock, const I num_blocks,
                                const I num_block_rows, const T x[], const T y[], const T z[], const I Sp[], const I Sj[], T Sx[])
{
    //Rename to something more familiar
    const T * B = x;
    const T * UB = y;
    const T * BtBinv = z;
    
    //Declare
    I block_size = RowsPerBlock*ColsPerBlock;
    I ColsPerBlockSq = ColsPerBlock*ColsPerBlock;

    //C will store an intermediate mat-mat product
    std::vector<T> Update(block_size,0);
    std::vector<T> C(ColsPerBlockSq,0);
    for(I i = 0; i < ColsPerBlockSq; i++)
    {   C[i] = 0.0; }

    //Begin Main Loop
    for(I i = 0; i < num_block_rows; i++)
    {
        I rowstart = Sp[i]; 
        I rowend = Sp[i+1];

        for(I j = rowstart; j < rowend; j++)
        {
            //Calculate C = BtBinv[i*blocksize => (i+1)*blocksize]  *  B[ Sj[j]*blocksize => (Sj[j]+1)*blocksize ]^T
            gemm(&(BtBinv[i*ColsPerBlockSq]), ColsPerBlock, ColsPerBlock, 'F', &(B[Sj[j]*ColsPerBlockSq]), ColsPerBlock, ColsPerBlock, 'F', &(C[0]), ColsPerBlock, ColsPerBlock, 'T');
            
            //Calculate Sx[ j*block_size => (j+1)*blocksize ] =  UB[ i*block_size => (i+1)*blocksize ] * C
            gemm(&(UB[i*block_size]), RowsPerBlock, ColsPerBlock, 'F', &(C[0]), ColsPerBlock, ColsPerBlock, 'F', &(Update[0]), RowsPerBlock, ColsPerBlock, 'F');
            
            //Update Sx
            for(I k = 0; k < block_size; k++)
            {   Sx[j*block_size + k] -= Update[k]; }
        }
    }
}


/*
 * Helper routine for energy_prolongation_smoother
 * Calculates the following python code:
 *
 *  Bblk = asarray(B).reshape(-1,NullDim,NullDim)
 *  colindices = array_split(Sparsity_Pattern.indices,Sparsity_Pattern.indptr[1:-1])
 *  for i,cols in enumerate(colindices):
 *      if len(cols) > 0:
 *      Bi = Bblk[cols].reshape(-1,NullDim)
 *      BtBinv[i] = pinv2(dot(Bi.T,Bi))
 *
 * Parameters:
 *   NullDim      Number of near nullspace vectors
 *   Nnodes       Number of nodes, i.e. block rows in BSR matrix, S
 *   ColsPerBlock Columns per block in S
 *   b            In row-major form, this is B-squared, i.e. it 
 *                is each column of B multiplied against each 
 *                other column of B.  For a Nx3 B,
 *                b[:,0] = B[:,0]*B[:,0]
 *                b[:,1] = B[:,0]*B[:,1]
 *                b[:,2] = B[:,0]*B[:,2]
 *                b[:,3] = B[:,1]*B[:,1]
 *                b[:,4] = B[:,1]*B[:,2]
 *                b[:,5] = B[:,2]*B[:,2]
 *   BsqCols      sum(range(NullDim+1)), i.e. number of columns in b
 *   x            BtBinv (output).  Should be zeros upon entry
 *   Sp,Sj        BSR indptr and indices members for matrix, S
 *
 * Returns:
 *  BtBinv      BtBinv[i] = pseudo_invers(B_i^T*B_i), where
 *              B_i is B[colindices,:], colindices = all the nonzero
 *              column indices for block row i in S
 */          

template<class I, class T>
void invert_BtB(const I NullDim, const I Nnodes,  const I ColsPerBlock, 
                const T b[],     const I BsqCols, T x[], 
                const I Sp[],    const I Sj[])
{
    //Rename to something more familiar
    const T * Bsq = b;
    T * BtBinv = x;
    
    //Declare workspace
    const I NullDimLoc = NullDim;
    const I NullDimSq  = NullDim*NullDim;
    const I work_size  = 5*NullDim + 10;

    T * BtB       = new T[NullDimSq];
    T * work      = new T[work_size];
    T * sing_vals = new T[NullDim];
    T * identity  = new T[NullDimSq];
    
    //Build an identity matrix in col major format for the Fortran routine called in svd_solve
    for(I i = 0; i < NullDimSq; i++)
    {   identity[i] = 0.0;}
    for(I i = 0; i < NullDimSq; i+= NullDim + 1)
    {   identity[i] = 1.0;}


    //Loop over each row
    for(I i = 0; i < Nnodes; i++)
    {
        const I rowstart = Sp[i];
        const I rowend   = Sp[i+1];
        for(I k = 0; k < NullDimSq; k++)
        {   BtB[k] = 0.0; }
        
        //Loop over row i in order to calculate B_i^T*B_i, where B_i is B 
        // with the rows restricted only to the nonzero column indices of row i of S
        for(I j = rowstart; j < rowend; j++)
        {
            // Calculate absolute column index start and stop 
            //  for block column j of BSR matrix, S
            const I colstart = Sj[j]*ColsPerBlock;
            const I colend   = colstart + ColsPerBlock;

            //Loop over each absolute column index, k, of block column, j
            for(I k = colstart; k < colend; k++)
            {          
                // Do work in computing Diagonal of  BtB  
                I BtBcounter = 0; 
                I BsqCounter = k*BsqCols;
                for(I m = 0; m < NullDim; m++)
                {
                    BtB[BtBcounter] += Bsq[BsqCounter];
                    BtBcounter += NullDim + 1;
                    BsqCounter += (NullDim - m);
                }
                // Do work in computing offdiagonals of BtB, noting that BtB is symmetric
                BsqCounter = k*BsqCols;
                for(I m = 0; m < NullDim; m++)
                {
                    I counter = 1;
                    for(I n = m+1; n < NullDim; n++)
                    {
                        T elmt_bsq = Bsq[BsqCounter + counter];
                        BtB[m*NullDim + n] += elmt_bsq;
                        BtB[n*NullDim + m] += elmt_bsq;
                        counter ++;
                    }
                    BsqCounter += (NullDim - m);
                }
            } // end k loop
        } // end j loop

        // pseudo_inverse(BtB) ==> blockinverse
        // since BtB is symmetric theres no need to convert to row major
        T * blockinverse = BtBinv + i*NullDimSq; //pseudoinverse output
        for(I k = 0; k < NullDimSq; k++)
        {   blockinverse[k] = identity[k]; }
        svd_solve(BtB, (int) NullDimLoc, (int) NullDimLoc, blockinverse, (int) NullDimLoc, sing_vals, work, (int) work_size);
          

    } // end i loop

    delete[] BtB; 
    delete[] work;
    delete[] sing_vals; 
    delete[] identity;
}

#endif
