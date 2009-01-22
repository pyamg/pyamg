#ifndef SMOOTHED_AGGREGATION_H
#define SMOOTHED_AGGREGATION_H

#include <iostream>
#include <vector>
#include <iterator>
#include <algorithm>

#include <assert.h>
#include <cmath>

#include "linalg.h"
 
template<class I, class T, class F>
void symmetric_strength_of_connection(const I n_row, 
                                      const F theta,
                                      const I Ap[], const I Aj[], const T Ax[],
                                            I Sp[],       I Sj[],       T Sx[])
{
    //Sp,Sj form a CSR representation where the i-th row contains
    //the indices of all the strong connections from node i
    std::vector<F> diags(n_row);

    //compute norm of diagonal values
    for(I i = 0; i < n_row; i++){
        T diag = 0.0;
        for(I jj = Ap[i]; jj < Ap[i+1]; jj++){
            if(Aj[jj] == i){
                diag += Ax[jj]; //gracefully handle duplicates
            }
        }    
        diags[i] = mynorm(diag);
    }

    I nnz = 0;
    Sp[0] = 0;

    for(I i = 0; i < n_row; i++){

        F eps_Aii = theta*theta*diags[i];

        for(I jj = Ap[i]; jj < Ap[i+1]; jj++){
            const I   j = Aj[jj];
            const T Aij = Ax[jj];

            if(i == j){continue;}  //skip diagonal

            //  |A(i,j)| >= theta * sqrt(|A(i,i)|*|A(j,j)|) 
            if(mynormsq(Aij) >= eps_Aii * diags[j]){    
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



template <class I, class S, class T, class DOT, class NORM>
void fit_candidates_common(const I n_row,
                           const I n_col,
                           const I   K1,
                           const I   K2,
                           const I Ap[], 
                           const I Ai[],
                                 T Ax[],
                           const T  B[],
                                 T  R[],
                           const S  tol,
                           const DOT& dot,
                           const NORM& norm)
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
            S norm_j = 0;

            {
                T * Ax_col = Ax_start + bj;
                while(Ax_col < Ax_end){
                    norm_j += norm(*Ax_col);
                    Ax_col += K2;
                }
                norm_j = std::sqrt(norm_j);
            }
            
            const S threshold_j = tol * norm_j;
    
            //orthogonalize bj against previous columns
            for(I bi = 0; bi < bj; bi++){

                //compute dot product with column bi
                T dot_prod = 0;

                {
                    T * Ax_bi = Ax_start + bi;
                    T * Ax_bj = Ax_start + bj;
                    while(Ax_bi < Ax_end){
                        dot_prod += dot(*Ax_bj,*Ax_bi);
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
            norm_j = 0;
            {
                T * Ax_bj = Ax_start + bj;
                while(Ax_bj < Ax_end){
                    norm_j += norm(*Ax_bj);
                    Ax_bj  += K2;
                }
                norm_j = std::sqrt(norm_j);
            }
           

            //normalize column bj if, after orthogonalization, its
            //euclidean norm exceeds the threshold. otherwise set 
            //column bj to 0.
            T scale;
            if(norm_j > threshold_j){
                scale = 1.0/norm_j;
                R_start[K2 * bj + bj] = norm_j;
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

template<class T>
struct real_norm
{
    T operator()(const T& a) const { return a*a; }
};

template<class T>
struct real_dot
{
    T operator()(const T& a, const T& b) const { return b*a; }
};

template<class T>
struct complex_dot
{
    T operator()(const T& a, const T& b) const { return T(b.real,-b.imag) * a; }
};

template<class S, class T>
struct complex_norm
{
    S operator()(const T& a) const { return a.real * a.real + a.imag * a.imag; }
};

template <class I, class T>
void fit_candidates_real(const I n_row,
                         const I n_col,
                         const I   K1,
                         const I   K2,
                         const I Ap[], 
                         const I Ai[],
                               T Ax[],
                         const T  B[],
                               T  R[],
                         const T  tol)
{ fit_candidates_common(n_row, n_col, K1, K2, Ap, Ai, Ax, B, R, tol, real_dot<T>(), real_norm<T>()); }

template <class I, class S, class T>
void fit_candidates_complex(const I n_row,
                            const I n_col,
                            const I   K1,
                            const I   K2,
                            const I Ap[], 
                            const I Ai[],
                                  T Ax[],
                            const T  B[],
                                  T  R[],
                            const S  tol)
{ fit_candidates_common(n_row, n_col, K1, K2, Ap, Ai, Ax, B, R, tol, complex_dot<T>(), complex_norm<S,T>()); }


/*
 * Helper routine for satisfy_constraints routine called 
 *     by energy_prolongation_smoother(...) in smooth.py
 * This implements the python code:
 *
 *   # U is a BSR matrix, B is num_block_rows x ColsPerBlock x ColsPerBlock
 *   # UB is num_block_rows x RowsPerBlock x ColsPerBlock,  BtBinv is num_block_rows x ColsPerBlock x ColsPerBlock
 *   B  = asarray(B).reshape(-1,ColsPerBlock,B.shape[1])
 *   UB = asarray(UB).reshape(-1,RowsPerBlock,UB.shape[1])
 *
 *   rows = csr_matrix((U.indices,U.indices,U.indptr), shape=(U.shape[0]/RowsPerBlock,U.shape[1]/ColsPerBlock)).tocoo(copy=False).row
 *   for n,j in enumerate(U.indices):
 *      i = rows[n]
 *      Bi  = mat(B[j])
 *      UBi = UB[i]
 *      U.data[n] -= dot(UBi,dot(BtBinv[i],Bi.H))
 *
 * Parameters:
 *  RowsPerBlock     rows per block in the BSR matrix, S
 *  ColsPerBlock     cols per block in the BSR matrix, S
 *  num_blocks       number of stored blocks in Sx
 *  num_block_rows   S.shape[0]/RowsPerBlock
 *  x                Conjugate of near-nullspace vectors, B, in row major
 *  y                S*B, in row major
 *  z                BtBinv, in row major
 *  Sp,Sj,Sx         BSR matrix, S, that is the update to the prolongator
 *                   Note that data array Sx is in row major
 *  
 * Returns:
 *  Sx is modified such that S*B = 0
 *
 * Notes:
 *  
 */          

template<class I, class T, class F>
void satisfy_constraints_helper(const I RowsPerBlock,   const I ColsPerBlock, const I num_blocks,
                                const I num_block_rows, const T x[], const T y[], const T z[], const I Sp[], const I Sj[], T Sx[])
{
    //Rename to something more familiar
    const T * Bt = x;
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
            // Calculate C = BtBinv[i*blocksize => (i+1)*blocksize]  *  B[ Sj[j]*blocksize => (Sj[j]+1)*blocksize ]^H
            // Implicit transpose of conjugate(B_i) is done through gemm assuming Bt is in column major
            gemm(&(BtBinv[i*ColsPerBlockSq]), ColsPerBlock, ColsPerBlock, 'F', &(Bt[Sj[j]*ColsPerBlockSq]), ColsPerBlock, ColsPerBlock, 'F', &(C[0]), ColsPerBlock, ColsPerBlock, 'T');
            
            //Calculate Sx[ j*block_size => (j+1)*blocksize ] =  UB[ i*block_size => (i+1)*blocksize ] * C
            // Note that C actually stores C^T in row major, or C in col major.  gemm assumes C is in col major, so we're OK
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
 *   RowsPerBlock = Sparsity_Pattern.blocksize[0]
 *   BtBinv = zeros((Nnodes,NullDim,NullDim), dtype=B.dtype)
 *   S2 = Sparsity_Pattern.tocsr()
 *   for i in range(Nnodes):
 *       Bi = mat( B[S2.indices[S2.indptr[i*RowsPerBlock]:S2.indptr[i*RowsPerBlock + 1]],:] )
 *       BtBinv[i,:,:] = pinv2(Bi.H*Bi) 
 *
 * Parameters:
 *   NullDim      Number of near nullspace vectors
 *   Nnodes       Number of nodes, i.e. block rows in BSR matrix, S
 *   ColsPerBlock Columns per block in S
 *   b            In row-major form, this is B-squared, i.e. it 
 *                is each column of B multiplied against each 
 *                other column of B.  For a Nx3 B,
 *                b[:,0] = conjugate(B[:,0])*B[:,0]
 *                b[:,1] = conjugate(B[:,0])*B[:,1]
 *                b[:,2] = conjugate(B[:,0])*B[:,2]
 *                b[:,3] = conjugate(B[:,1])*B[:,1]
 *                b[:,4] = conjugate(B[:,1])*B[:,2]
 *                b[:,5] = conjugate(B[:,2])*B[:,2]
 *   BsqCols      sum(range(NullDim+1)), i.e. number of columns in b
 *   x            BtBinv (output).  Should be zeros upon entry
 *   Sp,Sj        BSR indptr and indices members for matrix, S
 *
 * Returns:
 *  BtBinv      BtBinv[i] = pseudo_invers(B_i^T*B_i), in row major format
 *              where B_i is B[colindices,:], colindices = all the nonzero
 *              column indices for block row i in S
 */          

template<class I, class T, class F>
void invert_BtB(const I NullDim, const I Nnodes,  const I ColsPerBlock, 
                const T b[],     const I BsqCols, T x[], 
                const I Sp[],    const I Sj[])
{
    //Rename to something more familiar
    const T * Bsq = b;
    T * BtBinv = x;
    
    //Declare workspace
    //const I NullDimLoc = NullDim;
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
        
        //Loop over row i in order to calculate B_i^H*B_i, where B_i is B 
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
                I BsqCounter = k*BsqCols;                   // Row-major index
                for(I m = 0; m < NullDim; m++)
                {
                    BtB[BtBcounter] += Bsq[BsqCounter];
                    BtBcounter += NullDim + 1;
                    BsqCounter += (NullDim - m);
                }
                // Do work in computing offdiagonals of BtB, noting that BtB is Hermitian and that
                // svd_solve needs BtB in column-major form, because svd_solve is Fortran
                BsqCounter = k*BsqCols;
                for(I m = 0; m < NullDim; m++)  // Loop over cols
                {
                    I counter = 1;
                    for(I n = m+1; n < NullDim; n++) // Loop over Rows
                    {
                        T elmt_bsq = Bsq[BsqCounter + counter];
                        BtB[m*NullDim + n] += conjugate(elmt_bsq);      // entry(n, m)
                        BtB[n*NullDim + m] += elmt_bsq;                 // entry(m, n)
                        counter ++;
                    }
                    BsqCounter += (NullDim - m);
                }
            } // end k loop
        } // end j loop

        // pseudo_inverse(BtB), output begins at the ptr location BtBinv offset by i*NullDimSq
        T * blockinverse = BtBinv + i*NullDimSq; 
        
        /*  svd_solve doesn't work for imaginary 
         *  Should uncomment the NullDimLoc declaration above
         *   for(I k = 0; k < NullDimSq; k++)
         *   {   blockinverse[k] = identity[k]; }
         *   svd_solve(BtB, (int) NullDimLoc, (int) NullDimLoc, blockinverse, (int) NullDimLoc, sing_vals, work, (int) work_size);
         */

        // For now, copy BtB into BtBinv and do pseudo-inverse in python.
        // Be careful to move the data from column major in BtB to row major in blockinverse.
        // Later when svd_solve is working, will need to convert from column to row major in blockinverse
        // Both of the conversions form row to col major only involve taking the conjugate, as BtB is Hermitian  
        for(I k = 0; k < NullDimSq; k++)
        {   blockinverse[k] = conjugate(BtB[k]); }
    
    } // end i loop

    delete[] BtB; 
    delete[] work;
    delete[] sing_vals; 
    delete[] identity;
}

/* For use in my_BSRinner(...)
 * B is in BSC format
 * return B(row,col), where col is the current column pointed to by Bptr
 * return Bptr pointing at the first entry past B(row,col)
 */
template<class I, class T>
inline void find_BSRmatval( const I Bj[],  const T Bx[],  const I BptrLim,
                            const I row,         I &Bptr, const T Aval[],
                                  T blockproduct[],             I &flag,
                            const I brows, const I bcols)
{
    const char trans = 'F';
    I blocksize = brows*bcols;

    while(Bptr < BptrLim)
    {
        if(Bj[Bptr] == row)
        {   
            flag = 1;
            // block multiply
            gemm(&(Aval[0]),            brows, brows, trans, 
                 &(Bx[Bptr*blocksize]), brows, bcols, trans, 
                 &(blockproduct[0]),    brows, bcols, trans);
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

/* For use in incomplete_BSRmatmat(...)
 * Calcuate <A_{row,:}, B_{:, col}>
 * A is in BSR, B is in BSC
 */
template<class I, class T>
inline void my_BSRinner( const I Ap[],  const I Aj[],    const T Ax[], 
                      const I Bp[],  const I Bj[],    const T Bx[], 
                      const I row,   const I col ,          T sum[],
                      const I brows, const I bcols)
{
    I flag;
    I Bptr = Bp[col];
    I BptrLim = Bp[col+1];
    I Ablocksize = brows*brows;
    I blocksize = brows*bcols;
    I Aoffset = Ablocksize*Ap[row];
    T blockproduct[blocksize];

    for(I index = 0; index < blocksize; index++)
    {   sum[index] = 0.0; }

    for(I colptr = Ap[row]; colptr < Ap[row+1]; colptr++)
    {
        if(Bptr == BptrLim)
        {   return;}

        I Acol = Aj[colptr];
        if(Bj[Bptr] <= Acol)
        {
            //increment sum by the block multiply A(row,col)*B(Acol,col)
            flag = 0;
            find_BSRmatval(Bj, Bx, BptrLim, Acol, Bptr, &(Ax[Aoffset]), &(blockproduct[0]), flag, brows, bcols);
            if(flag)
            {
                for(I index = 0; index < blocksize; index++)
                {   sum[index] += blockproduct[index]; }
            }
        }

        Aoffset += Ablocksize;
    }
    return;
}

/* BEWARE, this routine is designed ONLY for A, B and S such that,
 * A:  BSR,  nxn square,      sorted indices, brows x brows blocksize
 * B:  BSC,  nxm rectangular, sorted indices, brows x bcols blocksize, Bx in col major format
 * S:  BSR,  nxm rectangular, sorted indices, brows x bcols blocksize, values are overwritten
 * rows:  n
 * brows: blockrows
 * bcols: blockcols
 *
 * Calculate A*B = S, but only at the current sparsity pattern of S
 * Algorithm is naive, S(i,j) = <A_{i,:}, B_{:,j}>
 * But, we know apriori that S's sparsity pattern is a subset of A*B, 
 *      so this algorithm should work well.
 */
template<class I, class T, class F>
void incomplete_BSRmatmat( const I Ap[],  const I Aj[],    const T Ax[], 
                           const I Bp[],  const I Bj[],    const T Bx[], 
                           const I Sp[],  const I Sj[],          T Sx[], 
                           const I n,     const I brows,   const I bcols)
{
    I blocksize = brows*bcols;
    I offset = 0;
    I Srows = n/brows;

    for(I row = 0; row < Srows; row++)
    {
        I rowstart = Sp[row];
        I rowend = Sp[row+1];

        for(I colptr = rowstart; colptr < rowend; colptr++)
        {
            //calculate S(row, Sj[colptr]) = <A_{row,:}, B_{:,Sj[colptr]}>
            my_BSRinner(Ap, Aj, Ax, Bp, Bj, Bx, row, Sj[colptr], &(Sx[offset]), brows, bcols);
            offset += blocksize;
        }
    }
}

#endif
