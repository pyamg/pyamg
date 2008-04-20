#ifndef SMOOTHED_AGGREGATION_H
#define SMOOTHED_AGGREGATION_H

#include <iostream>
#include <vector>
#include <iterator>
#include <algorithm>

#include <assert.h>
#include <cmath>

// *gelss calculates the min norm solution, using the SVD, 
//   of a rectangular matrix A and possibly multiple RHS's
extern "C" void  dgelss_(int* M,      int* N,     int* NRHS, double* A,     int* LDA, 
                        double* B,    int* LDB,   double* S, double* RCOND, int* RANK, 
                        double* WORK, int* LWORK, int* INFO );

extern "C" void  sgelss_(int* M,      int* N,     int* NRHS, double* A,     int* LDA, 
                        double* B,    int* LDB,   double* S, double* RCOND, int* RANK, 
                        double* WORK, int* LWORK, int* INFO );
 
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

/*
 * Compute A*B ==> S
 *
 * Parameters:
 * A      -  Left operand in row major
 * B      -  Right operand in column major
 * S      -  A*B, in row-major
 * Atrans -  Whether to transpose A before multiply
 * Btrans -  Whether to transpose B before multiply
 * Strans -  Whether to transpose S after multiply, Outputted in row-major         
 *
 * Returns:
 *  S = A*B
 *
 * Notes:
 *    Not fully implemented, 
 *    - Atrans and Btrans not implemented
 *    - No error checking on inputs
 *
 */

template<class I, class T>
void gemm(const T Ax[], const I Arows, const I Acols, const char Atrans, 
          const T Bx[], const I Brows, const I Bcols, const char Btrans, 
          T Sx[], const I Srows, const I Scols, const char Strans)
{
    //Add checks for dimensions, but leaving them out speeds things up
    //Add functionality for transposes

    if(Strans == 'T')
    {
        I s_counter = 0; I a_counter =0; I b_counter =0; I a_start = 0;
        for(I i = 0; i < Arows; i++)
        {
            s_counter = i;
            b_counter = 0; 
            for(I j = 0; j < Bcols; j++)
            {
                Sx[s_counter] = 0.0;
                a_counter = a_start;
                for(I k = 0; k < Brows; k++)
                {
                    //S[i,j] += Ax[i,k]*B[k,j]
                    Sx[s_counter] += Ax[a_counter]*Bx[b_counter];
                    a_counter++; b_counter++;
                }
                s_counter+=Scols;
            }
            a_start += Acols;
        }
    }
    else if(Strans == 'F')
    {
        I s_counter = 0; I a_counter =0; I b_counter =0; I a_start = 0;
        for(I i = 0; i < Arows; i++)
        {
            b_counter = 0; 
            for(I j = 0; j < Bcols; j++)
            {
                Sx[s_counter] = 0.0;
                a_counter = a_start;
                for(I k = 0; k < Brows; k++)
                {
                    //S[i,j] += A[i,k]*B[k,j]
                    Sx[s_counter] += Ax[a_counter]*Bx[b_counter];
                    a_counter++; b_counter++;
                }
                s_counter++;
            }
            a_start += Acols;
        }
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
 * Compute pseudo_inverse(A)*B ==> B
 *
 * Parameters:
 * Ax      -  Matrix to invert             (column major)
 * Bx      -  RHS (possibly multiple)      (column major)
 * Sx      -  Vector of singular values         
 * x       -  Workspace
 *
 * Arows  -  rows(A)
 * Acols  -  cols(A)
 * Bcols  -  cols(B)
 * xdim   -  size of x in double words
 *
 * Returns:
 *   pinv(A)*B ==> S
 *
 * Notes:
 *    Not fully implemented, 
 *    - No error checking on inputs (presumably LAPACK does that)
 *
 */

void svd_solve(double * Ax, int Arows, int Acols, double * Bx, int Bcols, double * Sx, double * x, int xdim)
{
    //set up unused parameters
    double RCOND = -1.0;         // Uses machine epsilon instead of the condition 
                                 // number when calculating singular value drop-tol
    int RANK;
    int INFO;
    
    dgelss_(&(Arows), &(Acols),  &(Bcols),   Ax,    &(Arows),  
               Bx,    &(Acols),     Sx,    &(RCOND), &(RANK), 
               x,    &(xdim),   &(INFO) );

    if(INFO != 0)
    {   std::cerr << "svd_solve failed with dgelss giving flag: " << INFO << '\n'; }
}
/*void svd_solve(float * Ax, int Arows, int Acols, float * Bx, int Bcols, float * Sx, float * x, int xdim)
{
    //set up unused parameters
    float RCOND = -1.0;         // Uses machine epsilon instead of the condition 
                            // number when calculating singular value drop-tol
    int RANK;
    int INFO;
    
    sgelss_(&(Arows), &(Acols),  &(Bcols),   Ax,    &(Arows),  
               Bx,    &(Acols),     Sx,    &(RCOND), &(RANK), 
                x,    &(xdim),   &(INFO) );

    if(INFO != 0)
    {   std::cout << "svd_solve failed with sgelss giving flag: " << INFO << '\n'; }

}
*/

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

//template<class I, class T>
void invert_BtB(const int NullDim, const int Nnodes,  const int ColsPerBlock, 
                const double b[],  const int BsqCols,       double     x[], 
                const int Sp[],    const int Sj[])
{
    //Rename to something more familiar
    const double * Bsq = b;
    double * BtBinv = x;
    
    //Declare workspace
    int NullDimLoc  = NullDim;
    int NullDimPone = NullDim+1;
    int NullDimSq   = NullDim*NullDim;
    int BtBinvcounter = 0;
    int work_size = 5*NullDim + 10;

    double * BtB          = new double[NullDimSq];
    double * work         = new double[work_size];
    double * sing_vals    = new double[NullDim];
    double * blockinverse = new double[NullDimSq];
    double * identity     = new double[NullDimSq];
    
    //Build an identity matrix in col major format for the Fortran routine called in svd_solve
    for(int i = 0; i < NullDimSq; i++)
    {   identity[i] = 0.0;}
    for(int i = 0; i < NullDimSq; i+= NullDimPone)
    {   identity[i] = 1.0;}


    //Loop over each row
    for(int i = 0; i < Nnodes; i++)
    {
        int rowstart = Sp[i];
        int rowend   = Sp[i+1];
        for(int k = 0; k < NullDimSq; k++)
        {   BtB[k] = 0.0; }
        
        //Loop over row i in order to calculate B_i^T*B_i, where B_i is B 
        // with the rows restricted only to the nonzero column indices of row i of S
        for(int j = rowstart; j < rowend; j++)
        {
            // Calculate absolute column index start and stop 
            //  for block column j of BSR matrix, S
            int colstart = Sj[j]*ColsPerBlock;
            int colend   = colstart + ColsPerBlock;

            //Loop over each absolute column index, k, of block column, j
            for(int k = colstart; k < colend; k++)
            {          
                // Do work in computing Diagonal of  BtB  
                int BtBcounter = 0; 
                int BsqCounter = k*BsqCols;
                for(int m = 0; m < NullDim; m++)
                {
                    BtB[BtBcounter] += Bsq[BsqCounter];
                    BtBcounter += NullDimPone;
                    BsqCounter += (NullDim - m);
                }
                // Do work in computing offdiagonals of BtB, noting that BtB is symmetric
                BsqCounter = k*BsqCols;
                for(int m = 0; m < NullDim; m++)
                {
                    int counter = 1;
                    for(int n = m+1; n < NullDim; n++)
                    {
                        double elmt_bsq = Bsq[BsqCounter + counter];
                        BtB[m*NullDim + n] += elmt_bsq;
                        BtB[n*NullDim + m] += elmt_bsq;
                        counter ++;
                    }
                    BsqCounter += (NullDim - m);
                }
            } // end k loop
        } // end j loop

        // pseudo_inverse(BtB) ==> blockinverse
        for(int k = 0; k < NullDimSq; k++)
        {   blockinverse[k] = identity[k]; }
        svd_solve(BtB, NullDimLoc, NullDimLoc, blockinverse, NullDimLoc, sing_vals, work, work_size);
          
        // Write result to output vector
        for(int k = 0; k < NullDimSq; k++)
        {   BtBinv[BtBinvcounter + k] = blockinverse[k]; }
        BtBinvcounter += NullDimSq;

    } // end i loop


    delete[] BtB; 
    delete[] work;
    delete[] sing_vals; 
    delete[] blockinverse;
    delete[] identity;
}

#endif
