#ifndef SMOOTHED_AGGREGATION_H
#define SMOOTHED_AGGREGATION_H

#include <iostream>
#include <vector>
#include <iterator>
#include <algorithm>
#include <map>

#include <assert.h>
#include <cmath>

#include "linalg.h"


/*
 * Compute a strength of connection matrix using the standard symmetric
 * Smoothed Aggregation heuristic.  Both the input and output matrices
 * are stored in CSR format.  A nonzero connection A[i,j] is considered
 * strong if:
 *
 * ..
 *     abs(A[i,j]) >= theta * sqrt( abs(A[i,i]) * abs(A[j,j]) )
 *
 * The strength of connection matrix S is simply the set of nonzero entries
 * of A that qualify as strong connections.
 *
 * Parameters
 * ----------
 * num_rows : int
 *     number of rows in A
 * theta : float
 *     stength of connection tolerance
 * Ap : array
 *     CSR row pointer
 * Aj : array
 *     CSR index array
 * Ax : array
 *     CSR data array
 * Sp : array, inplace
 *     CSR row pointer
 * Sj : array, inplace
 *     CSR index array
 * Sx : array, inplace
 *     CSR data array
 *
 * Notes
 * -----
 * Storage for S must be preallocated.  Since S will consist of a subset
 * of A's nonzero values, a conservative bound is to allocate the same
 * storage for S as is used by A.
 *
 */
template<class I, class T, class F>
void symmetric_strength_of_connection(const I n_row,
                                      const F theta,
                                      const I Ap[], const int Ap_size,
                                      const I Aj[], const int Aj_size,
                                      const T Ax[], const int Ax_size,
                                            I Sp[], const int Sp_size,
                                            I Sj[], const int Sj_size,
                                            T Sx[], const int Sx_size)
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

            if(i == j){
                // Always add the diagonal
                Sj[nnz] =   j;
                Sx[nnz] = Aij;
                nnz++;
            }
            else if(mynormsq(Aij) >= eps_Aii * diags[j]){
                //  |A(i,j)| >= theta * sqrt(|A(i,i)|*|A(j,j)|)
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
 * Parameters
 * ----------
 * n_row : int
 *     number of rows in A
 * Ap : array, n_row + 1
 *     CSR row pointer
 * Aj : array, nnz
 *     CSR column indices
 * x : array, n_row, inplace
 *     aggregate numbers for each node
 * y : array, n_row, inplace
 *     will hold Cpts upon return
 *
 * Returns
 * -------
 * int
 *     The number of aggregates (``== max(x[:]) + 1``)
 *
 * Notes
 * -----
 * - It is assumed that A is symmetric.
 * - A may contain diagonal entries (self loops)
 * - Unaggregated nodes are marked with a -1
 *
 */
template <class I>
I standard_aggregation(const I n_row,
                       const I Ap[], const int Ap_size,
                       const I Aj[], const int Aj_size,
                             I  x[], const int  x_size,
                             I  y[], const int  y_size)
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
            y[next_aggregate-1] = i;              //y stores a list of the Cpts
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
        y[next_aggregate] = i;              //y stores a list of the Cpts

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
 * Compute aggregates for a matrix A stored in CSR format
 *
 * Parameters
 * ----------
 * n_row : int
 *     number of rows in A
 * Ap : array, n_row + 1
 *     CSR row pointer
 * Aj : array, nnz
 *     CSR column indices
 * x : array, n_row, inplace
 *     aggregate numbers for each node
 * y : array, n_row, inplace
 *     will hold Cpts upon return
 *
 * Returns
 * -------
 * int
 *     The number of aggregates (``== max(x[:]) + 1``)
 *
 * Notes
 * -----
 * Differs from standard aggregation.  Each dof is considered.
 * If it has been aggregated, skip over.  Otherwise, put dof
 * and any unaggregated neighbors in an aggregate.  Results
 * in possibly much higher complexities.
 */
template <class I>
I naive_aggregation(const I n_row,
                       const I Ap[], const int Ap_size,
                       const I Aj[], const int Aj_size,
                             I  x[], const int  x_size,
                             I  y[], const int  y_size)
{
    // x[n] == 0 means i-th node has not been aggregated
    std::fill(x, x + n_row, 0);

    I next_aggregate = 1; // number of aggregates + 1

    for(I i = 0; i < n_row; i++){
        if(x[i]){ continue; } //already marked
        else
        {
            const I row_start = Ap[i];
            const I row_end   = Ap[i+1];

           //Make an aggregate out of this node and its unaggregated neighbors
           x[i] = next_aggregate;
           for(I jj = row_start; jj < row_end; jj++){
               if(!x[Aj[jj]]){
                   x[Aj[jj]] = next_aggregate;}
           }

           //y stores a list of the Cpts
           y[next_aggregate-1] = i;
           next_aggregate++;
        }
    }

    return (next_aggregate-1); //number of aggregates
}



/*
 * Compute aggregates for a matrix S stored in CSR format
 *
 * Parameters
 * ----------
 * n_row : int
 *     number of rows in S
 * Sp : array, n_row+1
 *     CSR row pointer
 * Sj : array, nnz
 *     CSR column indices
 * Sx : array, nnz
 *     CSR data array
 * x : array, n_row, inplace
 *     aggregate numbers for each node
 * y : array, n_row, inplace
 *     will hold Cpts upon return
 *
 * Returns
 * -------
 * int
 *  The number of aggregates (== max(x[:]) + 1 )
 *
 * Notes
 * -----
 * S is the strength matrix. Assumes that the strength matrix is for
 * classic strength with min norm.
 */
template <class I, class T>
I pairwise_aggregation(const I n_row,
                       const I Sp[], const int Sp_size,
                       const I Sj[], const int Sj_size,
                       const T Sx[], const int Sx_size,
                             I  x[], const int  x_size,
                             I  y[], const int  y_size)
{
    // x[n] == 0 means i-th node has not been aggregated
    std::fill(x, x + n_row, 0);

    std::vector<I> m(n_row, 0);
    for(I i = 0; i < n_row; i++){
        const I row_start = Sp[i];
        const I row_end   = Sp[i+1];
        for (I jj = row_start; jj < row_end; jj++) {
            if (Sj[jj] != i) {
                m[Sj[jj]]++;
            }
        }
    }
    std::multimap<I, I> mmap;
    std::vector<decltype(mmap.begin())> mmap_iterators(n_row);

    auto it = mmap.begin();
    for(I i = 0; i < n_row; i++){
        it = mmap.insert({m[i], i});
        mmap_iterators[i] = it;
    }

    I next_aggregate = 1; // number of aggregates + 1

    while (!mmap.empty()) {
        // select minimum of m_i.
        // Since mmap is a sorted container, first element is the minimum
        I i = mmap.begin()->second;

        const I row_start = Sp[i];
        const I row_end   = Sp[i+1];

        I j = 0;
        bool found = false;
        T max_val = std::numeric_limits<T>::lowest();

        // x stores a list of the aggregate numbers
        x[i] = next_aggregate;

        // select minimum of a_ij. (algorithm looks for minimum j in original matrix,
        // and checks whether it is strongly connected. In the code we look in
        // strength matrix only since a_ij less than a strongly connected j' implies
        // j is also strongly connected.
        for (I jj = row_start; jj < row_end; jj++) {
            if (!x[Sj[jj]] && Sx[jj] >= max_val) {
                max_val = Sx[jj];
                j = Sj[jj];
                found = true;
            }
        }
        if (found) {
            x[j] = next_aggregate;
        }
        // y stores a list of the Cpts
        y[next_aggregate-1] = i;
        for (I jj = row_start; jj < row_end; jj++) {
            if (x[Sj[jj]] == 0) {
                // to change the key of a multimap, add a new entry and remove the old entry.
                // finally update mmap_iterators with the iterator to the new entry
                auto old_it = mmap_iterators[Sj[jj]];
                auto new_it = mmap.insert({old_it->first-1, Sj[jj]});
                mmap.erase(old_it);
                mmap_iterators[Sj[jj]] = new_it;
            }
        }
        // Remove node i from mmap
        mmap.erase(mmap_iterators[i]);
        if (found) {
            x[j] = next_aggregate;
            const I row_start2 = Sp[j];
            const I row_end2   = Sp[j+1];
            for (I jj = row_start2; jj < row_end2; jj++) {
                if (x[Sj[jj]] == 0) {
                    auto old_it = mmap_iterators[Sj[jj]];
                    auto new_it = mmap.insert({old_it->first-1, Sj[jj]});
                    mmap.erase(old_it);
                    mmap_iterators[Sj[jj]] = new_it;
                }
            }
            // Remove node j from mmap
            mmap.erase(mmap_iterators[j]);
        }
        next_aggregate++;
    }

    return (next_aggregate-1); //number of aggregates
}



/*
 * Given a set of near-nullspace candidates stored in the columns of B, and
 * an aggregation operator stored in A using BSR format, this method computes
 * Ax, the data array of the tentative prolongator in BSR format, and
 * R, the coarse level near-nullspace candidates.
 *
 * The tentative prolongator A and coarse near-nullspaces candidates satisfy
 * the following relationships:
 * - ``B = A @ R``
 * - ``transpose(A) @ A = identity``
 *
 * Parameters
 * ----------
 * num_rows : int
 *     number of rows in A
 * num_cols : int
 *     number of columns in A
 * K1 : int
 *     BSR row blocksize
 * K2 : int
 *     BSR column blocksize
 * Ap : array
 *     BSR row pointer
 * Aj : array
 *     BSR index array
 * Ax : array, inplace
 *     BSR data array
 * B : array
 *     fine-level near-nullspace candidates (n_row x K2)
 * R : array, inplace
 *     coarse-level near-nullspace candidates (n_coarse x K2)
 * tol :float
 *     tolerance used to drop numerically linearly dependent vectors
 *
 * Notes
 * -----
 * - Storage for Ax and R must be preallocated.
 * - The tol parameter is applied to the candidates restricted to each
 * aggregate to discard (redundant) numerically linear dependencies.
 * For instance, if the restriction of two different fine-level candidates
 * to a single aggregate are equal, then the second candidate will not
 * contribute to the range of A.
 * - When the aggregation operator does not aggregate all fine-level
 * nodes, the corresponding rows of A will simply be zero.  In this case,
 * the two relationships mentioned above do not hold.  Instead the following
 * relationships are maintained: ``B[i,:] = A[i,:] @ R`` where ``A[i,:]`` is nonzero
 * and ``transpose(A[i,:]) * A[i,:] = 1`` where ``A[i,:]``is nonzero
 *
 */
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
            //euclidean norm exceeds the threshold. Otherwise set
            //column bj to 0.
            T scale;
            if(norm_j > threshold_j){
                scale = 1.0/norm_j;
                R_start[K2 * bj + bj] = norm_j;
            } else {
                scale = 0;

                // Explicitly zero out this column of R
                //for(I bi = 0; bi <= bj; bi++){
                //    R_start[K2 * bi + bj] = 0.0;
                //}
                // Nathan's code that just sets the diagonal entry of R to 0
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
    T operator()(const T& a, const T& b) const { return T(b.real(),-b.imag()) * a; }
};

template<class S, class T>
struct complex_norm
{
    S operator()(const T& a) const { return a.real() * a.real() + a.imag() * a.imag(); }
};

template <class I, class T>
void fit_candidates_real(const I n_row,
                         const I n_col,
                         const I   K1,
                         const I   K2,
                         const I Ap[], const int Ap_size,
                         const I Ai[], const int Ai_size,
                               T Ax[], const int Ax_size,
                         const T  B[], const int  B_size,
                               T  R[], const int  R_size,
                         const T  tol)
{ fit_candidates_common(n_row, n_col, K1, K2, Ap, Ai, Ax, B, R, tol, real_dot<T>(), real_norm<T>()); }

template <class I, class S, class T>
void fit_candidates_complex(const I n_row,
                            const I n_col,
                            const I   K1,
                            const I   K2,
                            const I Ap[], const int Ap_size,
                            const I Ai[], const int Ai_size,
                                  T Ax[], const int Ax_size,
                            const T  B[], const int  B_size,
                                  T  R[], const int  R_size,
                            const S  tol)
{ fit_candidates_common(n_row, n_col, K1, K2, Ap, Ai, Ax, B, R, tol, complex_dot<T>(), complex_norm<S,T>()); }


/*
 * Helper routine for satisfy_constraints routine called
 * by energy_prolongation_smoother(...) in smooth.py
 *
 * Parameters
 * ----------
 * rows_per_block : int
 *      rows per block in the BSR matrix, S
 * cols_per_block : int
 *      cols per block in the BSR matrix, S
 * num_block_rows : int
 *      Number of block rows, S.shape[0]/rows_per_block
 * NullDim : int
 *      Null-space dimension, i.e., the number of columns in B
 * x : array
 *      Conjugate of near-nullspace vectors, B, in row major
 * y : array
 *      S*B, in row major
 * z : array
 *      BtBinv, in row major, i.e. z[i] = pinv(B_i.H Bi), where
 *      B_i is B restricted to the neighborhood of dof of i.
 * Sp : array
 *      Row pointer array for BSR matrix S
 * Sj : array
 *      Col index array for BSR matrix S
 * Sx : array
 *      Value array for BSR matrix S
 *
 * Returns
 * -------
 * Sx is modified such that S*B = 0.  S ends up being the
 * update to the prolongator in the energy_minimization algorithm.
 *
 * Notes
 * -----
 * Principle calling routine is energy_prolongation_smoother(...) in smooth.py.
 *
 * This implements the python code:
 *
 * .. code-block:: python
 *
 *   # U is a BSR matrix, B is num_block_rows x cols_per_block x cols_per_block
 *   # UB is num_block_rows x rows_per_block x cols_per_block,  BtBinv is
 *        num_block_rows x cols_per_block x cols_per_block
 *   B  = asarray(B).reshape(-1,cols_per_block,B.shape[1])
 *   UB = asarray(UB).reshape(-1,rows_per_block,UB.shape[1])
 *   rows = csr_matrix((U.indices,U.indices,U.indptr), \
 *           shape=(U.shape[0]/rows_per_block,U.shape[1]/cols_per_block)).tocoo(copy=False).row
 *   for n,j in enumerate(U.indices):
 *      i = rows[n]
 *      Bi  = mat(B[j])
 *      UBi = UB[i]
 *      U.data[n] -= dot(UBi,dot(BtBinv[i],Bi.H))
 */
template<class I, class T, class F>
void satisfy_constraints_helper(const I rows_per_block,
                                const I cols_per_block,
                                 const I num_block_rows,
                                 const I NullDim,
                                 const T x[], const int x_size,
                                 const T y[], const int y_size,
                                 const T z[], const int z_size,
                                 const I Sp[], const int Sp_size,
                                 const I Sj[], const int Sj_size,
                                       T Sx[], const int Sx_size)
{
    //Rename to something more familiar
    const T * Bt = x;
    const T * UB = y;
    const T * BtBinv = z;

    //Declare
    I BlockSize = rows_per_block*cols_per_block;
    I NullDimSq = NullDim*NullDim;
    I NullDim_Cols = NullDim*cols_per_block;
    I NullDim_Rows = NullDim*rows_per_block;

    //C will store an intermediate mat-mat product
    std::vector<T> Update(BlockSize,0);
    std::vector<T> C(NullDim_Cols,0);
    for(I i = 0; i < NullDim_Cols; i++)
    {   C[i] = 0.0; }

    //Begin Main Loop
    for(I i = 0; i < num_block_rows; i++)
    {
        I rowstart = Sp[i];
        I rowend = Sp[i+1];

        for(I j = rowstart; j < rowend; j++)
        {
            // Calculate C = BtBinv[i*NullDimSq => (i+1)*NullDimSq]  *  B[ Sj[j]*blocksize => (Sj[j]+1)*blocksize ]^H
            // Implicit transpose of conjugate(B_i) is done through gemm assuming Bt is in column major
            gemm(&(BtBinv[i*NullDimSq]), NullDim, NullDim, 'F', &(Bt[Sj[j]*NullDim_Cols]), NullDim, cols_per_block, 'F', &(C[0]), NullDim, cols_per_block, 'T', 'T');

            // Calculate Sx[ j*BlockSize => (j+1)*blocksize ] =  UB[ i*BlockSize => (i+1)*blocksize ] * C
            // Note that C actually stores C^T in row major, or C in col major.  gemm assumes C is in col major, so we're OK
            gemm(&(UB[i*NullDim_Rows]), rows_per_block, NullDim, 'F', &(C[0]), NullDim, cols_per_block, 'F', &(Update[0]), rows_per_block, cols_per_block, 'F', 'T');

            //Update Sx
            for(I k = 0; k < BlockSize; k++)
            {   Sx[j*BlockSize + k] -= Update[k]; }
        }
    }
}


/*
 * Helper routine for energy_prolongation_smoother
 *
 * Parameters
 * ----------
 * NullDim : int
 *      Number of near nullspace vectors
 * Nnodes : int
 *      Number of nodes, i.e. number of block rows in BSR matrix, S
 * cols_per_block : int
 *      Columns per block in S
 * b : array
 *      Nnodes x BsqCols array, in row-major form.
 *      This is B-squared, i.e. it is each column of B
 *      multiplied against each other column of B.  For a Nx3 B,
 *
 *      .. code-block:: python
 *
 *          b[:,0] = conjugate(B[:,0])*B[:,0]
 *          b[:,1] = conjugate(B[:,0])*B[:,1]
 *          b[:,2] = conjugate(B[:,0])*B[:,2]
 *          b[:,3] = conjugate(B[:,1])*B[:,1]
 *          b[:,4] = conjugate(B[:,1])*B[:,2]
 *          b[:,5] = conjugate(B[:,2])*B[:,2]
 *
 * BsqCols : int
 *      sum(range(NullDim+1)), i.e. number of columns in b
 * x  : {float|complex array}
 *      Modified inplace for output.  Should be zeros upon entry
 * Sp,Sj : int array
 *      BSR indptr and indices members for matrix, S
 *
 * Returns
 * -------
 * ``BtB[i] = B_i.H*B_i`` in column major format
 * where B_i is B[colindices,:], colindices = all the nonzero
 * column indices for block row i in S
 *
 * Notes
 * -----
 * Principle calling routine is energy_prolongation_smoother(...) in smooth.py.
 *
 * Calculates the following python code:
 *
 * .. code-block:: python
 *
 *     rows_per_block = Sparsity_Pattern.blocksize[0]
 *     BtB = zeros((Nnodes,NullDim,NullDim), dtype=B.dtype)
 *     S2 = Sparsity_Pattern.tocsr()
 *     for i in range(Nnodes):
 *         Bi = mat( B[S2.indices[S2.indptr[i*rows_per_block]:S2.indptr[i*rows_per_block + 1]],:] )
 *         BtB[i,:,:] = Bi.H*Bi
 *
 */
template<class I, class T, class F>
void calc_BtB(const I NullDim,
              const I Nnodes,
              const I cols_per_block,
              const T  b[], const int  b_size,
              const I BsqCols,
                    T  x[], const int  x_size,
              const I Sp[], const int Sp_size,
              const I Sj[], const int Sj_size)
{
    //Rename to something more familiar
    const T * Bsq = b;
    T * BtB = x;

    //Declare workspace
    //const I NullDimLoc = NullDim;
    const I NullDimSq  = NullDim*NullDim;
    const I work_size  = 5*NullDim + 10;

    T * BtB_loc   = new T[NullDimSq];
    T * work      = new T[work_size];

    //Loop over each row
    for(I i = 0; i < Nnodes; i++)
    {
        const I rowstart = Sp[i];
        const I rowend   = Sp[i+1];
        for(I k = 0; k < NullDimSq; k++)
        {   BtB_loc[k] = 0.0; }

        //Loop over row i in order to calculate B_i^H*B_i, where B_i is B
        // with the rows restricted only to the nonzero column indices of row i of S
        for(I j = rowstart; j < rowend; j++)
        {
            // Calculate absolute column index start and stop
            //  for block column j of BSR matrix, S
            const I colstart = Sj[j]*cols_per_block;
            const I colend   = colstart + cols_per_block;

            //Loop over each absolute column index, k, of block column, j
            for(I k = colstart; k < colend; k++)
            {
                // Do work in computing Diagonal of  BtB_loc
                I BtBcounter = 0;
                I BsqCounter = k*BsqCols;                   // Row-major index
                for(I m = 0; m < NullDim; m++)
                {
                    BtB_loc[BtBcounter] += Bsq[BsqCounter];
                    BtBcounter += NullDim + 1;
                    BsqCounter += (NullDim - m);
                }
                // Do work in computing off-diagonals of BtB_loc, noting that BtB_loc is Hermitian and that
                // svd_solve needs BtB_loc in column-major form, because svd_solve is Fortran
                BsqCounter = k*BsqCols;
                for(I m = 0; m < NullDim; m++)  // Loop over cols
                {
                    I counter = 1;
                    for(I n = m+1; n < NullDim; n++) // Loop over Rows
                    {
                        T elmt_bsq = Bsq[BsqCounter + counter];
                        BtB_loc[m*NullDim + n] += conjugate(elmt_bsq);      // entry(n, m)
                        BtB_loc[n*NullDim + m] += elmt_bsq;                 // entry(m, n)
                        counter ++;
                    }
                    BsqCounter += (NullDim - m);
                }
            } // end k loop
        } // end j loop

        // Copy BtB_loc into BtB at the ptr location offset by i*NullDimSq
        // Note that we are moving the data from column major in BtB_loc to column major in curr_block.
        T * curr_block = BtB + i*NullDimSq;
        for(I k = 0; k < NullDimSq; k++)
        {   curr_block[k] = BtB_loc[k]; }

    } // end i loop

    delete[] BtB_loc;
    delete[] work;
}

/*
 * Calculate A*B = S, but only at the pre-existing sparsity
 * pattern of S, i.e. do an exact, but incomplete mat-mat mult.
 *
 * A, B and S must all be in BSR, may be rectangular, but the
 * indices need not be sorted.
 * Also, A.blocksize[0] must equal S.blocksize[0],
 * A.blocksize[1] must equal B.blocksize[0], and
 * B.blocksize[1] must equal S.blocksize[1]
 *
 * Parameters
 * ----------
 * Ap : {int array}
 *      BSR row pointer array
 * Aj : {int array}
 *      BSR col index array
 * Ax : {float|complex array}
 *      BSR value array
 * Bp : {int array}
 *      BSR row pointer array
 * Bj : {int array}
 *      BSR col index array
 * Bx : {float|complex array}
 *      BSR value array
 * Sp : {int array}
 *      BSR row pointer array
 * Sj : {int array}
 *      BSR col index array
 * Sx : {float|complex array}
 *      BSR value array
 * n_brow : {int}
 *      Number of block-rows in A
 * n_bcol : {int}
 *      Number of block-cols in S
 * brow_A : {int}
 *      row blocksize for A
 * bcol_A : {int}
 *      column blocksize for A
 * bcol_B : {int}
 *      column blocksize for B
 *
 * Returns
 * -------
 * Sx is modified in-place to reflect S(i,j) = <A_{i,:}, B_{:,j}>
 * but only for those entries already present in the sparsity pattern
 * of S.
 *
 * Notes
 * -----
 *
 * Algorithm is SMMP
 *
 * Principle calling routine is energy_prolongation_smoother(...) in
 * smooth.py.  Here it is used to calculate the descent direction
 * A*P_tent, but only within an accepted sparsity pattern.
 *
 * Is generally faster than the commented out incomplete_BSRmatmat(...)
 * routine below, except when S has far few nonzeros than A or B.
 *
 */
template<class I, class T, class F>
void incomplete_mat_mult_bsr(const I Ap[], const int Ap_size,
                             const I Aj[], const int Aj_size,
                             const T Ax[], const int Ax_size,
                             const I Bp[], const int Bp_size,
                             const I Bj[], const int Bj_size,
                             const T Bx[], const int Bx_size,
                             const I Sp[], const int Sp_size,
                             const I Sj[], const int Sj_size,
                                   T Sx[], const int Sx_size,
                             const I n_brow,
                             const I n_bcol,
                             const I brow_A,
                             const I bcol_A,
                             const I bcol_B )
{

    std::vector<T*> S(n_bcol);
    std::fill(S.begin(), S.end(), (T *) NULL);

    I A_blocksize = brow_A*bcol_A;
    I B_blocksize = bcol_A*bcol_B;
    I S_blocksize = brow_A*bcol_B;
    I one_by_one_blocksize = 0;
    if ((A_blocksize == B_blocksize) && (B_blocksize == S_blocksize) && (A_blocksize == 1)){
        one_by_one_blocksize = 1; }

    // Loop over rows of A
    for(I i = 0; i < n_brow; i++){

        // Initialize S to be NULL, except for the nonzero entries in S[i,:],
        // where S will point to the correct location in Sx
        I jj_start = Sp[i];
        I jj_end   = Sp[i+1];
        for(I jj = jj_start; jj < jj_end; jj++){
            S[ Sj[jj] ] = &(Sx[jj*S_blocksize]); }

        // Loop over columns in row i of A
        jj_start = Ap[i];
        jj_end   = Ap[i+1];
        for(I jj = jj_start; jj < jj_end; jj++){
            I j = Aj[jj];

            // Loop over columns in row j of B
            I kk_start = Bp[j];
            I kk_end   = Bp[j+1];
            for(I kk = kk_start; kk < kk_end; kk++){
                I k = Bj[kk];
                T * Sk = S[k];

                // If this is an allowed entry in S, then accumulate to it with a block multiply
                if (Sk != NULL){
                    if(one_by_one_blocksize){
                        // Just do a scalar multiply for the case of 1x1 blocks
                        *(Sk) += Ax[jj]*Bx[kk];
                    }
                    else{
                        gemm(&(Ax[jj*A_blocksize]), brow_A, bcol_A, 'F',
                             &(Bx[kk*B_blocksize]), bcol_A, bcol_B, 'T',
                             Sk,                    brow_A, bcol_B, 'F',
                             'F');
                    }
                }
            }
        }

        // Revert S back to it's state of all NULL
        jj_start = Sp[i];
        jj_end   = Sp[i+1];
        for(I jj = jj_start; jj < jj_end; jj++){
            S[ Sj[jj] ] = NULL; }

    }
}

/* Swap x[i] and x[j], and
 *      y[i] and y[j]
 * Use in the qsort_twoarrays funcion
 */
template<class I, class T>
inline void swap(T x[], I y[], I i, I j )
{
   T temp_t;
   I temp_i;

   temp_t = x[i];
   x[i]   = x[j];
   x[j]   = temp_t;
   temp_i = y[i];
   y[i]   = y[j];
   y[j]   = temp_i;
}

/* Apply quicksort to the array x, while simultaneously shuffling
 * the array y to mirror the swapping of entries done in x.   Then
 * aftwards x[i] and y[i] correspond to some x[k] and y[k]
 * before the sort.
 *
 * This function is particularly useful for sorting the rows (or columns)
 * of a sparse matrix according to the values
 * */
template<class I, class T>
void qsort_twoarrays( T x[], I y[], I left, I right )
{
    I i, last;

    if (left >= right)
    {   return; }
    swap( x, y, left, (left+right)/2);
    last = left;
    for (i = left+1; i <= right; i++)
    {
       if (mynorm(x[i]) < mynorm(x[left]))
       {    swap(x, y, ++last, i); }
    }
    swap(x, y, left, last);

    /* Recursive calls */
    qsort_twoarrays(x, y, left, last-1);
    qsort_twoarrays(x, y, last+1, right);
}

/*
 *  Truncate the entries in A, such that only the largest (in magnitude)
 *  k entries per row are left.   Smaller entries are zeroed out.
 *
 *  Parameters
 *      n_row      - number of rows in A
 *      k          - number of entries per row to keep
 *      Sp[]       - CSR row pointer
 *      Sj[]       - CSR index array
 *      Sx[]       - CSR data array
 *
 *
 *  Returns:
 *      Nothing, A will be stored in Sp, Sj, Sx with some entries zeroed out
 *
 */
template<class I, class T, class F>
void truncate_rows_csr(const I n_row,
                       const I k,
                       const I Sp[],  const int Sp_size,
                             I Sj[],  const int Sj_size,
                             T Sx[],  const int Sx_size)
{

    // Loop over each row of A, sort based on the entries in Sx,
    // and then truncate all but the largest k entries
    for(I i = 0; i < n_row; i++)
    {
        // Only truncate entries if row is long enough
        I rowstart = Sp[i];
        I rowend = Sp[i+1];
        if(rowend - rowstart > k)
        {
            // Sort this row
            qsort_twoarrays( Sx, Sj, rowstart, rowend-1);
            // Zero out all but the largest k
            for(I jj = rowstart; jj < rowend-k; jj++)
            {   Sx[jj] = 0.0; }
        }
    }

    return;
}

#endif
