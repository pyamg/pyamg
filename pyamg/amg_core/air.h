#ifndef AIR_H
#define AIR_H

#include <iostream>
#include <cmath>
#include <vector>
#include <iterator>
#include <cassert>
#include <limits>
#include <algorithm>

#include "linalg.h"
#include "graph.h"
#include "krylov.h"

#define F_NODE 0
#define C_NODE 1


/* Interpolate C-points by value and each F-point by value from its strongest
 * connected C-neighbor. 
 * 
 * Parameters
 * ----------
 *      Rp : const array<int> 
 *          Pre-determined row-pointer for P in CSR format
 *      Rj : array<int>
 *          Empty array for column indices for P in CSR format
 *      Cp : const array<int>
 *          Row pointer for SOC matrix, C
 *      Cj : const array<int>
 *          Column indices for SOC matrix, C
 *      Cx : const array<float>
 *          Data array for SOC matrix, C
 *      splitting : const array<int>
 *          Boolean array with 1 denoting C-points and 0 F-points
 *
 * Returns
 * -------
 * Nothing, Rj[] modified in place.
 *
 */
template<class I, class T>
void one_point_interpolation(      I Pp[],    const int Pp_size,
                                   I Pj[],   const int Pj_size,
                                   T Px[],   const int Px_size,
                             const I Cp[],  const int Cp_size,
                             const I Cj[], const int Cj_size,
                             const T Cx[],    const int Cx_size,
                             const I splitting[], const int splitting_size)
{
    I n = Pp_size-1;

    // Get enumeration of C-points, where if i is the jth C-point,
    // then pointInd[i] = j.
    std::vector<I> pointInd(n);
    pointInd[0] = 0;
    for (I i=1; i<n; i++) {
        pointInd[i] = pointInd[i-1] + splitting[i-1];
    }

    Pp[0] = 0;
    // Build interpolation operator as CSR matrix
    I next = 0;
    for (I row=0; row<n; row++) {

        // Set C-point as identity
        if (splitting[row] == C_NODE) {
            Pj[next] = pointInd[row];
            next += 1;
        }
        // For F-points, find strongest connection to C-point
        // and interpolate directly from C-point. 
        else {
            T max = -1.0;
            I ind = -1;
            T val = 0.0;
            for (I i=Cp[row]; i<Cp[row+1]; i++) {
                if (splitting[Cj[i]] == C_NODE) {
                    double vv = std::abs(Cx[i]);
                    if (vv > max) {
                        max = vv;
                        ind = Cj[i];
                        val = Cx[i];
                    }
                }
            }
            if (ind > -1) {
              Pj[next] = pointInd[ind];
              Px[next] = -val;
              next += 1;
            }
        }
        Pp[row+1] = next;
    }
}


/* Build row_pointer for approximate ideal restriction in CSR or BSR form.
 * 
 * Parameters
 * ----------
 *      Rp : array<int> 
 *          Empty row-pointer for R
 *      Cp : const array<int>
 *          Row pointer for SOC matrix, C
 *      Cj : const array<int>
 *          Column indices for SOC matrix, C
 *      Cpts : array<int>
 *          List of global C-point indices
 *      splitting : const array<int>
 *          Boolean array with 1 denoting C-points and 0 F-points
 *      distance : int, default 2
 *          Distance of F-point neighborhood to consider, options are 1 and 2.
 *
 * Returns
 * -------
 * Nothing, Rp[] modified in place.
 */
template<class I>
void approx_ideal_restriction_pass1(      I Rp[], const int Rp_size,
                                    const I Cp[], const int Cp_size,
                                    const I Cj[], const int Cj_size,
                                    const I Cpts[], const int Cpts_size,
                                    const I splitting[], const int splitting_size,
                                    const I distance = 2)
{
    I nnz = 0;
    Rp[0] = 0;

    // Deterimine number of nonzeros in each row of R.
    for (I row=0; row<Cpts_size; row++) {
        I cpoint = Cpts[row];

        // Determine number of strongly connected F-points in sparsity for R.
        for (I i=Cp[cpoint]; i<Cp[cpoint+1]; i++) {
            I this_point = Cj[i];
            if (splitting[this_point] == F_NODE) {
                nnz++;

                // Strong distance-two F-to-F connections
                if (distance == 2) {
                    for (I kk = Cp[this_point]; kk < Cp[this_point+1]; kk++){
                        if ((splitting[Cj[kk]] == F_NODE) && (this_point != cpoint)) {
                            nnz++;
                        }
                    } 
                }
            }
        }

        // Set row-pointer for this row of R (including identity on C-points).
        nnz += 1;
        Rp[row+1] = nnz; 
    }
    if ((distance != 1) && (distance != 2)) {
        std::cerr << "Error approx_ideal_restriction_pass1: can only choose distance one or two neighborhood for AIR.\n";
    }
}


/* Build column indices and data array for approximate ideal restriction
 * in CSR format.
 * 
 * Parameters
 * ----------
 *      Rp : const array<int> 
 *          Pre-determined row-pointer for R in CSR format
 *      Rj : array<int>
 *          Empty array for column indices for R in CSR format
 *      Rx : array<float>
 *          Empty array for data for R in CSR format
 *      Ap : const array<int>
 *          Row pointer for matrix A
 *      Aj : const array<int>
 *          Column indices for matrix A
 *      Ax : const array<float>
 *          Data array for matrix A
 *      Cp : const array<int>
 *          Row pointer for SOC matrix, C
 *      Cj : const array<int>
 *          Column indices for SOC matrix, C
 *      Cx : const array<float>
 *          Data array for SOC matrix, C
 *      Cpts : array<int>
 *          List of global C-point indices
 *      splitting : const array<int>
 *          Boolean array with 1 denoting C-points and 0 F-points
 *      distance : int, default 2
 *          Distance of F-point neighborhood to consider, options are 1 and 2.
 *      use_gmres : bool, default 0
 *          Use GMRES for local dense solve
 *      maxiter : int, default 10
 *          Maximum GMRES iterations
 *      precondition : bool, default True
 *          Diagonally precondition GMRES
 *
 * Returns
 * -------
 * Nothing, Rj[] and Rx[] modified in place.
 *
 * Notes
 * -----
 * Rx[] must be passed in initialized to zero.
 */
template<class I, class T>
void approx_ideal_restriction_pass2(const I Rp[], const int Rp_size,
                                          I Rj[], const int Rj_size,
                                          T Rx[], const int Rx_size,
                                    const I Ap[], const int Ap_size,
                                    const I Aj[], const int Aj_size,
                                    const T Ax[], const int Ax_size,
                                    const I Cp[], const int Cp_size,
                                    const I Cj[], const int Cj_size,
                                    const T Cx[], const int Cx_size,
                                    const I Cpts[], const int Cpts_size,
                                    const I splitting[], const int splitting_size,
                                    const I distance = 2,
                                    const I use_gmres = 0,
                                    const I maxiter = 10,
                                    const I precondition = 1 )
{
    I is_col_major = true;

    // Build column indices and data for each row of R.
    for (I row=0; row<Cpts_size; row++) {

        I cpoint = Cpts[row];
        I ind = Rp[row];

        // Set column indices for R as strongly connected F-points.
        for (I i=Cp[cpoint]; i<Cp[cpoint+1]; i++) {
            I this_point = Cj[i];
            if (splitting[this_point] == F_NODE) {
                Rj[ind] = Cj[i];
                ind +=1 ;

                // Strong distance-two F-to-F connections
                if (distance == 2) {
                    for (I kk = Cp[this_point]; kk < Cp[this_point+1]; kk++){
                        if ((splitting[Cj[kk]] == F_NODE) && (this_point != cpoint)) {
                            Rj[ind] = Cj[kk];
                            ind +=1 ;
                        }
                    } 
                }
            }
        }

        if (ind != (Rp[row+1]-1)) {
            std::cerr << "Error approx_ideal_restriction_pass2: Row pointer does not agree with neighborhood size.\n\t"
                         "ind = " << ind << ", Rp[row] = " << Rp[row] <<
                         ", Rp[row+1] = " << Rp[row+1] << "\n";
        }

        // Build local linear system as the submatrix A restricted to the neighborhood,
        // Nf, of strongly connected F-points to the current C-point, that is A0 =
        // A[Nf, Nf]^T, stored in column major form. Since A in row-major = A^T in
        // column-major, A (CSR) is iterated through and A[Nf,Nf] stored in row-major.
        I size_N = ind - Rp[row];
        std::vector<T> A0(size_N*size_N);
        I temp_A = 0;
        for (I j=Rp[row]; j<ind; j++) { 
            I this_ind = Rj[j];
            for (I i=Rp[row]; i<ind; i++) {
                // Search for indice in row of A
                I found_ind = 0;
                for (I k=Ap[this_ind]; k<Ap[this_ind+1]; k++) {
                    if (Rj[i] == Aj[k]) {
                        A0[temp_A] = Ax[k];
                        found_ind = 1;
                        temp_A += 1;
                        break;
                    }
                }
                // If indice not found, set element to zero
                if (found_ind == 0) {
                    A0[temp_A] = 0.0;
                    temp_A += 1;
                }
            }
        }

        // Build local right hand side given by b_j = -A_{cpt,N_j}, where N_j
        // is the jth indice in the neighborhood of strongly connected F-points
        // to the current C-point. 
        I temp_b = 0;
        std::vector<T> b0(size_N, 0);
        for (I i=Rp[row]; i<ind; i++) {
            // Search for indice in row of A. If indice not found, b0 has been
            // intitialized to zero.
            for (I k=Ap[cpoint]; k<Ap[cpoint+1]; k++) {
                if (Rj[i] == Aj[k]) {
                    b0[temp_b] = -Ax[k];
                    break;
                }
            }
            temp_b += 1;
        }

        // Solve linear system (least squares solves exactly when full rank)
        // s.t. (RA)_ij = 0 for (i,j) within the sparsity pattern of R. Store
        // solution in data vector for R.
        if (size_N > 0) {
            if (use_gmres) {
                dense_GMRES(&A0[0], &b0[0], &Rx[Rp[row]], size_N, is_col_major, maxiter, precondition);
            }
            else {
                least_squares(&A0[0], &b0[0], &Rx[Rp[row]], size_N, size_N, is_col_major);
            }
        }

        // Add identity for C-point in this row
        Rj[ind] = cpoint;
        Rx[ind] = 1.0;
    }
}


/* Build column indices and data array for approximate ideal restriction
 * in BSR format.
 * 
 * Parameters
 * ----------
 *      Rp : const array<int> 
 *          Pre-determined row-pointer for R in CSR format
 *      Rj : array<int>
 *          Empty array for column indices for R in CSR format
 *      Rx : array<float>
 *          Empty array for data for R in CSR format
 *      Ap : const array<int>
 *          Row pointer for matrix A
 *      Aj : const array<int>
 *          Column indices for matrix A
 *      Ax : const array<float>
 *          Data array for matrix A
 *      Cp : const array<int>
 *          Row pointer for SOC matrix, C
 *      Cj : const array<int>
 *          Column indices for SOC matrix, C
 *      Cx : const array<float>
 *          Data array for SOC matrix, C
 *      Cpts : array<int>
 *          List of global C-point indices
 *      splitting : const array<int>
 *          Boolean array with 1 denoting C-points and 0 F-points
 *      blocksize : int
 *          Blocksize of matrix (assume square blocks)
 *      distance : int, default 2
 *          Distance of F-point neighborhood to consider, options are 1 and 2.
 *      use_gmres : bool, default 0
 *          Use GMRES for local dense solve
 *      maxiter : int, default 10
 *          Maximum GMRES iterations
 *      precondition : bool, default True
 *          Diagonally precondition GMRES
 *
 * Returns
 * -------
 * Nothing, Rj[] and Rx[] modified in place.
 *
 * Notes
 * -----
 * Rx[] must be passed in initialized to zero.
 */
template<class I, class T>
void block_approx_ideal_restriction_pass2(const I Rp[], const int Rp_size,
                                                I Rj[], const int Rj_size,
                                                T Rx[], const int Rx_size,
                                          const I Ap[], const int Ap_size,
                                          const I Aj[], const int Aj_size,
                                          const T Ax[], const int Ax_size,
                                          const I Cp[], const int Cp_size,
                                          const I Cj[], const int Cj_size,
                                          const T Cx[], const int Cx_size,
                                          const I Cpts[], const int Cpts_size,
                                          const I splitting[], const int splitting_size,
                                          const I blocksize,
                                          const I distance = 2,
                                          const I use_gmres = 0,
                                          const I maxiter = 10,
                                          const I precondition = 1 )
{
    I is_col_major = true;

    // Build column indices and data for each row of R.
    for (I row=0; row<Cpts_size; row++) {

        I cpoint = Cpts[row];
        I ind = Rp[row];

        // Set column indices for R as strongly connected F-points.
        for (I i=Cp[cpoint]; i<Cp[cpoint+1]; i++) {
            I this_point = Cj[i];
            if (splitting[this_point] == F_NODE) {
                Rj[ind] = Cj[i];
                ind += 1 ;

                // Strong distance-two F-to-F connections
                if (distance == 2) {
                    for (I kk = Cp[this_point]; kk < Cp[this_point+1]; kk++){
                        if ((splitting[Cj[kk]] == F_NODE) && (this_point != cpoint)) {
                            Rj[ind] = Cj[kk];
                            ind += 1 ;
                        }
                    } 
                }
            }
        }

        if (ind != (Rp[row+1]-1)) {
            std::cerr << "Error block_approx_ideal_restriction_pass2: Row pointer does not agree with neighborhood size.\n";
        }

        // Build local linear system as the submatrix A^T restricted to the neighborhood,
        // Nf, of strongly connected F-points to the current C-point, that is A0 =
        // A[Nf, Nf]^T, stored in column major form. Since A in row-major = A^T in
        // column-major, A (CSR) is iterated through and A[Nf,Nf] stored in row-major.
        //      - Initialize A0 to zero
        I size_N = ind - Rp[row];
        I num_DOFs = size_N * blocksize;
        std::vector<T> A0(num_DOFs*num_DOFs, 0.0);
        I this_block_row = 0;

        // Add each block in strongly connected neighborhood to dense linear system.
        // For each column indice in sparsity pattern for this row of R:
        for (I j=Rp[row]; j<ind; j++) { 
            I this_ind = Rj[j];
            I this_block_col = 0;

            // For this row of A, add blocks to A0 for each entry in sparsity pattern
            for (I i=Rp[row]; i<ind; i++) {

                // Block row/column indices to normal row/column indices
                I this_row = this_block_row*blocksize;
                I this_col = this_block_col*blocksize;

                // Search for indice in row of A
                for (I k=Ap[this_ind]; k<Ap[this_ind+1]; k++) {

                    // Add block of A to dense array. If indice not found, elements
                    // in A0 have already been initialized to zero.
                    if (Rj[i] == Aj[k]) {
                        I blockx_ind = k * blocksize * blocksize;

                        // For each row in block:
                        for (I block_row=0; block_row<blocksize; block_row++) {
                            I row_maj_ind = (this_row + block_row) * num_DOFs + this_col;

                            // For each column in block:
                            for (I block_col=0; block_col<blocksize; block_col++) {

                                // Blocks of A stored in row-major in Ax
                                I Ax_ind = blockx_ind + block_row * blocksize + block_col;
                                A0[row_maj_ind + block_col] = Ax[Ax_ind];
                                if ((row_maj_ind + block_col) > num_DOFs*num_DOFs) {
                                    std::cerr << "Warning block_approx_ideal_restriction_pass2: Accessing out of bounds index building A0.\n";
                                }
                            }
                        }
                        break;
                    }
                }
                // Increase block column count
                this_block_col += 1;
            }
            // Increase block row count
            this_block_row += 1;
        }

        // Build local right hand side given by blocks b_j = -A_{cpt,N_j}, where N_j
        // is the jth indice in the neighborhood of strongly connected F-points
        // to the current C-point, and c-point the global C-point index corresponding
        // to the current row of R. RHS for each row in block, stored in b0 at indices
        //      b0[0], b0[1*num_DOFs], ..., b0[ (blocksize-1)*num_DOFs ]
        // Mapping between this ordering, say row_ind, and bsr ordering given by
        //      for each block_ind:
        //          for each row in block:    
        //              for each col in block:
        //                  row_ind = num_DOFs*row + block_ind*blocksize + col
        //                  bsr_ind = block_ind*blocksize^2 + row*blocksize + col
        std::vector<T> b0(num_DOFs * blocksize, 0);
        for (I block_ind=0; block_ind<size_N; block_ind++) {
            I temp_ind = Rp[row] + block_ind;

            // Search for indice in row of A, store data in b0. If not found,
            // b0 has been initialized to zero.
            for (I k=Ap[cpoint]; k<Ap[cpoint+1]; k++) {
                if (Rj[temp_ind] == Aj[k]) {
                    for (I this_row=0; this_row<blocksize; this_row++) {
                        for (I this_col=0; this_col<blocksize; this_col++) {
                            I row_ind = num_DOFs*this_row + block_ind*blocksize + this_col;
                            I bsr_ind = k*blocksize*blocksize + this_row*blocksize + this_col;
                            b0[row_ind] = -Ax[bsr_ind];
                        }
                    }
                    break;
                }
            }
        }

        // Solve local linear system for each row in block
        if (use_gmres) {
                
            // Apply GMRES to right-hand-side for each DOF in block
            std::vector<T> rhs(num_DOFs);
            for (I this_row=0; this_row<blocksize; this_row++) {
                I b_ind0 = num_DOFs * this_row;

                // Transfer rhs in b[] to rhs[] (solution to all systems will be stored in b[])
                for (I i=0; i<num_DOFs; i++) {
                    rhs[i] = b0[b_ind0 + i];
                }

                // Solve system using GMRES
                dense_GMRES(&A0[0], &rhs[0], &b0[b_ind0], num_DOFs,
                            is_col_major, maxiter, precondition);
            }
        }
        else {
            // Take QR of local matrix for linear solves, R stored in A0
            std::vector<T> Q = QR(&A0[0], num_DOFs, num_DOFs, is_col_major);
            
            // Solve each block based on QR decomposition
            std::vector<T> rhs(num_DOFs);
            for (I this_row=0; this_row<blocksize; this_row++) {
                I b_ind0 = num_DOFs * this_row;

                // Multiply right hand side, rhs := Q^T*b (assumes Q stored in row-major)
                for (I i=0; i<num_DOFs; i++) {
                    rhs[i] = 0.0;
                    for (I k=0; k<num_DOFs; k++) {
                        rhs[i] += b0[b_ind0 + k] * Q[col_major(k,i,num_DOFs)];
                    }
                }

                // Solve upper triangular system from QR, store solution in b0
                upper_tri_solve(&A0[0], &rhs[0], &b0[b_ind0], num_DOFs, num_DOFs, is_col_major);
            }
        }

        // Add solution for each block row to data array. See section on RHS for
        // mapping between bsr data array and row-major array solution stored in
        for (I block_ind=0; block_ind<size_N; block_ind++) {
            for (I this_row=0; this_row<blocksize; this_row++) {
                for (I this_col=0; this_col<blocksize; this_col++) {
                    I bsr_ind = Rp[row]*blocksize*blocksize + block_ind*blocksize*blocksize + 
                                this_row*blocksize + this_col;
                    I row_ind = num_DOFs*this_row + block_ind*blocksize + this_col;
                    if (std::abs(b0[row_ind]) > 1e-15) {
                        Rx[bsr_ind] = b0[row_ind];                    
                    }
                    else {
                        Rx[bsr_ind] = 0.0;                   
                    }
                }
            }
        }

        // Add identity for C-point in this block row (assume data[] initialized to 0)
        Rj[ind] = cpoint;
        I identity_ind = ind*blocksize*blocksize;
        for (I this_row=0; this_row<blocksize; this_row++) {
            Rx[identity_ind + (blocksize+1)*this_row] = 1.0;
        }
    }
}


#endif
