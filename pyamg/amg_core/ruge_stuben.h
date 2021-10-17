#ifndef RUGE_STUBEN_H
#define RUGE_STUBEN_H

#include <iostream>
#include <cmath>
#include <vector>
#include <iterator>
#include <cassert>
#include <limits>
#include <algorithm>

#include "linalg.h"
#include "graph.h"

#define F_NODE 0
#define C_NODE 1
#define U_NODE 2
#define PRE_F_NODE 3

/*
 *  Compute a strength of connection matrix using the classical strength
 *  of connection measure by Ruge and Stuben. Both the input and output
 *  matrices are stored in CSR format.  An off-diagonal nonzero entry
 *  A[i,j] is considered strong if:
 *
 *      |A[i,j]| >= theta * max( |A[i,k]| )   where k != i
 *
 * Otherwise, the connection is weak.
 *
 *  Parameters
 *      num_rows   - number of rows in A
 *      theta      - stength of connection tolerance
 *      Ap[]       - CSR row pointer
 *      Aj[]       - CSR index array
 *      Ax[]       - CSR data array
 *      Sp[]       - (output) CSR row pointer
 *      Sj[]       - (output) CSR index array
 *      Sx[]       - (output) CSR data array
 *
 *
 *  Returns:
 *      Nothing, S will be stored in Sp, Sj, Sx
 *
 *  Notes:
 *      Storage for S must be preallocated.  Since S will consist of a subset
 *      of A's nonzero values, a conservative bound is to allocate the same
 *      storage for S as is used by A.
 *
 */
template<class I, class T, class F>
void classical_strength_of_connection_abs(const I n_row,
                                          const F theta,
                                          const I Ap[], const int Ap_size,
                                          const I Aj[], const int Aj_size,
                                          const T Ax[], const int Ax_size,
                                                I Sp[], const int Sp_size,
                                                I Sj[], const int Sj_size,
                                                T Sx[], const int Sx_size)
{
    I nnz = 0;
    Sp[0] = 0;

    for(I i = 0; i < n_row; i++){
        F max_offdiagonal = std::numeric_limits<F>::min();

        const I row_start = Ap[i];
        const I row_end   = Ap[i+1];

        for(I jj = row_start; jj < row_end; jj++){
            if(Aj[jj] != i){
                max_offdiagonal = std::max(max_offdiagonal, mynorm(Ax[jj]));
            }
        }

        F threshold = theta*max_offdiagonal;
        for(I jj = row_start; jj < row_end; jj++){
            F norm_jj = mynorm(Ax[jj]);

            // Add entry if it exceeds the threshold
            if(norm_jj >= threshold){
                if(Aj[jj] != i){
                    Sj[nnz] = Aj[jj];
                    Sx[nnz] = Ax[jj];
                    nnz++;
                }
            }

            // Always add the diagonal
            if(Aj[jj] == i){
                Sj[nnz] = Aj[jj];
                Sx[nnz] = Ax[jj];
                nnz++;
            }
        }

        Sp[i+1] = nnz;
    }
}

template<class I, class T>
void classical_strength_of_connection_min(const I n_row,
                                          const T theta,
                                          const I Ap[], const int Ap_size,
                                          const I Aj[], const int Aj_size,
                                          const T Ax[], const int Ax_size,
                                                I Sp[], const int Sp_size,
                                                I Sj[], const int Sj_size,
                                                T Sx[], const int Sx_size)
{
    I nnz = 0;
    Sp[0] = 0;

    for(I i = 0; i < n_row; i++){
        T max_offdiagonal = 0.0;

        const I row_start = Ap[i];
        const I row_end   = Ap[i+1];

        for(I jj = row_start; jj < row_end; jj++){
            if(Aj[jj] != i){
                max_offdiagonal = std::max(max_offdiagonal, -Ax[jj]);
            }
        }

        T threshold = theta*max_offdiagonal;
        for(I jj = row_start; jj < row_end; jj++){
            T norm_jj = -Ax[jj];

            // Add entry if it exceeds the threshold
            if(norm_jj >= threshold){
                if(Aj[jj] != i){
                    Sj[nnz] = Aj[jj];
                    Sx[nnz] = Ax[jj];
                    nnz++;
                }
            }

            // Always add the diagonal
            if(Aj[jj] == i){
                Sj[nnz] = Aj[jj];
                Sx[nnz] = Ax[jj];
                nnz++;
            }
        }

        Sp[i+1] = nnz;
    }
}

/*
 *  Compute the maximum in magnitude row value for a CSR matrix
 *
 *  Parameters
 *      num_rows   - number of rows in A
 *      Ap[]       - CSR row pointer
 *      Aj[]       - CSR index array
 *      Ax[]       - CSR data array
 *       x[]       - num_rows array
 *
 *  Returns:
 *      Nothing, x[i] will hold row i's maximum magnitude entry
 *
 */
template<class I, class T, class F>
void maximum_row_value(const I n_row,
                              T x[], const int  x_size,
                       const I Ap[], const int Ap_size,
                       const I Aj[], const int Aj_size,
                       const T Ax[], const int Ax_size)
{

    for(I i = 0; i < n_row; i++){
        F max_entry = std::numeric_limits<F>::min();

        const I row_start = Ap[i];
        const I row_end   = Ap[i+1];

        // Find this row's max entry
        for(I jj = row_start; jj < row_end; jj++){
            max_entry = std::max(max_entry, mynorm(Ax[jj]) );
        }

        x[i] = max_entry;
    }
}


/* Compute a C/F (coarse-fine( splitting using the classical coarse grid
 * selection method of Ruge and Stuben.  The strength of connection matrix S,
 * and its transpose T, are stored in CSR format.  Upon return, the  splitting
 * array will consist of zeros and ones, where C-nodes (coarse nodes) are
 * marked with the value 1 and F-nodes (fine nodes) with the value 0.
 *
 * Parameters:
 *   n_nodes   - number of rows in A
 *   C_rowptr[]      - CSR row pointer array for SOC matrix
 *   C_colinds[]      - CSR column index array for SOC matrix
 *   Tp[]      - CSR row pointer array for transpose of SOC matrix
 *   Tj[]      - CSR column index array for transpose of SOC matrix
 *   influence - array that influences splitting (values stored here are
 *               added to lambda for each point)
 *   splitting - array to store the C/F splitting
 *
 * Notes:
 *   The splitting array must be preallocated
 *
 */
template<class I>
void rs_cf_splitting(const I n_nodes,
                     const I C_rowptr[], const int C_rowptr_size,
                     const I C_colinds[], const int C_colinds_size,
                     const I Tp[], const int Tp_size,
                     const I Tj[], const int Tj_size,
                     const I influence[], const int influence_size,
                           I splitting[], const int splitting_size)
{
    std::vector<I> lambda(n_nodes,0);

    // Compute initial lambda based on C^T
    I lambda_max = 0;
    for (I i = 0; i < n_nodes; i++) {
        lambda[i] = Tp[i+1] - Tp[i] + influence[i];
        if (lambda[i] > lambda_max) {
            lambda_max = lambda[i];
        }
    }

    // For each value of lambda, create an interval of nodes with that value
    //      interval_ptr - the first index of the interval
    //      interval_count - the number of indices in that interval
    //      index_to_node - the node located at a given index
    //      node_to_index - the index of a given node
    lambda_max = lambda_max*2;
    if (n_nodes+1 > lambda_max) {
        lambda_max = n_nodes+1;
    }

    std::vector<I> interval_ptr(lambda_max,0);
    std::vector<I> interval_count(lambda_max,0);
    std::vector<I> index_to_node(n_nodes);
    std::vector<I> node_to_index(n_nodes);

    for (I i = 0; i < n_nodes; i++) {
        interval_count[lambda[i]]++;
    }
    for (I i = 0, cumsum = 0; i < lambda_max; i++) {
        interval_ptr[i] = cumsum;
        cumsum += interval_count[i];
        interval_count[i] = 0;
    }
    for (I i = 0; i < n_nodes; i++) {
        I lambda_i = lambda[i];

        I index    = interval_ptr[lambda_i] + interval_count[lambda_i];
        index_to_node[index] = i;
        node_to_index[i]     = index;
        interval_count[lambda_i]++;
    }

    std::fill(splitting, splitting + n_nodes, U_NODE);

    // All nodes with no neighbors become F nodes
    for (I i = 0; i < n_nodes; i++) {
        if (lambda[i] == 0 || (lambda[i] == 1 && Tj[Tp[i]] == i))
            splitting[i] = F_NODE;
    }

    // Add elements to C and F, in descending order of lambda
    for (I top_index=(n_nodes - 1); top_index>-1; top_index--) {
        
        I i        = index_to_node[top_index];
        I lambda_i = lambda[i];

        // Remove i from its interval
        interval_count[lambda_i]--;

        // ----------------- Sorting every iteration = O(n^2) complexity ----------------- //
        // Search over this interval to make sure we process nodes in descending node order
        // I max_node = i;
        // I max_index = top_index;
        // for (I j = interval_ptr[lambda_i]; j < interval_ptr[lambda_i] + interval_count[lambda_i]; j++) {
        //     if (index_to_node[j] > max_node) {
        //         max_node = index_to_node[j];
        //         max_index = j;
        //     }
        // }
        // node_to_index[index_to_node[top_index]] = max_index;
        // node_to_index[index_to_node[max_index]] = top_index;
        // std::swap(index_to_node[top_index], index_to_node[max_index]);
        // i = index_to_node[top_index];

        // If maximum lambda = 0, break out of loop
        if (lambda[i] <= 0) {
            break;
        }

        // If node is unmarked, set maximum node as C-node and modify
        // lambda values in neighborhood
        if ( splitting[i] == U_NODE) {
            splitting[i] = C_NODE;

            // For each j in S^T_i /\ U, mark j as tentative F-point
            for (I jj = Tp[i]; jj < Tp[i+1]; jj++) {
                I j = Tj[jj];
                if(splitting[j] == U_NODE) {
                    splitting[j] = PRE_F_NODE;
                }
            }

            // For each j in S^T_i /\ U marked as tentative F-point, modify lamdba
            // values for neighborhood of j
            for (I jj = Tp[i]; jj < Tp[i+1]; jj++)
            {
                I j = Tj[jj];
                if(splitting[j] == PRE_F_NODE)
                {
                    splitting[j] = F_NODE;
                    
                    // For each k in S_j /\ U, modify lambda value, lambda_k += 1
                    for (I kk = C_rowptr[j]; kk < C_rowptr[j+1]; kk++){
                        I k = C_colinds[kk];

                        if(splitting[k] == U_NODE){

                            // Move k to the end of its current interval
                            if(lambda[k] >= n_nodes - 1) {
                                continue;
                            }

                            I lambda_k = lambda[k];
                            I old_pos  = node_to_index[k];
                            I new_pos  = interval_ptr[lambda_k] + interval_count[lambda_k] - 1;

                            node_to_index[index_to_node[old_pos]] = new_pos;
                            node_to_index[index_to_node[new_pos]] = old_pos;
                            std::swap(index_to_node[old_pos], index_to_node[new_pos]);

                            // Update intervals
                            interval_count[lambda_k]   -= 1;
                            interval_count[lambda_k+1] += 1; //invalid write!
                            interval_ptr[lambda_k+1]    = new_pos;

                            // Increment lambda_k
                            lambda[k]++;
                        }
                    }
                }
            }

            // For each j in S_i /\ U, set lambda_j -= 1
            for (I jj = C_rowptr[i]; jj < C_rowptr[i+1]; jj++) {
                I j = C_colinds[jj];
                // Decrement lambda for node j
                if (splitting[j] == U_NODE) {
                    if (lambda[j] == 0) {
                        continue;
                    }

                    // Move j to the beginning of its current interval
                    I lambda_j = lambda[j];
                    I old_pos  = node_to_index[j];
                    I new_pos  = interval_ptr[lambda_j];

                    node_to_index[index_to_node[old_pos]] = new_pos;
                    node_to_index[index_to_node[new_pos]] = old_pos;
                    std::swap(index_to_node[old_pos],index_to_node[new_pos]);

                    // Update intervals
                    interval_count[lambda_j]   -= 1;
                    interval_count[lambda_j-1] += 1;
                    interval_ptr[lambda_j]     += 1;
                    interval_ptr[lambda_j-1]    = interval_ptr[lambda_j] - interval_count[lambda_j-1];

                    // Decrement lambda_j
                    lambda[j]--;
                }
            }
        }
    }

    // Set any unmarked nodes as F-points
    for (I i=0; i<n_nodes; i++) {
        if (splitting[i] == U_NODE) {
            splitting[i] = F_NODE;
        }
    }
}


template<class I>
void rs_cf_splitting_pass2(const I n_nodes,
                           const I C_rowptr[], const int C_rowptr_size,
                           const I C_colinds[], const int C_colinds_size,
                                 I splitting[], const int splitting_size)
{
    // For each F-point
    for (I row=0; row<n_nodes; row++) {
        if (splitting[row] == F_NODE) {

            // Tentative C-point count
            I Cpt0 = -1;

            // For each j in S_row /\ F, test dependence of j on S_row /\ C
            for (I jj=C_rowptr[row]; jj<C_rowptr[row+1]; jj++) {
                I j = C_colinds[jj];

                if (splitting[j] == F_NODE) {

                    // Test dependence, i.e. check that S_j /\ S_row /\ C is
                    // nonempty. This is simply checking that nodes j and row
                    // have a common strong C-point connection.
                    bool dependence = false;
                    for (I ii=C_rowptr[row]; ii<C_rowptr[row+1]; ii++) {
                        I row_ind = C_colinds[ii];
                        if (splitting[row_ind] == C_NODE) {
                            for (I kk=C_rowptr[j]; kk<C_rowptr[j+1]; kk++) {
                                if (C_colinds[kk] == row_ind) {
                                    dependence = true;
                                }
                            }
                        }
                        if (dependence) {
                            break;
                        }
                    }

                    // Node j passed dependence test
                    if (dependence) {
                        continue;   
                    }
                    // Node j did not pass dependence test
                    else {
                        // If no tentative C-point, mark j as tentative C-point
                        if (Cpt0 < 0) {
                            Cpt0 = j;
                            splitting[j] = C_NODE;
                        }
                        // If there is a tentative C-point already, put it back in
                        // set of F-points and mark j as tentative C-point.
                        else {
                            splitting[Cpt0] = F_NODE;
                            Cpt0 = j;
                            splitting[j] = C_NODE;
                        }
                    }
                }
            }
        }
    }
}


/*
 *  Compute a CLJP splitting
 *
 *  Parameters
 *      n          - number of rows in A (number of vertices)
 *      Sp[]       - CSR row pointer (strength matrix)
 *      Sj[]       - CSR index array
 *      Tp[]       - CSR row pointer (transpose of the strength matrix)
 *      Tj[]       - CSR index array
 *      splitting  - array to store the C/F splitting
 *      colorflag  - flag to indicate coloring
 *
 *  Notes:
 *      The splitting array must be preallocated.
 *      CLJP naive since it requires the transpose.
 */

template<class I>
void cljp_naive_splitting(const I n,
                          const I Sp[], const int Sp_size,
                          const I Sj[], const int Sj_size,
                          const I Tp[], const int Tp_size,
                          const I Tj[], const int Tj_size,
                                I splitting[], const int splitting_size,
                          const I colorflag)
{
  // initialize sizes
  int ncolors;
  I unassigned = n;
  I nD;
  int nnz = Sp[n];

  // initialize vectors
  // complexity = 5n
  // storage = 4n
  std::vector<int> edgemark(nnz,1);
  std::vector<int> coloring(n);
  std::vector<double> weight(n);
  std::vector<I> D(n,0);      // marked nodes  in the ind set
  std::vector<I> Dlist(n,0);      // marked nodes  in the ind set
  std::fill(splitting, splitting + n, U_NODE);
  int * c_dep_cache = new int[n];
  std::fill_n(c_dep_cache, n, -1);

  // INITIALIZE WEIGHTS
  // complexity = O(n^2)?!? for coloring
  // or
  // complexity = n for random
  if(colorflag==1){ // with coloring
    //vertex_coloring_jones_plassmann(n, Sp, Sj, &coloring[0],&weight[0]);
    //vertex_coloring_IDO(n, Sp, Sj, &coloring[0]);
    vertex_coloring_mis(n, Sp, Sp_size, Sj, Sj_size, &coloring[0], n);
    ncolors = *std::max_element(coloring.begin(), coloring.end()) + 1;
    for(I i=0; i < n; i++){
      weight[i] = double(coloring[i])/double(ncolors);
    }
  }
  else {
    srand(2448422);
    for(I i=0; i < n; i++){
      weight[i] = double(rand())/RAND_MAX;
    }
  }

  for(I i=0; i < n; i++){
    for(I jj = Sp[i]; jj < Sp[i+1]; jj++){
      I j = Sj[jj];
      if(i != j) {
        weight[j]++;
      }
    }
  }
  // end INITIALIZE WEIGHTS

  // SELECTION LOOP
  I pass = 0;
  while(unassigned > 0){
    pass++;

    // SELECT INDEPENDENT SET
    // find i such that w_i > w_j for all i in union(S_i,S_i^T)
    nD = 0;
    for(I i=0; i<n; i++){
      if(splitting[i]==U_NODE){
        D[i] = 1;
        // check row (S_i^T)
        for(I jj = Sp[i]; jj < Sp[i+1]; jj++){
          I j = Sj[jj];
          if(splitting[j]==U_NODE && weight[j]>weight[i]){
            D[i] = 0;
            break;
          }
        }
        // check col (S_i)
        if(D[i] == 1) {
          for(I jj = Tp[i]; jj < Tp[i+1]; jj++){
            I j = Tj[jj];
            if(splitting[j]==U_NODE && weight[j]>weight[i]){
              D[i] = 0;
              break;
            }
          }
        }
        if(D[i] == 1) {
          Dlist[nD] = i;
          unassigned--;
          nD++;
        }
      }
      else{
        D[i]=0;
      }
    } // end for
    for(I i = 0; i < nD; i++) {
      splitting[Dlist[i]] = C_NODE;
    }
    // end SELECT INDEPENDENT SET

    // UPDATE WEIGHTS
    // P5
    // nbrs that influence C points are not good C points
    for(I iD=0; iD < nD; iD++){
      I c = Dlist[iD];
      for(I jj = Sp[c]; jj < Sp[c+1]; jj++){
        I j = Sj[jj];
        // c <---j
        if(splitting[j]==U_NODE && edgemark[jj] != 0){
          edgemark[jj] = 0;  // "remove" edge
          weight[j]--;
          if(weight[j]<1){
            splitting[j] = F_NODE;
            unassigned--;
          }
        }
      }
    } // end P5

    // P6
    // If k and j both depend on c, a C point, and j influces k, then j is less
    // valuable as a C point.
    for(I iD=0; iD < nD; iD++){
      I c = Dlist[iD];
      for(I jj = Tp[c]; jj < Tp[c+1]; jj++){
        I j = Tj[jj];
        if(splitting[j]==U_NODE)                 // j <---c
          c_dep_cache[j] = c;
      }

      for(I jj = Tp[c]; jj < Tp[c+1]; jj++) {
        I j = Tj[jj];
        for(I kk = Sp[j]; kk < Sp[j+1]; kk++) {
          I k = Sj[kk];
          if(splitting[k] == U_NODE && edgemark[kk] != 0) { // j <---k
            // does c ---> k ?
            if(c_dep_cache[k] == c) {
              edgemark[kk] = 0; // remove edge
              weight[k]--;
              if(weight[k] < 1) {
                splitting[k] = F_NODE;
                unassigned--;
                //kk = Tp[j+1]; // to break second loop
              }
            }
          }
        }
      }
    } // end P6
  }
  // end SELECTION LOOP

  for(I i = 0; i < Sp[n]; i++){
    if(edgemark[i] == 0){
      edgemark[i] = -1;
    }
  }
  for(I i = 0; i < n; i++){
    if(splitting[i] == U_NODE){
      splitting[i] = F_NODE;
    }
  }
  delete[] c_dep_cache;
}


/*
 *   Produce the Ruge-Stuben prolongator using "Direct Interpolation"
 *
 *
 *   The first pass uses the strength of connection matrix 'S'
 *   and C/F splitting to compute the row pointer for the prolongator.
 *
 *   The second pass fills in the nonzero entries of the prolongator
 *
 *   Reference:
 *      Page 479 of "Multigrid"
 *
 */
template<class I>
void rs_direct_interpolation_pass1(const I n_nodes,
                                   const I Sp[], const int Sp_size,
                                   const I Sj[], const int Sj_size,
                                   const I splitting[], const int splitting_size,
                                         I Bp[], const int Bp_size)
{
    I nnz = 0;
    Bp[0] = 0;
    for(I i = 0; i < n_nodes; i++){
        if( splitting[i] == C_NODE ){
            nnz++;
        } else {
            for(I jj = Sp[i]; jj < Sp[i+1]; jj++){
                if ( (splitting[Sj[jj]] == C_NODE) && (Sj[jj] != i) )
                    nnz++;
            }
        }
        Bp[i+1] = nnz;
    }
}


template<class I, class T>
void rs_direct_interpolation_pass2(const I n_nodes,
                                   const I Ap[], const int Ap_size,
                                   const I Aj[], const int Aj_size,
                                   const T Ax[], const int Ax_size,
                                   const I Sp[], const int Sp_size,
                                   const I Sj[], const int Sj_size,
                                   const T Sx[], const int Sx_size,
                                   const I splitting[], const int splitting_size,
                                   const I Bp[], const int Bp_size,
                                         I Bj[], const int Bj_size,
                                         T Bx[], const int Bx_size)
{

    for(I i = 0; i < n_nodes; i++){
        if(splitting[i] == C_NODE){
            Bj[Bp[i]] = i;
            Bx[Bp[i]] = 1;
        } else {
            T sum_strong_pos = 0, sum_strong_neg = 0;
            for(I jj = Sp[i]; jj < Sp[i+1]; jj++){
                if ( (splitting[Sj[jj]] == C_NODE) && (Sj[jj] != i) ){
                    if (Sx[jj] < 0)
                        sum_strong_neg += Sx[jj];
                    else
                        sum_strong_pos += Sx[jj];
                }
            }

            T sum_all_pos = 0, sum_all_neg = 0;
            T diag = 0;
            for(I jj = Ap[i]; jj < Ap[i+1]; jj++){
                if (Aj[jj] == i){
                    diag += Ax[jj];
                } else {
                    if (Ax[jj] < 0)
                        sum_all_neg += Ax[jj];
                    else
                        sum_all_pos += Ax[jj];
                }
            }

            T alpha = sum_all_neg / sum_strong_neg;
            T beta  = sum_all_pos / sum_strong_pos;

            if (sum_strong_pos == 0){
                diag += sum_all_pos;
                beta = 0;
            }

            T neg_coeff = -alpha/diag;
            T pos_coeff = -beta/diag;

            I nnz = Bp[i];
            for(I jj = Sp[i]; jj < Sp[i+1]; jj++){
                if ( (splitting[Sj[jj]] == C_NODE) && (Sj[jj] != i) ){
                    Bj[nnz] = Sj[jj];
                    if (Sx[jj] < 0)
                        Bx[nnz] = neg_coeff * Sx[jj];
                    else
                        Bx[nnz] = pos_coeff * Sx[jj];
                    nnz++;
                }
            }
        }
    }


    std::vector<I> map(n_nodes);
    for(I i = 0, sum = 0; i < n_nodes; i++){
        map[i]  = sum;
        sum    += splitting[i];
    }
    for(I i = 0; i < Bp[n_nodes]; i++){
        Bj[i] = map[Bj[i]];
    }
}




template<class I, class T>
void rs_standard_interpolation(const I n_nodes,
                               const I Ap[], const I Aj[], const T Ax[],
                               const I Sp[], const I Sj[], const T Sx[],
                               const I Tp[], const I Tj[], const T Tx[],
                                     I Bp[],       I Bj[],       T Bx[])
{
    // Not implemented
}


/* Helper function for compatible relaxation to perform steps 3.1d - 3.1f
 * in Falgout / Brannick (2010).
 *
 * Input:
 * ------
 * A_rowptr : const {int array}
 *      Row pointer for sparse matrix in CSR format.
 * A_colinds : const {int array}
 *      Column indices for sparse matrix in CSR format.
 * B : const {float array}
 *      Target near null space vector for computing candidate set measure.
 * e : {float array}
 *      Relaxed vector for computing candidate set measure.
 * indices : {int array}
 *      Array of indices, where indices[0] = the number of F indices, nf,
 *      followed by F indices in elements 1:nf, and C indices in (nf+1):n.
 * splitting : {int array}
 *      Integer array with current C/F splitting of nodes, 0 = C-point,
 *      1 = F-point.
 * gamma : {float array}
 *      Preallocated vector to store candidate set measure.
 * thetacs : const {float}
 *      Threshold for coarse grid candidates from set measure.
 *
 * Returns:
 * --------
 * Nothing, updated C/F-splitting and corresponding indices modified in place.
 */
template<class I, class T>
void cr_helper(const I A_rowptr[], const int A_rowptr_size,
               const I A_colinds[], const int A_colinds_size,
               const T B[], const int B_size,
               T e[], const int e_size,
               I indices[], const int indices_size,
               I splitting[], const int splitting_size,
               T gamma[], const int gamma_size,
               const T thetacs  )
{
    I n = splitting_size;
    I &num_Fpts = indices[0];

    // Steps 3.1d, 3.1e in Falgout / Brannick (2010)
    // Divide each element in e by corresponding index in initial target vector.
    // Get inf norm of new e.
    T inf_norm = 0;
    for (I i=1; i<(num_Fpts+1); i++) {
        I pt = indices[i];
        e[pt] = std::abs(e[pt] / B[pt]);
        if (e[pt] > inf_norm) {
            inf_norm = e[pt];
        }
    }

    // Compute candidate set measure, pick coarse grid candidates.
    std::vector<I> Uindex;
    for (I i=1; i<(num_Fpts+1); i++) {
        I pt = indices[i];
        gamma[pt] = e[pt] / inf_norm;
        if (gamma[pt] > thetacs) {
            Uindex.push_back(pt);
        }
    }
    I set_size = Uindex.size();

    // Step 3.1f in Falgout / Brannick (2010)
    // Find weights: omega_i = |N_i\C| + gamma_i
    std::vector<T> omega(n,0);
    for (I i=0; i<set_size; i++) {
        I pt = Uindex[i];
        I num_neighbors = 0;
        I A_ind0 = A_rowptr[pt];
        I A_ind1 = A_rowptr[pt+1];
        for (I j=A_ind0; j<A_ind1; j++) {
            I neighbor = A_colinds[j];
            if (splitting[neighbor] == 0) {
                num_neighbors += 1;
            }
        }
        omega[pt] = num_neighbors + gamma[pt];
    }

    // Form maximum independent set
    while (true) {
        // 1. Add point i in U with maximal weight to C
        T max_weight = 0;
        I new_pt = -1;
        for (I i=0; i<set_size; i++) {
            I pt = Uindex[i];
            if (omega[pt] > max_weight) {
                max_weight = omega[pt];
                new_pt = pt;
            }
        }
        // If all points have zero weight (index set is empty) break loop
        if (new_pt < 0) {
            break;
        }
        splitting[new_pt] = 1;
        gamma[new_pt] = 0;

        // 2. Remove from candidate set all nodes connected to
        // new C-point by marking weight zero.
        std::vector<I> neighbors;
        I A_ind0 = A_rowptr[new_pt];
        I A_ind1 = A_rowptr[new_pt+1];
        for (I i=A_ind0; i<A_ind1; i++) {
            I temp = A_colinds[i];
            neighbors.push_back(temp);
            omega[temp] = 0;
        }

        // 3. For each node removed in step 2, set the weight for
        // each of its neighbors still in the candidate set +1.
        I num_neighbors = neighbors.size();
        for (I i=0; i<num_neighbors; i++) {
            I pt = neighbors[i];
            I A_ind0 = A_rowptr[pt];
            I A_ind1 = A_rowptr[pt+1];
            for (I j=A_ind0; j<A_ind1; j++) {
                I temp = A_colinds[j];
                if (omega[temp] != 0) {
                    omega[temp] += 1;
                }
            }
        }
    }

    // Reorder indices array, with the first element giving the number
    // of F indices, nf, followed by F indices in elements 1:nf, and
    // C indices in (nf+1):n. Note, C indices sorted largest to smallest.
    num_Fpts = 0;
    I next_Find = 1;
    I next_Cind = n;
    for (I i=0; i<n; i++) {
        if (splitting[i] == 0) {
            indices[next_Find] = i;
            next_Find += 1;
            num_Fpts += 1;
        }
        else {
            indices[next_Cind] = i;
            next_Cind -= 1;
        }
    }
}



//#define NodeType char
// // The following function closely approximates the
// // method described in the 1987 Ruge-Stuben paper
//
//template<class I, class T>
//void rs_interpolation(const I n_nodes,
//        const I Ap[], const I Aj[], const T Ax[],
//        const I Sp[], const I Sj[], const T Sx[],
//        const I Tp[], const I Tj[], const T Tx[],
//        std::vector<I> * Bp, std::vector<I> * Bj, std::vector<T> * Bx){
//
//    std::vector<I> lambda(n_nodes,0);
//
//    //compute lambdas
//    for(I i = 0; i < n_nodes; i++){
//        lambda[i] = Tp[i+1] - Tp[i];
//    }
//
//
//    //for each value of lambda, create an interval of nodes with that value
//    // ptr - is the first index of the interval
//    // count - is the number of indices in that interval
//    // index to node - the node located at a given index
//    // node to index - the index of a given node
//    std::vector<I> interval_ptr(n_nodes,0);
//    std::vector<I> interval_count(n_nodes,0);
//    std::vector<I> index_to_node(n_nodes);
//    std::vector<I> node_to_index(n_nodes);
//
//    for(I i = 0; i < n_nodes; i++){
//        interval_count[lambda[i]]++;
//    }
//    for(I i = 0, cumsum = 0; i < n_nodes; i++){
//        interval_ptr[i] = cumsum;
//        cumsum += interval_count[i];
//        interval_count[i] = 0;
//    }
//    for(I i = 0; i < n_nodes; i++){
//        I lambda_i = lambda[i];
//        I index    = interval_ptr[lambda_i]+interval_count[lambda_i];
//        index_to_node[index] = i;
//        node_to_index[i]     = index;
//        interval_count[lambda_i]++;
//    }
//
//
//
//
//
//    std::vector<NodeType> NodeSets(n_nodes,U_NODE);
//
//    //Now add elements to C and F, in decending order of lambda
//    for(I top_index = n_nodes - 1; top_index > -1; top_index--){
//        I i        = index_to_node[top_index];
//        I lambda_i = lambda[i];
//#ifdef DEBUG
//        {
//#ifdef DEBUG_PRINT
//            std::cout << "top_index " << top_index << std::endl;
//            std::cout << "i         " << i << std::endl;
//            std::cout << "lambda_i  " << lambda_i << std::endl;
//
//            for(I i = 0; i < n_nodes; i++){
//                std::cout << i << "=";
//                if(NodeSets[i] == U_NODE)
//                    std::cout << "U";
//                else if(NodeSets[i] == F_NODE)
//                    std::cout << "F";
//                else
//                    std::cout << "C";
//                std::cout << " ";
//            }
//            std::cout << std::endl;
//
//            std::cout << "node_to_index" << std::endl;
//            for(I i = 0; i < n_nodes; i++){
//                std::cout << i << "->" << node_to_index[i] << "  ";
//            }
//            std::cout << std::endl;
//            std::cout << "index_to_node" << std::endl;
//            for(I i = 0; i < n_nodes; i++){
//                std::cout << i << "->" << index_to_node[i] << "  ";
//            }
//            std::cout << std::endl;
//
//            std::cout << "interval_count ";
//            for(I i = 0; i < n_nodes; i++){
//                std::cout << interval_count[i] << " ";
//            }
//            std::cout << std::endl;
//#endif
//
//            //make sure arrays are correct
//            for(I n = 0; n < n_nodes; n++){
//                assert(index_to_node[node_to_index[n]] == n);
//            }
//
//            //make sure intervals are reasonable
//            I sum_intervals = 0;
//            for(I n = 0; n < n_nodes; n++){
//                assert(interval_count[n] >= 0);
//                if(interval_count[n] > 0){
//                    assert(interval_ptr[n] == sum_intervals);
//                }
//                sum_intervals += interval_count[n];
//            }
//            assert(sum_intervals == top_index+1);
//
//
//            if(interval_count[lambda_i] <= 0){
//                std::cout << "top_index " << top_index << std::endl;
//                std::cout << "lambda_i " << lambda_i << std::endl;
//                std::cout << "interval_count[lambda_i] " << interval_count[lambda_i] << std::endl;
//                std::cout << "top_index " << top_index << std::endl;
//                std::cout << "i         " << i << std::endl;
//                std::cout << "lambda_i  " << lambda_i << std::endl;
//            }
//
//
//            for(I n = 0; n <= top_index; n++){
//                assert(NodeSets[index_to_node[n]] != C_NODE);
//            }
//        }
//        assert(node_to_index[i] == top_index);
//        assert(interval_ptr[lambda_i] + interval_count[lambda_i] - 1 == top_index);
//        //max interval should have at least one element
//        assert(interval_count[lambda_i] > 0);
//#endif
//
//
//        //remove i from its interval
//        interval_count[lambda_i]--;
//
//
//        if(NodeSets[i] == F_NODE){
//            continue;
//        } else {
//            assert(NodeSets[i] == U_NODE);
//
//            NodeSets[i] = C_NODE;
//
//            //For each j in S^T_i /\ U
//            for(I jj = Tp[i]; jj < Tp[i+1]; jj++){
//                I j = Tj[jj];
//
//                if(NodeSets[j] == U_NODE){
//                    NodeSets[j] = F_NODE;
//
//                    //For each k in S_j /\ U
//                    for(I kk = Sp[j]; kk < Sp[j+1]; kk++){
//                        I k = Sj[kk];
//
//                        if(NodeSets[k] == U_NODE){
//                            //move k to the end of its current interval
//                            assert(lambda[j] < n_nodes - 1);//this would cause problems!
//
//                            I lambda_k = lambda[k];
//                            I old_pos  = node_to_index[k];
//                            I new_pos  = interval_ptr[lambda_k] + interval_count[lambda_k] - 1;
//
//                            node_to_index[index_to_node[old_pos]] = new_pos;
//                            node_to_index[index_to_node[new_pos]] = old_pos;
//                            std::swap(index_to_node[old_pos],index_to_node[new_pos]);
//
//                            //update intervals
//                            interval_count[lambda_k]   -= 1;
//                            interval_count[lambda_k+1] += 1;
//                            interval_ptr[lambda_k+1]    = new_pos;
//
//                            //increment lambda_k
//                            lambda[k]++;
//
//#ifdef DEBUG
//                            assert(interval_count[lambda_k]   >= 0);
//                            assert(interval_count[lambda_k+1] >  0);
//                            assert(interval_ptr[lambda[k]] <= node_to_index[k]);
//                            assert(node_to_index[k] < interval_ptr[lambda[k]] + interval_count[lambda[k]]);
//#endif
//                        }
//                    }
//                }
//            }
//
//            //For each j in S_i /\ U
//            for(I jj = Sp[i]; jj < Sp[i+1]; jj++){
//                I j = Sj[jj];
//                if(NodeSets[j] == U_NODE){            //decrement lambda for node j
//                    assert(lambda[j] > 0);//this would cause problems!
//
//                    //move j to the beginning of its current interval
//                    I lambda_j = lambda[j];
//                    I old_pos  = node_to_index[j];
//                    I new_pos  = interval_ptr[lambda_j];
//
//                    node_to_index[index_to_node[old_pos]] = new_pos;
//                    node_to_index[index_to_node[new_pos]] = old_pos;
//                    std::swap(index_to_node[old_pos],index_to_node[new_pos]);
//
//                    //update intervals
//                    interval_count[lambda_j]   -= 1;
//                    interval_count[lambda_j-1] += 1;
//                    interval_ptr[lambda_j]     += 1;
//                    interval_ptr[lambda_j-1]    = interval_ptr[lambda_j] - interval_count[lambda_j-1];
//
//                    //decrement lambda_j
//                    lambda[j]--;
//
//#ifdef DEBUG
//                    assert(interval_count[lambda_j]   >= 0);
//                    assert(interval_count[lambda_j-1] >  0);
//                    assert(interval_ptr[lambda[j]] <= node_to_index[j]);
//                    assert(node_to_index[j] < interval_ptr[lambda[j]] + interval_count[lambda[j]]);
//#endif
//                }
//            }
//        }
//    }
//
//
//
//
//#ifdef DEBUG
//    //make sure each f-node has at least one strong c-node neighbor
//    for(I i = 0; i < n_nodes; i++){
//        if(NodeSets[i] == F_NODE){
//            I row_start = Sp[i];
//            I row_end   = Sp[i+1];
//            bool has_c_neighbor = false;
//            for(I jj = row_start; jj < row_end; jj++){
//                if(NodeSets[Sj[jj]] == C_NODE){
//                    has_c_neighbor = true;
//                    break;
//                }
//            }
//            assert(has_c_neighbor);
//        }
//    }
//#endif
//
//    //Now construct interpolation operator
//    std::vector<T> d_k(n_nodes,0);
//    std::vector<bool> C_i(n_nodes,0);
//    Bp->push_back(0);
//    for(I i = 0; i < n_nodes; i++){
//        if(NodeSets[i] == C_NODE){
//            //interpolate directly
//            Bj->push_back(i);
//            Bx->push_back(1);
//            Bp->push_back(Bj->size());
//        } else {
//            //F_NODE
//
//            //Step 4
//            T d_i = 0; //denominator for this row
//            for(I jj = Ap[i]; jj < Ap[i+1]; jj++){ d_i += Ax[jj]; }
//            for(I jj = Sp[i]; jj < Sp[i+1]; jj++){ d_i -= Sx[jj]; }
//
//            //Create C_i, initialize d_k
//            for(I jj = Sp[i]; jj < Sp[i+1]; jj++){
//                I j = Sj[jj];
//                if(NodeSets[j] == C_NODE){
//                    C_i[j] = true;
//                    d_k[j] = Sx[jj];
//                }
//            }
//
//            bool Sj_intersects_Ci = true; //in the case that i has no F-neighbors
//            for(I jj = Sp[i]; jj < Sp[i+1]; jj++){ //for j in D^s_i
//                I    j = Sj[jj];
//                T   a_ij = Sx[jj];
//                T   a_jl = 0;
//
//                if(NodeSets[j] != F_NODE){continue;}
//
//                //Step 5
//                Sj_intersects_Ci = false;
//
//                //compute sum a_jl
//                for(I ll = Sp[j]; ll < Sp[j+1]; ll++){
//                    if(C_i[Sj[ll]]){
//                        Sj_intersects_Ci = true;
//                        a_jl += Sx[ll];
//                    }
//                }
//
//                if(!Sj_intersects_Ci){ break; }
//
//                for(I kk = Sp[j]; kk < Sp[j+1]; kk++){
//                    I   k = Sj[kk];
//                    T  a_jk = Sx[kk];
//                    if(C_i[k]){
//                        d_k[k] += a_ij*a_jk / a_jl;
//                    }
//                }
//            }
//
//            //Step 6
//            if(Sj_intersects_Ci){
//                for(I jj = Sp[i]; jj < Sp[i+1]; jj++){
//                    I j = Sj[jj];
//                    if(NodeSets[j] == C_NODE){
//                        Bj->push_back(j);
//                        Bx->push_back(-d_k[j]/d_i);
//                    }
//                }
//                Bp->push_back(Bj->size());
//            } else { //make i a C_NODE
//                NodeSets[i] = C_NODE;
//                Bj->push_back(i);
//                Bx->push_back(1);
//                Bp->push_back(Bj->size());
//            }
//
//
//            //Clear C_i,d_k
//            for(I jj = Sp[i]; jj < Sp[i+1]; jj++){
//                I j = Sj[jj];
//                C_i[j] = false;
//                d_k[j] = 0;
//            }
//
//        }
//
//    }
//
//    //for each c-node, determine its index in the coarser lvl
//    std::vector<I> cnode_index(n_nodes,-1);
//    I n_cnodes = 0;
//    for(I i = 0; i < n_nodes; i++){
//        if(NodeSets[i] == C_NODE){
//            cnode_index[i] = n_cnodes++;
//        }
//    }
//    //map old C indices to coarse indices
//    for(typename std::vector<I>::iterator iter = Bj->begin(); iter != Bj->end(); iter++){
//        *iter = cnode_index[*iter];
//    }
//}

#endif
