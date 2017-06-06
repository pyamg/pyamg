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


/*  Compute a strength of connection matrix using the classical strength
 *  of connection measure by Ruge and Stuben. Both the input and output
 *  matrices are stored in CSR format.  An off-diagonal nonzero entry
 *  A[i,j] is considered strong if:
 *
 *     | A[i,j] | >= theta * max( -A[i,k] )   where k != i      (_abs)
 *      -A[i,j]   >= theta * max( -A[i,k] )   where k != i      (_min)
 *
 * Otherwise, the connection is weak.
 *
 *  Parameters
 *      num_rows   - number of rows in A
 *      theta      - stength of connection tolerance
 *      A_rowptr[]       - CSR row pointer
 *      A_colinds[]       - CSR index array
 *      A_data[]       - CSR data array
 *      C_rowptr[]       - (output) CSR row pointer
 *      C_colinds[]       - (output) CSR index array
 *      C_data[]       - (output) CSR data array
 *
 *
 *  Returns:
 *      Nothing, S will be stored in C_rowptr, C_colinds, C_data
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
                                          const I A_rowptr[], const int A_rowptr_size,
                                          const I A_colinds[], const int A_colinds_size,
                                          const T A_data[], const int A_data_size,
                                                I C_rowptr[], const int C_rowptr_size,
                                                I C_colinds[], const int C_colinds_size,
                                                T C_data[], const int C_data_size)
{
    I nnz = 0;
    C_rowptr[0] = 0;

    // Compute SOC for each row
    for (I i = 0; i < n_row; i++) {
        F max_offdiagonal = std::numeric_limits<F>::min();

        const I row_start = A_rowptr[i];
        const I row_end   = A_rowptr[i+1];

        // Find maximum off-diagonal in this row
        for (I jj = row_start; jj < row_end; jj++){
            if(A_colinds[jj] != i){
                max_offdiagonal = std::max(max_offdiagonal,mynorm(A_data[jj]));
            }
        }

        // Set threshold for strong connections
        F threshold = theta*max_offdiagonal;
        for (I jj = row_start; jj < row_end; jj++){
            F norm_jj = mynorm(A_data[jj]);

            // Add entry if it exceeds the threshold
            if(norm_jj >= threshold){
                if(A_colinds[jj] != i){
                    C_colinds[nnz] = A_colinds[jj];
                    C_data[nnz] = A_data[jj];
                    nnz++;
                }
            }

            // Always add the diagonal
            if(A_colinds[jj] == i){
                C_colinds[nnz] = A_colinds[jj];
                C_data[nnz] = A_data[jj];
                nnz++;
            }
        }

        C_rowptr[i+1] = nnz;
    }
}

template<class I, class T>
void classical_strength_of_connection_min(const I n_row,
                                          const T theta,
                                          const I A_rowptr[], const int A_rowptr_size,
                                          const I A_colinds[], const int A_colinds_size,
                                          const T A_data[], const int A_data_size,
                                                I C_rowptr[], const int C_rowptr_size,
                                                I C_colinds[], const int C_colinds_size,
                                                T C_data[], const int C_data_size)
{
    I nnz = 0;
    C_rowptr[0] = 0;

    for (I i = 0; i < n_row; i++){
        T max_offdiagonal = 0.0;

        const I row_start = A_rowptr[i];
        const I row_end   = A_rowptr[i+1];

        // Find maximum off-diagonal negative entry in this row
        for (I jj = row_start; jj < row_end; jj++){
            if(A_colinds[jj] != i){
                max_offdiagonal = std::max(max_offdiagonal, -A_data[jj]);
            }
        }

        // Set threshold for strong connections
        T threshold = theta*max_offdiagonal;
        for (I jj = row_start; jj < row_end; jj++){
            T norm_jj = -A_data[jj];

            // Add entry if -A_ij exceeds the threshold
            if(norm_jj >= threshold){
                if(A_colinds[jj] != i){
                    C_colinds[nnz] = A_colinds[jj];
                    C_data[nnz] = A_data[jj];
                    nnz++;
                }
            }

            // Always add the diagonal
            if(A_colinds[jj] == i){
                C_colinds[nnz] = A_colinds[jj];
                C_data[nnz] = A_data[jj];
                nnz++;
            }
        }

        C_rowptr[i+1] = nnz;
    }
}


/*
 *  Compute the maximum in magnitude row value for a CSR matrix
 *
 *  Parameters
 *      n_row : const int
 *          Number of rows in A
 *      x : array<int>
 *          Empty array of size n
 *      A_rowptr : const array<int>
 *          Row pointer for A
 *      A_colinds : const array<int>
 *          Column indices for A
 *      A_data : const array<float>
 *          Data array for A
 *
 *  Returns:
 *      Nothing, x[i] will hold row i's maximum magnitude entry
 *
 */
template<class I, class T, class F>
void maximum_row_value(const I n_row,
                             T x[], const int  x_size,
                       const I A_rowptr[], const int A_rowptr_size,
                       const I A_colinds[], const int A_colinds_size,
                       const T A_data[], const int A_data_size)
{

    for (I i = 0; i < n_row; i++){
        F max_entry = std::numeric_limits<F>::min();

        const I row_start = A_rowptr[i];
        const I row_end   = A_rowptr[i+1];

        // Find this row's max entry
        for (I jj = row_start; jj < row_end; jj++){
            max_entry = std::max(max_entry, mynorm(A_data[jj]) );
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
 *   influence - array that influences splitting (values stored here are added to lambda for each point)
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
 *      C_rowptr[]       - CSR row pointer (strength matrix)
 *      C_colinds[]       - CSR index array
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
                          const I C_rowptr[], const int C_rowptr_size,
                          const I C_colinds[], const int C_colinds_size,
                          const I Tp[], const int Tp_size,
                          const I Tj[], const int Tj_size,
                                I splitting[], const int splitting_size,
                          const I colorflag)
{
  // initialize sizes
  int ncolors;
  I unassigned = n;
  I nD;
  int nnz = C_rowptr[n];

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
    //vertex_coloring_jones_plassmann(n, C_rowptr, C_colinds, &coloring[0],&weight[0]);
    //vertex_coloring_IDO(n, C_rowptr, C_colinds, &coloring[0]);
    vertex_coloring_mis(n, C_rowptr, C_rowptr_size, C_colinds, C_colinds_size, &coloring[0], n);
    ncolors = *std::max_element(coloring.begin(), coloring.end()) + 1;
    for (I i=0; i < n; i++){
      weight[i] = double(coloring[i])/double(ncolors);
    }
  }
  else {
    srand(2448422);
    for (I i=0; i < n; i++){
      weight[i] = double(rand())/RAND_MAX;
    }
  }

  for (I i=0; i < n; i++){
    for (I jj = C_rowptr[i]; jj < C_rowptr[i+1]; jj++){
      I j = C_colinds[jj];
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
    for (I i=0; i<n; i++){
      if(splitting[i]==U_NODE){
        D[i] = 1;
        // check row (S_i^T)
        for (I jj = C_rowptr[i]; jj < C_rowptr[i+1]; jj++){
          I j = C_colinds[jj];
          if(splitting[j]==U_NODE && weight[j]>weight[i]){
            D[i] = 0;
            break;
          }
        }
        // check col (S_i)
        if(D[i] == 1) {
          for (I jj = Tp[i]; jj < Tp[i+1]; jj++){
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
    for (I i = 0; i < nD; i++) {
      splitting[Dlist[i]] = C_NODE;
    }
    // end SELECT INDEPENDENT SET

    // UPDATE WEIGHTS
    // P5
    // nbrs that influence C points are not good C points
    for (I iD=0; iD < nD; iD++){
      I c = Dlist[iD];
      for (I jj = C_rowptr[c]; jj < C_rowptr[c+1]; jj++){
        I j = C_colinds[jj];
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
    for (I iD=0; iD < nD; iD++){
      I c = Dlist[iD];
      for (I jj = Tp[c]; jj < Tp[c+1]; jj++){
        I j = Tj[jj];
        if(splitting[j]==U_NODE)                 // j <---c
          c_dep_cache[j] = c;
      }

      for (I jj = Tp[c]; jj < Tp[c+1]; jj++) {
        I j = Tj[jj];
        for (I kk = C_rowptr[j]; kk < C_rowptr[j+1]; kk++) {
          I k = C_colinds[kk];
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

  for (I i = 0; i < C_rowptr[n]; i++){
    if(edgemark[i] == 0){
      edgemark[i] = -1;
    }
  }
  for (I i = 0; i < n; i++){
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
                                   const I C_rowptr[], const int C_rowptr_size,
                                   const I C_colinds[], const int C_colinds_size,
                                   const I splitting[], const int splitting_size,
                                         I P_rowptr[], const int P_rowptr_size)
{
    I nnz = 0;
    P_rowptr[0] = 0;
    for (I i = 0; i < n_nodes; i++){
        if( splitting[i] == C_NODE ){
            nnz++;
        }
        else {
            for (I jj = C_rowptr[i]; jj < C_rowptr[i+1]; jj++){
                if ( (splitting[C_colinds[jj]] == C_NODE) && (C_colinds[jj] != i) )
                    nnz++;
            }
        }
        P_rowptr[i+1] = nnz;
    }
}


template<class I, class T>
void rs_direct_interpolation_pass2(const I n_nodes,
                                   const I A_rowptr[], const int A_rowptr_size,
                                   const I A_colinds[], const int A_colinds_size,
                                   const T A_data[], const int A_data_size,
                                   const I C_rowptr[], const int C_rowptr_size,
                                   const I C_colinds[], const int C_colinds_size,
                                   const T C_data[], const int C_data_size,
                                   const I splitting[], const int splitting_size,
                                   const I P_rowptr[], const int P_rowptr_size,
                                         I P_colinds[], const int P_colinds_size,
                                         T P_data[], const int P_data_size)
{

    for (I i = 0; i < n_nodes; i++){
        if(splitting[i] == C_NODE){
            P_colinds[P_rowptr[i]] = i;
            P_data[P_rowptr[i]] = 1;
        } else {
            T sum_strong_pos = 0, sum_strong_neg = 0;
            for (I jj = C_rowptr[i]; jj < C_rowptr[i+1]; jj++){
                if ( (splitting[C_colinds[jj]] == C_NODE) && (C_colinds[jj] != i) ){
                    if (C_data[jj] < 0)
                        sum_strong_neg += C_data[jj];
                    else
                        sum_strong_pos += C_data[jj];
                }
            }

            T sum_all_pos = 0, sum_all_neg = 0;
            T diag = 0;
            for (I jj = A_rowptr[i]; jj < A_rowptr[i+1]; jj++){
                if (A_colinds[jj] == i){
                    diag += A_data[jj];
                } else {
                    if (A_data[jj] < 0)
                        sum_all_neg += A_data[jj];
                    else
                        sum_all_pos += A_data[jj];
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

            I nnz = P_rowptr[i];
            for (I jj = C_rowptr[i]; jj < C_rowptr[i+1]; jj++){
                if ( (splitting[C_colinds[jj]] == C_NODE) && (C_colinds[jj] != i) ){
                    P_colinds[nnz] = C_colinds[jj];
                    if (C_data[jj] < 0)
                        P_data[nnz] = neg_coeff * C_data[jj];
                    else
                        P_data[nnz] = pos_coeff * C_data[jj];
                    nnz++;
                }
            }
        }
    }


    std::vector<I> map(n_nodes);
    for (I i = 0, sum = 0; i < n_nodes; i++){
        map[i]  = sum;
        sum    += splitting[i];
    }
    for (I i = 0; i < P_rowptr[n_nodes]; i++){
        P_colinds[i] = map[P_colinds[i]];
    }
}


/* Helper function for compatible relaxation to perform steps 3.1d - 3.1f
 * in Falgout / Brannick (2010).  
 *
 * Parameters
 * ----------
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
 * Returns
 * -------  
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
               const T thetacs, 
               T cost[], const int cost_size )
{
    const T &Annz = A_colinds_size;
    const I &n = splitting_size;
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
    cost[0] += num_Fpts / Annz;

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
    cost[0] += num_Fpts / Annz;

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


/* First pass of classical AMG interpolation to build row pointer for P based
 * on SOC matrix and CF-splitting. Same method used for standard and modified
 * AMG interpolation below. 
 *
 * Parameters:
 * -----------
 *      n_nodes : const int
 *          Number of rows in A
 *      C_rowptr : const array<int>
 *          Row pointer for SOC matrix, C
 *      C_colinds : const array<int>
 *          Column indices for SOC matrix, C
 *      C_data : const array<float>
 *          Data array for SOC matrix, C
 *      splitting : const array<int>
 *          Boolean array with 1 denoting C-points and 0 F-points
 *      P_rowptr : array<int>
 *          empty array to store row pointer for matrix P
 *
 * Returns:
 * --------
 * Nothing, P_rowptr is modified in place. 
 *
 */
template<class I>
void rs_standard_interpolation_pass1(const I n_nodes,
                                     const I C_rowptr[], const int C_rowptr_size,
                                     const I C_colinds[], const int C_colinds_size,
                                     const I splitting[], const int splitting_size,
                                           I P_rowptr[], const int P_rowptr_size)
{
    I nnz = 0;
    P_rowptr[0] = 0;
    for (I i = 0; i < n_nodes; i++){
        if( splitting[i] == C_NODE ){
            nnz++;
        }
        else {
            for (I jj = C_rowptr[i]; jj < C_rowptr[i+1]; jj++){
                if ( (splitting[C_colinds[jj]] == C_NODE) && (C_colinds[jj] != i) )
                    nnz++;
            }
        }
        P_rowptr[i+1] = nnz;
    }
}


/* Produce the classical "standard" AMG interpolation operator. The first pass
 * uses the strength of connection matrix and C/F splitting to compute the row
 * pointer for the prolongator. The second pass fills in the nonzero entries of
 * the prolongator. Formula can be found in Eq. (3.7) in [1].
 *
 * Parameters:
 * -----------
 *      A_rowptr : const array<int>
 *          Row pointer for matrix A
 *      A_colinds : const array<int>
 *          Column indices for matrix A
 *      A_data : const array<float>
 *          Data array for matrix A
 *      C_rowptr : const array<int>
 *          Row pointer for SOC matrix, C
 *      C_colinds : const array<int>
 *          Column indices for SOC matrix, C
 *      C_data : const array<float>
 *          Data array for SOC matrix, C -- MUST HAVE VALUES OF A
 *      splitting : const array<int>
 *          Boolean array with 1 denoting C-points and 0 F-points
 *      P_rowptr : const array<int>
 *          Row pointer for matrix P
 *      P_colinds : array<int>
 *          Column indices for matrix P
 *      P_data : array<float>
 *          Data array for matrix P
 *
 * Returns:
 * --------
 * Nothing, P_colinds[] and P_data[] modified in place.
 *
 * References:
 * -----------
 * [0] J. W. Ruge and K. Stu ̈ben, Algebraic multigrid (AMG), in : S. F.
 *      McCormick, ed., Multigrid Methods, vol. 3 of Frontiers in Applied
 *      Mathematics (SIAM, Philadelphia, 1987) 73–130.
 *
 * [1] "Distance-Two Interpolation for Parallel Algebraic Multigrid,"
 *      H. De Sterck, R. Falgout, J. Nolting, U. M. Yang, (2007).
 */
template<class I, class T>
void rs_standard_interpolation_pass2(const I n_nodes,
                                     const I A_rowptr[], const int A_rowptr_size,
                                     const I A_colinds[], const int A_colinds_size,
                                     const T A_data[], const int A_data_size,
                                     const I C_rowptr[], const int C_rowptr_size,
                                     const I C_colinds[], const int C_colinds_size,
                                     const T C_data[], const int C_data_size,
                                     const I splitting[], const int splitting_size,
                                     const I P_rowptr[], const int P_rowptr_size,
                                           I P_colinds[], const int P_colinds_size,
                                           T P_data[], const int P_data_size)
{

    for (I i = 0; i < n_nodes; i++) {
        // If node i is a C-point, then set interpolation as injection
        if(splitting[i] == C_NODE) {
            P_colinds[P_rowptr[i]] = i;
            P_data[P_rowptr[i]] = 1;
        } 
        // Otherwise, use RS standard interpolation formula
        else {

            // Calculate denominator
            T denominator = 0;

            // Start by summing entire row of A
            for (I mm = A_rowptr[i]; mm < A_rowptr[i+1]; mm++) {
                denominator += A_data[mm];
            }

            // Then subtract off the strong connections so that you are left with 
            // denominator = a_ii + sum_{m in weak connections} a_im
            for (I mm = C_rowptr[i]; mm < C_rowptr[i+1]; mm++) {
                if ( C_colinds[mm] != i ) {
                    denominator -= C_data[mm]; // making sure to leave the diagonal entry in there
                }
            }

            // Set entries in P (interpolation weights w_ij from strongly connected C-points)
            I nnz = P_rowptr[i];
            for (I jj = C_rowptr[i]; jj < C_rowptr[i+1]; jj++) {

                if (splitting[C_colinds[jj]] == C_NODE) {

                    // Set temporary value for P_colinds as global index, j. Will be mapped to
                    // appropriate coarse-grid column index after all data is filled in. 
                    P_colinds[nnz] = C_colinds[jj];
                    I j = C_colinds[jj];

                    // Initialize numerator as a_ij
                    T numerator = C_data[jj];

                    // Sum over strongly connected fine points
                    for (I kk = C_rowptr[i]; kk < C_rowptr[i+1]; kk++) {
                        if ( (splitting[C_colinds[kk]] == F_NODE) && (C_colinds[kk] != i) ) {
                            
                            // Get column k and value a_ik
                            I k = C_colinds[kk];
                            T a_ik = C_data[kk];

                            // Get a_kj (have to search over k'th row in A for connection a_kj)
                            T a_kj = 0;
                            for (I search_ind = A_rowptr[k]; search_ind < A_rowptr[k+1]; search_ind++) {
                                if ( A_colinds[search_ind] == j ){
                                    a_kj = A_data[search_ind];
                                    break;
                                }
                            }

                            // If a_kj == 0, then we don't need to do any more work, otherwise
                            // proceed to account for node k's contribution
                            if (std::abs(a_kj) > 1e-16) {
                                
                                // Calculate sum for inner denominator (loop over strongly connected C-points)
                                T inner_denominator = 0;
                                for (I ll = C_rowptr[i]; ll < C_rowptr[i+1]; ll++) {
                                    if (splitting[C_colinds[ll]] == C_NODE) {
                                        
                                        // Get column l
                                        I l = C_colinds[ll];
                                        
                                        // Add connection a_kl if present in matrix (search over kth row in A for connection)
                                        for (I search_ind = A_rowptr[k]; search_ind < A_rowptr[k+1]; search_ind++) {
                                            if (A_colinds[search_ind] == l) {
                                                inner_denominator += A_data[search_ind];
                                            }
                                        }
                                    }
                                }

                                // Add a_ik * a_kj / inner_denominator to the numerator 
                                if (std::abs(inner_denominator) < 1e-16) {
                                    printf("Inner denominator was zero.\n");
                                }
                                numerator += a_ik * a_kj / inner_denominator;
                            }
                        }
                    }

                    // Set w_ij = -numerator/denominator
                    if (std::abs(denominator) < 1e-16) {
                        printf("Outer denominator was zero: diagonal plus sum of weak connections was zero.\n");
                    }
                    P_data[nnz] = -numerator / denominator;
                    nnz++;
                }
            }
        }
    }

    // Column indices were initially stored as global indices. Build map to switch
    // to C-point indices.
    std::vector<I> map(n_nodes);
    for (I i = 0, sum = 0; i < n_nodes; i++) {
        map[i]  = sum;
        sum    += splitting[i];
    }
    for (I i = 0; i < P_rowptr[n_nodes]; i++) {
        P_colinds[i] = map[P_colinds[i]];
    }
}


/* Remove strong F-to-F connections that do NOT have a common C-point from
 * the set of strong connections. Specifically, set the data value in CSR
 * format to 0. Removing zero entries afterwards will adjust row pointer
 * and column indices. 
 *
 * Parameters:
 * -----------
 *      n_nodes : const int
 *          Number of rows in A
 *      C_rowptr : const array<int>
 *          Row pointer for SOC matrix, C
 *      C_colinds : const array<int>
 *          Column indices for SOC matrix, C
 *      C_data : array<float>
 *          Data array for SOC matrix, C
 *      splitting : const array<int>
 *          Boolean array with 1 denoting C-points and 0 F-points
 *
 * Returns:
 * --------
 *      Nothing, C_data[] is set to zero to eliminate connections.
 */
template<class I, class T>
void remove_strong_FF_connections(const I n_nodes,
                                  const I C_rowptr[], const int C_rowptr_size,
                                  const I C_colinds[], const int C_colinds_size,
                                        T C_data[], const int C_data_size,
                                  const I splitting[], const int splitting_size)
{
    // For each F-point
    for (I row=0; row<n_nodes; row++) {
        if (splitting[row] == F_NODE) {

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
                    // Node j did not pass dependence test. That is, the two F-points
                    // do not have a common C neighbor, and we thus remove the strong
                    // connection.
                    else {
                        C_data[jj] = 0;
                    }
                }
            }
        }
    }
}


/* Produce a modified "standard" AMG interpolation operator for the case in which
 * two strongly connected F -points do NOT have a common C-neighbor. Formula can
 * be found in Eq. (3.8) of [1].
 *
 * Parameters:
 * -----------
 *      A_rowptr : const array<int>
 *          Row pointer for matrix A
 *      A_colinds : const array<int>
 *          Column indices for matrix A
 *      A_data : const array<float>
 *          Data array for matrix A
 *      C_rowptr : const array<int>
 *          Row pointer for SOC matrix, C
 *      C_colinds : const array<int>
 *          Column indices for SOC matrix, C
 *      C_data : const array<float>
 *          Data array for SOC matrix, C -- MUST HAVE VALUES OF A
 *      splitting : const array<int>
 *          Boolean array with 1 denoting C-points and 0 F-points
 *      P_rowptr : const array<int>
 *          Row pointer for matrix P
 *      P_colinds : array<int>
 *          Column indices for matrix P
 *      P_data : array<float>
 *          Data array for matrix P
 *
 * Notes:
 * ------
 * It is assumed that SOC matrix C is passed in WITHOUT any F-to-F connections
 * that do not share a common C-point neighbor. Any SOC matrix C can be set as
 * such by calling remove_strong_FF_connections().
 *
 * Returns:
 * --------
 * Nothing, P_colinds[] and P_data[] modified in place.
 *
 * References:
 * -----------
 * [0] V. E. Henson and U. M. Yang, BoomerAMG: a parallel algebraic multigrid
 *      solver and preconditioner, Applied Numerical Mathematics 41 (2002).
 *
 * [1] "Distance-Two Interpolation for Parallel Algebraic Multigrid,"
 *      H. De Sterck, R. Falgout, J. Nolting, U. M. Yang, (2007).
 */
template<class I, class T>
void mod_standard_interpolation_pass2(const I n_nodes,
                                      const I A_rowptr[], const int A_rowptr_size,
                                      const I A_colinds[], const int A_colinds_size,
                                      const T A_data[], const int A_data_size,
                                      const I C_rowptr[], const int C_rowptr_size,
                                      const I C_colinds[], const int C_colinds_size,
                                      const T C_data[], const int C_data_size,
                                      const I splitting[], const int splitting_size,
                                      const I P_rowptr[], const int P_rowptr_size,
                                            I P_colinds[], const int P_colinds_size,
                                            T P_data[], const int P_data_size)
{
    for (I i = 0; i < n_nodes; i++) {
        // If node i is a C-point, then set interpolation as injection
        if(splitting[i] == C_NODE) {
            P_colinds[P_rowptr[i]] = i;
            P_data[P_rowptr[i]] = 1;
        } 
        // Otherwise, use RS standard interpolation formula
        else {

            // Calculate denominator
            T denominator = 0;

            // Start by summing entire row of A
            for (I mm = A_rowptr[i]; mm < A_rowptr[i+1]; mm++) {
                denominator += A_data[mm];
            }

            // Then subtract off the strong connections so that you are left with 
            // denominator = a_ii + sum_{m in weak connections} a_im
            for (I mm = C_rowptr[i]; mm < C_rowptr[i+1]; mm++) {
                if ( C_colinds[mm] != i ) {
                    denominator -= C_data[mm]; // making sure to leave the diagonal entry in there
                }
            }

            // Set entries in P (interpolation weights w_ij from strongly connected C-points)
            I nnz = P_rowptr[i];
            for (I jj = C_rowptr[i]; jj < C_rowptr[i+1]; jj++) {

                if (splitting[C_colinds[jj]] == C_NODE) {

                    // Set temporary value for P_colinds as global index, j. Will be mapped to
                    // appropriate coarse-grid column index after all data is filled in. 
                    P_colinds[nnz] = C_colinds[jj];
                    I j = C_colinds[jj];

                    // Initialize numerator as a_ij
                    T numerator = C_data[jj];

                    // Sum over strongly connected fine points
                    for (I kk = C_rowptr[i]; kk < C_rowptr[i+1]; kk++) {
                        if ( (splitting[C_colinds[kk]] == F_NODE) && (C_colinds[kk] != i) ) {
                            
                            // Get column k and value a_ik
                            I k = C_colinds[kk];
                            T a_ik = C_data[kk];

                            // Get a_kj (have to search over k'th row in A for connection a_kj)
                            T a_kj = 0;
                            T a_kk = 0;
                            for (I search_ind = A_rowptr[k]; search_ind < A_rowptr[k+1]; search_ind++) {
                                if (A_colinds[search_ind] == j) {
                                    a_kj = A_data[search_ind];
                                }
                                else if (A_colinds[search_ind] == k) {
                                    a_kk = A_data[search_ind];
                                }
                            }

                            // If sign of a_kj matches sign of a_kk, ignore a_kj in sum
                            // (i.e. leave as a_kj = 0)
                            if (signof(a_kj) == signof(a_kk)) {
                                a_kj = 0;
                            }

                            // If a_kj == 0, then we don't need to do any more work, otherwise
                            // proceed to account for node k's contribution
                            if (std::abs(a_kj) > 1e-16) {
                                
                                // Calculate sum for inner denominator (loop over strongly connected C-points)
                                T inner_denominator = 0;
                                for (I ll = C_rowptr[i]; ll < C_rowptr[i+1]; ll++) {
                                    if (splitting[C_colinds[ll]] == C_NODE) {
                                        
                                        // Get column l
                                        I l = C_colinds[ll];
                                        
                                        // Add connection a_kl if present in matrix (search over kth row in A for connection)
                                        // Only add if sign of a_kl does not equal sign of a_kk
                                        for (I search_ind = A_rowptr[k]; search_ind < A_rowptr[k+1]; search_ind++) {
                                            if (A_colinds[search_ind] == l) {
                                                T a_kl = A_data[search_ind];
                                                if (signof(a_kl) != signof(a_kk)) {
                                                    inner_denominator += a_kl;
                                                }
                                                break;
                                            }
                                        }
                                    }
                                }

                                // Add a_ik * a_kj / inner_denominator to the numerator 
                                if (std::abs(inner_denominator) < 1e-16) {
                                    printf("Inner denominator was zero.\n");
                                }
                                numerator += a_ik * a_kj / inner_denominator;
                            }
                        }
                    }

                    // Set w_ij = -numerator/denominator
                    if (std::abs(denominator) < 1e-16) {
                        printf("Outer denominator was zero: diagonal plus sum of weak connections was zero.\n");
                    }
                    P_data[nnz] = -numerator / denominator;
                    nnz++;
                }
            }
        }
    }

    // Column indices were initially stored as global indices. Build map to switch
    // to C-point indices.
    std::vector<I> map(n_nodes);
    for (I i = 0, sum = 0; i < n_nodes; i++) {
        map[i]  = sum;
        sum    += splitting[i];
    }
    for (I i = 0; i < P_rowptr[n_nodes]; i++) {
        P_colinds[i] = map[P_colinds[i]];
    }
}


/* First pass of distance-two AMG interpolation to build row pointer for P based
 * on SOC matrix and CF-splitting.
 *
 * Parameters:
 * -----------
 *      n_nodes : const int
 *          Number of rows in A
 *      C_rowptr : const array<int>
 *          Row pointer for SOC matrix, C
 *      C_colinds : const array<int>
 *          Column indices for SOC matrix, C
 *      C_data : const array<float>
 *          Data array for SOC matrix, C
 *      splitting : const array<int>
 *          Boolean array with 1 denoting C-points and 0 F-points
 *      P_rowptr : array<int>
 *          empty array to store row pointer for matrix P
 *
 * Returns:
 * --------
 * Nothing, P_rowptr is modified in place. 
 */
template<class I>
void distance_two_amg_interpolation_pass1(const I n_nodes,
                                          const I C_rowptr[], const int C_rowptr_size,
                                          const I C_colinds[], const int C_colinds_size,
                                          const I splitting[], const int splitting_size,
                                                I P_rowptr[], const int P_rowptr_size)
{
    I nnz = 0;
    P_rowptr[0] = 0;
    for (I i = 0; i < n_nodes; i++){
        // +1 nnz for C-point rows
        if( splitting[i] == C_NODE ){
            nnz++;
        }
        // For F-point row i: interpolate from (i) all strongly connected C-points,
        // and (ii) for all F-points strongly connected to F-point i, say {Fj}, all
        // C-points strongly connected to Fj
        else {
            for (I jj = C_rowptr[i]; jj < C_rowptr[i+1]; jj++){
                I this_point = C_colinds[jj];
                // Strong C-point connections
                if (splitting[this_point] == C_NODE) {
                    nnz++;
                }
                // Strong F-point connections (excluding self)
                else if (this_point != i) {
                    for (I kk = C_rowptr[this_point]; kk < C_rowptr[this_point+1]; kk++){
                        // Strong C-point connections
                        if (splitting[C_colinds[kk]] == C_NODE) {
                            nnz++;
                        }
                    }
                }
            }
        }

        // Set value in row-pointer
        P_rowptr[i+1] = nnz;
    }
}


/* Compute distance-two standard AMG interpolation. Referred to as "extended+i
 * interpolation" in [0]. Uses neighbors within distance two for interpolation
 * weights. Formula can be found in Eqs. (4.10-4.11) in [0].
 *
 * Parameters:
 * -----------
 *      A_rowptr : const array<int>
 *          Row pointer for matrix A
 *      A_colinds : const array<int>
 *          Column indices for matrix A
 *      A_data : const array<float>
 *          Data array for matrix A
 *      C_rowptr : const array<int>
 *          Row pointer for SOC matrix, C
 *      C_colinds : const array<int>
 *          Column indices for SOC matrix, C
 *      C_data : const array<float>
 *          Data array for SOC matrix, C -- MUST HAVE VALUES OF A
 *      splitting : const array<int>
 *          Boolean array with 1 denoting C-points and 0 F-points
 *      P_rowptr : const array<int>
 *          Row pointer for matrix P
 *      P_colinds : array<int>
 *          Column indices for matrix P
 *      P_data : array<float>
 *          Data array for matrix P
 *
 * Returns:
 * --------
 * Nothing, P_colinds[] and P_data[] modified in place.
 *
 * References:
 * -----------
 * [0] "Distance-Two Interpolation for Parallel Algebraic Multigrid,"
 *      H. De Sterck, R. Falgout, J. Nolting, U. M. Yang, (2007).
 */
template<class I, class T>
void distance_two_amg_interpolation_pass2(const I n_nodes,
                                          const I A_rowptr[], const int A_rowptr_size,
                                          const I A_colinds[], const int A_colinds_size,
                                          const T A_data[], const int A_data_size,
                                          const I C_rowptr[], const int C_rowptr_size,
                                          const I C_colinds[], const int C_colinds_size,
                                          const T C_data[], const int C_data_size,
                                          const I splitting[], const int splitting_size,
                                          const I P_rowptr[], const int P_rowptr_size,
                                                I P_colinds[], const int P_colinds_size,
                                                T P_data[], const int P_data_size)
{

    for (I i = 0; i < n_nodes; i++) {
        // If node i is a C-point, then set interpolation as injection
        if(splitting[i] == C_NODE) {
            P_colinds[P_rowptr[i]] = i;
            P_data[P_rowptr[i]] = 1;
        } 
        // Otherwise, use extended+i distance-two AMG interpolation formula
        // (see Eqs. 4.10-4.11 in [0])
        else {

            // -------------------------------------------------------------------------------- //
            // -------------------------------------------------------------------------------- //
            // Calculate outer denominator
            T denominator = 0;

            // Start by summing entire row of A
            for (I mm = A_rowptr[i]; mm < A_rowptr[i+1]; mm++) {
                denominator += A_data[mm];
            }

            // Then subtract off the strong connections so that you are left with 
            // denominator = a_ii + sum_{m in weak connections} a_im
            for (I mm = C_rowptr[i]; mm < C_rowptr[i+1]; mm++) {
                if ( C_colinds[mm] != i ) {
                    denominator -= C_data[mm]; // making sure to leave the diagonal entry in there
                }
            }

            // Sum over strongly connected fine points for outer denominator
            for (I kk = C_rowptr[i]; kk < C_rowptr[i+1]; kk++) {
                if ( (splitting[C_colinds[kk]] == F_NODE) && (C_colinds[kk] != i) ) {

                    // Get column k and value a_ik
                    I k = C_colinds[kk];
                    T a_ik = C_data[kk];

                    // Get a_ki (have to search over k'th row in A for connection a_ki)
                    T a_ki = 0;
                    T a_kk = 0;
                    for (I search_ind = A_rowptr[k]; search_ind < A_rowptr[k+1]; search_ind++) {
                        if (A_colinds[search_ind] == i) {
                            a_ki = A_data[search_ind];
                        }
                        else if (A_colinds[search_ind] == k) {
                            a_kk = A_data[search_ind];
                        }
                    }

                    // If sign of a_ki matches sign of a_kk, ignore a_ki in sum
                    // (i.e. leave as a_ki = 0)
                    if (signof(a_ki) == signof(a_kk)) {
                        a_ki = 0;
                    }

                    // If a_ki == 0, then we don't need to do any more work, otherwise
                    // proceed to account for node k's contribution
                    if (std::abs(a_ki) > 1e-16) {
                        // Calculate sum for inner denominator (loop over strongly connected C-points
                        // and distance-two strongly connected C-points).
                        T inner_denominator = 0;
                        for (I ll = C_rowptr[i]; ll < C_rowptr[i+1]; ll++) {
                            I this_point = C_colinds[ll];

                            // Strong C-connections
                            if (splitting[this_point] == C_NODE) {

                                // Add connection a_kl if present in matrix (search over kth row in A for connection)
                                // Only add if sign of a_kl does not equal sign of a_kk
                                for (I search_ind = A_rowptr[k]; search_ind < A_rowptr[k+1]; search_ind++) {
                                    if (A_colinds[search_ind] == this_point) {
                                        T a_kl = A_data[search_ind];
                                        if (signof(a_kl) != signof(a_kk)) {
                                            inner_denominator += a_kl;
                                        }
                                        break;
                                    }
                                }
                            }
                            // Strong F-connections (excluding self)
                            else if (this_point != i) {
                                for (I ff = C_rowptr[this_point]; ff < C_rowptr[this_point+1]; ff++) {
                                    I d2_point = C_colinds[ff];

                                    // Strong C-connections to strong F-connections (distance two C connections)
                                    if (splitting[d2_point] == C_NODE) {

                                        // Add connection a_kl if present in matrix (search over kth row in A
                                        // for connection). Only add if sign of a_kl does not equal sign of a_kk
                                        for (I search_ind = A_rowptr[k]; search_ind < A_rowptr[k+1]; search_ind++) {
                                            if (A_colinds[search_ind] == d2_point) {
                                                T a_kl = A_data[search_ind];
                                                if (signof(a_kl) != signof(a_kk)) {
                                                    inner_denominator += a_kl;
                                                }
                                                break;
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        // Add a_ki to inner denominator
                        inner_denominator += a_ki;

                        // Add a_ik * a_ki / inner_denominator to the denominator 
                        if (std::abs(inner_denominator) < 1e-16) {
                            std::cout << "Inner denominator of outer denominator is zero.\n";
                        }
                        denominator += a_ik * a_ki / inner_denominator;
                    }
                }
            }

            // -------------------------------------------------------------------------------- //
            // -------------------------------------------------------------------------------- //
            // Set entries in P (interpolation weights w_ij from strongly connected C-points)
            I nnz = P_rowptr[i];
            for (I jj = C_rowptr[i]; jj < C_rowptr[i+1]; jj++) {
                I neighbor = C_colinds[jj];

                // ---------------------------------------------------------------------------- //
                // Build interpolation for strong distance-one C-points from F-point i
                if (splitting[neighbor] == C_NODE) {

                    // Set temporary value for P_colinds as global index. Will be mapped to
                    // appropriate coarse-grid column index after all data is filled in. 
                    P_colinds[nnz] = neighbor;

                    // Initialize numerator as a_ij
                    T numerator = C_data[jj];

                    // Sum over strongly connected F points
                    for (I kk = C_rowptr[i]; kk < C_rowptr[i+1]; kk++) {
                        if ( (splitting[C_colinds[kk]] == F_NODE) && (C_colinds[kk] != i) ) {
                            
                            // Get column k and value a_ik
                            I k = C_colinds[kk];
                            T a_ik = C_data[kk];

                            // Get a_kj (have to search over k'th row in A for connection a_kj)
                            T a_kj = 0;
                            T a_kk = 0;
                            for (I search_ind = A_rowptr[k]; search_ind < A_rowptr[k+1]; search_ind++) {
                                if (A_colinds[search_ind] == neighbor) {
                                    a_kj = A_data[search_ind];
                                }
                                else if (A_colinds[search_ind] == k) {
                                    a_kk = A_data[search_ind];
                                }
                            }

                            // If sign of a_kj matches sign of a_kk, ignore a_kj in sum
                            // (i.e. leave as a_kj = 0)
                            if (signof(a_kj) == signof(a_kk)) {
                                a_kj = 0;
                            }

                            // If a_kj == 0, then we don't need to do any more work, otherwise
                            // proceed to account for node k's contribution
                            if (std::abs(a_kj) > 1e-16) {
                                
                                // Calculate sum for inner denominator (loop over strongly connected C-points
                                // and distance-two strongly connected C-points).
                                T inner_denominator = 0;
                                for (I ll = C_rowptr[i]; ll < C_rowptr[i+1]; ll++) {
                                    I this_point = C_colinds[ll];

                                    // Strong C-connections
                                    if (splitting[this_point] == C_NODE) {
                                        
                                        // Add connection a_kl if present in matrix (search over kth row in A for connection)
                                        // Only add if sign of a_kl does not equal sign of a_kk
                                        for (I search_ind = A_rowptr[k]; search_ind < A_rowptr[k+1]; search_ind++) {
                                            if (A_colinds[search_ind] == this_point) {
                                                T a_kl = A_data[search_ind];
                                                if (signof(a_kl) != signof(a_kk)) {
                                                    inner_denominator += a_kl;
                                                }
                                                break;
                                            }
                                        }
                                    }
                                    // Strong F-connections (excluding self)
                                    else if (this_point != i) {
                                        for (I ff = C_rowptr[this_point]; ff < C_rowptr[this_point+1]; ff++) {
                                            I d2_point = C_colinds[ff];

                                            // Strong C-connections to strong F-connections (distance two C connections)
                                            if (splitting[d2_point] == C_NODE) {

                                                // Add connection a_kl if present in matrix (search over kth row in A
                                                // for connection). Only add if sign of a_kl does not equal sign of a_kk
                                                for (I search_ind = A_rowptr[k]; search_ind < A_rowptr[k+1]; search_ind++) {
                                                    if (A_colinds[search_ind] == d2_point) {
                                                        T a_kl = A_data[search_ind];
                                                        if (signof(a_kl) != signof(a_kk)) {
                                                            inner_denominator += a_kl;
                                                        }
                                                        break;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }

                                // Add a_ki to inner denominator
                                T a_ki = 0;
                                for (I search_ind = A_rowptr[k]; search_ind < A_rowptr[k+1]; search_ind++) {
                                    if (A_colinds[search_ind] == i) {
                                        a_ki = A_data[search_ind];
                                        break;
                                    }
                                }
                                if (signof(a_ki) == signof(a_kk)) {
                                    a_ki = 0;
                                }
                                inner_denominator += a_ki;

                                // Add a_ik * a_kj / inner_denominator to the numerator 
                                if (std::abs(inner_denominator) < 1e-16) {
                                    printf("Inner denominator was zero.\n");
                                }
                                numerator += a_ik * a_kj / inner_denominator;
                            }
                        }
                    }

                    // Set w_ij = -numerator/denominator
                    if (std::abs(denominator) < 1e-16) {
                        printf("Outer denominator was zero.\n");
                    }
                    P_data[nnz] = -numerator / denominator;
                    nnz++;
                }
                // ---------------------------------------------------------------------------- //
                // Build interpolation for strong distance-two C-points from F-point i
                else if (neighbor != i) {
                    for (I dd = C_rowptr[neighbor]; dd < C_rowptr[neighbor+1]; dd++){
                        I neighbor2 = C_colinds[dd];

                        // Strong distance-two C-point connections
                        if (splitting[neighbor2] == C_NODE) {

                            // Set temporary value for P_colinds as global index. Will be mapped to
                            // appropriate coarse-grid column index after all data is filled in. 
                            P_colinds[nnz] = neighbor2;

                            // Initialize numerator as a_ij (j is neighbor2, need to search in matrix for value)
                            T a_ij = 0;
                            for (I search_ind = A_rowptr[i]; search_ind < A_rowptr[i+1]; search_ind++) {
                                if (A_colinds[search_ind] == neighbor2) {
                                    a_ij = A_data[search_ind];
                                    break;
                                }
                            }
                            T numerator = a_ij;

                            // Sum over strongly connected F points
                            for (I kk = C_rowptr[i]; kk < C_rowptr[i+1]; kk++) {
                                if ( (splitting[C_colinds[kk]] == F_NODE) && (C_colinds[kk] != i) ) {
                                    
                                    // Get column k and value a_ik
                                    I k = C_colinds[kk];
                                    T a_ik = C_data[kk];

                                    // Get a_kj (have to search over k'th row in A for connection a_kj)
                                    T a_kj = 0;
                                    T a_kk = 0;
                                    for (I search_ind = A_rowptr[k]; search_ind < A_rowptr[k+1]; search_ind++) {
                                        if (A_colinds[search_ind] == neighbor2) {
                                            a_kj = A_data[search_ind];
                                        }
                                        else if (A_colinds[search_ind] == k) {
                                            a_kk = A_data[search_ind];
                                        }
                                    }

                                    // If sign of a_kj matches sign of a_kk, ignore a_kj in sum
                                    // (i.e. leave as a_kj = 0)
                                    if (signof(a_kj) == signof(a_kk)) {
                                        a_kj = 0;
                                    }

                                    // If a_kj == 0, then we don't need to do any more work, otherwise
                                    // proceed to account for node k's contribution
                                    if (std::abs(a_kj) > 1e-16) {
                                        
                                        // Calculate sum for inner denominator (loop over strongly connected C-points
                                        // and distance-two strongly connected C-points).
                                        T inner_denominator = 0;
                                        for (I ll = C_rowptr[i]; ll < C_rowptr[i+1]; ll++) {
                                            I this_point = C_colinds[ll];

                                            // Strong C-connections
                                            if (splitting[this_point] == C_NODE) {
                                                
                                                // Add connection a_kl if present in matrix (search over kth row in A for connection)
                                                // Only add if sign of a_kl does not equal sign of a_kk
                                                for (I search_ind = A_rowptr[k]; search_ind < A_rowptr[k+1]; search_ind++) {
                                                    if (A_colinds[search_ind] == this_point) {
                                                        T a_kl = A_data[search_ind];
                                                        if (signof(a_kl) != signof(a_kk)) {
                                                            inner_denominator += a_kl;
                                                        }
                                                        break;
                                                    }
                                                }
                                            }
                                            // Strong F-connections (excluding self)
                                            else if (this_point != i) {
                                                for (I ff = C_rowptr[this_point]; ff < C_rowptr[this_point+1]; ff++) {
                                                    I d2_point = C_colinds[ff];

                                                    // Strong C-connections to strong F-connections (distance two C connections)
                                                    if (splitting[d2_point] == C_NODE) {

                                                        // Add connection a_kl if present in matrix (search over kth row in A
                                                        // for connection). Only add if sign of a_kl does not equal sign of a_kk
                                                        for (I search_ind = A_rowptr[k]; search_ind < A_rowptr[k+1]; search_ind++) {
                                                            if (A_colinds[search_ind] == d2_point) {
                                                                T a_kl = A_data[search_ind];
                                                                if (signof(a_kl) != signof(a_kk)) {
                                                                    inner_denominator += a_kl;
                                                                }
                                                                break;
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }

                                        // Add a_ki to inner denominator
                                        T a_ki = 0;
                                        for (I search_ind = A_rowptr[k]; search_ind < A_rowptr[k+1]; search_ind++) {
                                            if (A_colinds[search_ind] == i) {
                                                a_ki = A_data[search_ind];
                                                break;
                                            }
                                        }
                                        if (signof(a_ki) == signof(a_kk)) {
                                            a_ki = 0;
                                        }
                                        inner_denominator += a_ki;

                                        // Add a_ik * a_kj / inner_denominator to the numerator 
                                        if (std::abs(inner_denominator) < 1e-16) {
                                            printf("Inner denominator was zero.\n");
                                        }
                                        numerator += a_ik * a_kj / inner_denominator;
                                    }
                                }
                            }

                            // Set w_ij = -numerator/denominator
                            if (std::abs(denominator) < 1e-16) {
                                printf("Outer denominator was zero.\n");
                            }
                            P_data[nnz] = -numerator / denominator;
                            nnz++;
                        }
                    }
                }
            }
        }
    }

    // ---------------------------------------------------------------------------------------- //
    // ---------------------------------------------------------------------------------------- //
    // Column indices were initially stored as global indices. Build map to switch
    // to C-point indices.
    std::vector<I> map(n_nodes);
    for (I i = 0, sum = 0; i < n_nodes; i++) {
        map[i]  = sum;
        sum    += splitting[i];
    }
    for (I i = 0; i < P_rowptr[n_nodes]; i++) {
        P_colinds[i] = map[P_colinds[i]];
    }
}








#endif
