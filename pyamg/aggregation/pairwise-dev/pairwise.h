#ifndef PAIRWISE_H
#define PAIRWISE_H

#include <iostream>
#include <vector>
#include <iterator>
#include <algorithm>

#include <assert.h>
#include <cmath>

#include "linalg.h"


/* Function determining which of two elements in the A.data structure is
 * larger. Allows for easy changing between, e.g., absolute value, hard   
 * minimum, etc.   
 *                    
 * Input:
 * ------
 * ind0 : const {int}
 *      Index for element in A.data
 * ind1 : const {int}
 *      Index for element in A.data
 * data : const {float array}  
 *      Data elements for A in sparse format
 *
 * Returns:
 * --------  
 * bool on if data[ind0] > data[ind1] for the measure of choice. For now this is
 * an absolute maximum.
 */
template<class I, class T>
bool is_larger(const I &ind0,
               const I &ind1,
               const T A_data[])
{
    if (std::abs(A_data[ind0]) >= std::abs(A_data[ind1]) ) {
        return true;
    }
    else {
        return false;
    }
}


/* Function that finds the maximum edge connected to a given node and adds
 * this pair to the matching, for a 'maximum' defined by is_larger(). If a 
 * pair is found, each node is marked as aggregated, and the new node index
 * returned. If there are no unaggregated connections to the input node, -1
 * is returned. 
 *
 * Input:
 * ------
 * A_rowptr : const {int array}
 *      Row pointer for sparse array in CSR format.
 * A_colinds : const {int array}
 *      Column indices for sparse array in CSR format.
 * A_data : const {float array}
 *      Data values for sparse array in CSR format.
 * M : {int array}
 *      Approximate matching stored in 1d array, with indices for a pair
 *      as consecutive elements. If new pair is formed, it is added to M[].
 * W : {float}
 *      Additive weight of current matching. If new pair is formed,
 *      corresponding weight is added to W.
 * row : const {int}
 *      Index of base node to form pair for matching. 
 *
 * Returns:
 * --------  
 * Integer index of node connected to input node as a pair in matching. If
 * no unaggregated nodes are connected to input node, -1 is returned. 
 */
template<class I, class T>
I add_edge(const I A_rowptr[],
           const I A_colinds[],
           const T A_data[],
           std::vector<I> &M,
           T &W,
           const I &row,
           T cost[] )
{
    I data_ind0 = A_rowptr[row];
    I data_ind1 = A_rowptr[row+1];
    I new_node = -1;
    I new_ind = data_ind0;

    // Find maximum edge attached to node 'row'
    for (I i=data_ind0; i<data_ind1; i++) {
        I temp_node = A_colinds[i];
        // Check for self-loops and make sure node has not been aggregated 
        if ( (temp_node != row) && (M[temp_node] == -1) ) {
            if (is_larger(i, new_ind, A_data)) {
                new_node = temp_node;
                new_ind = i;
            }
            cost[0] += 1.0;
        }
        cost[0] += 1.0;
    }

    // Add edge to matching and weight to total edge weight.
    // Mark each node in pair as aggregated. 
    if (new_node != -1) {
        W += std::abs(A_data[new_ind]);
        M[row] = new_node;
        M[new_node] = row;
        cost[0] += 1.0;
    }

    // Return node index in new edge
    return new_node;
}


/* Function to approximate a graph matching, and use the matching for 
 * a pairwise aggregation of matrix A. This version takes in a target
 * near null space vector and forms the sparse arrays for a tentative
 * prolongator, normalized as in Panayot and Pasqua (2013). Matching
 * done via Drake's 2003 1/2-matching algorithm.  
 *
 * Input:
 * ------
 * A_rowptr : const {int array}
 *      Row pointer for sparse array in CSR format.
 * A_colinds : const {int array}
 *      Column indices for sparse array in CSR format.
 * A_data : const {float array}
 *      Data values for sparse array in CSR format.
 * Agg_rowptr : {int array}
 *      Empty length(n+1) row pointer for sparse array in CSR format.
 * Agg_colinds : {int array}
 *      Empty length(n) column indices for sparse array in CSR format.
 * Agg_data : {float array}
 *      Empty length(n) data values for sparse array in CSR format.
 * Agg_shape : {int array, size 2} 
 *      Shape array for sparse matrix constructed in function.
 * B : const {float array}
 *      Target near null space components provided by user.
 * n : const {int} 
 *      Problem size
 *
 * Returns
 * -------
 * Nothing, sparse prolongation / aggregation matrix modified in place.
 *
 */
template<class I, class T>
void drake_matching_data(const I A_rowptr[],
                         const I A_colinds[], 
                         const T A_data[],
                         I Agg_rowptr[],
                         I Agg_colinds[],
                         T Agg_data[], 
                         I Agg_shape[],
                         const T B[],
                         const I &n,
                         T cost[] )
{
        
    // Store M1[:], M2[:] = -1 to start. When nodes are aggregated, 
    // say x and y, set M1[x] = y and M1[y] = x. 
    std::vector<I> M1(n,-1);     
    std::vector<I> M2(n,-1);

    // Empty initial weights.
    T W1 = 0;
    T W2 = 0;

    // Form two matchings, M1, M2, starting from last node in DOFs. 
    for (I row=(n-1); row>=0; row--) {
        I x = row;
        while (true) {       
            // Get new edge in matching, M1. Break loop if node x has no
            // edges to unaggregated nodes.
            if (M1[x] != -1) {
                break;
            }    
            I y = add_edge(A_rowptr, A_colinds, A_data, M1, W1, x, cost);
            if (y == -1) {
                break;
            }

            // Get new edge in matching, M2. Break loop if node y has no
            // edges to unaggregated nodes.
            if (M2[y] != -1) {
                break;
            }
            x = add_edge(A_rowptr, A_colinds, A_data, M2, W2, y, cost);
            if (x == -1) {
                break;
            }
        }
    }

    int *M = NULL; 
    if (std::abs(W1) >= std::abs(W2)) {
        M = &M1[0];
    }
    else {
        M = &M2[0];
    }

    // Form sparse structure of aggregation matrix 
    I Nc = 0;
    T max_single = 0.0;
    Agg_rowptr[0] = 0;
    std::vector<I> singletons;
    for (I i=0; i<n; i++) {

        // Set row pointer value for next row
        Agg_rowptr[i+1] = i+1;

        // Node has not been aggregated --> singleton
        if (M[i] == -1) {

            // Add singleton to sparse structure
            Agg_colinds[i] = Nc;
            Agg_data[i] = B[i];

            // Find largest singleton to normalize all singletons by
            singletons.push_back(i);
            if (std::abs(B[i]) > max_single) {
                max_single = std::abs(B[i]);
            }
            cost[0] += 1.0;

            // Mark node as stored (-2), increase coarse grid count
            M[i] = -2;
            Nc += 1;
        }
        // Node has been aggregated, mark pair in aggregation matrix
        else if (M[i] > -1) {

            // Reference to each node in pair for ease of notation
            const I &p1 = i;
            const I &p2 = M[i];

            // Set rows p1, p2 to have column Nc
            Agg_colinds[p1] = Nc;
            Agg_colinds[p2] = Nc;

            // Normalize bad guy over aggregate, store in data vector
            T norm_b = std::sqrt( B[p1]*B[p1] + B[p2]*B[p2] );
            Agg_data[p1] = B[p1] / norm_b;
            Agg_data[p2] = B[p2] / norm_b;
            cost[0] += 4.0;

            // Mark both nodes as stored (-2), and increase coarse grid count
            // Order is important, must modify M[p2] before M[p1], because 
            // p2 is a reference to M[p1].
            M[p2] = -2;
            M[p1] = -2;
            Nc += 1;
        }
        // Node has already been added to sparse aggregation matrix
        else { 
            continue;
        }
    }

    // Normalize singleton data value, s_k <-- s_k / max_k |s_k|.
    if (max_single > 0) {
        for (auto it=singletons.begin(); it!=singletons.end(); it++) {
            Agg_data[*it] /= max_single;
            cost[0] += 1.0;
        }
    }

    // Save shape of aggregation matrix
    Agg_shape[0] = n;
    Agg_shape[1] = Nc;
    M = NULL;
}


/* Function to approximate a graph matching, and use the matching for 
 * a pairwise aggregation of matrix A. This version only constructs the
 * row pointer and column indices for a CSR tentative prolongator. 
 * Matching done via Drake's 2003 1/2-matching algorithm.  
 *
 * Input:
 * ------
 * A_rowptr : const {int array}
 *      Row pointer for sparse array in CSR format.
 * A_colinds : const {int array}
 *      Column indices for sparse array in CSR format.
 * A_data : const {float array}
 *      Data values for sparse array in CSR format.
 * Agg_rowptr : {int array}
 *      Empty length(n+1) row pointer for sparse array in CSR format.
 * Agg_colinds : {int array}
 *      Empty length(n) column indices for sparse array in CSR format.
 * Agg_shape : {int array, size 2} 
 *      Shape array for sparse matrix constructed in function.
 * n : const {int} 
 *      Problem size
 *
 * Returns
 * -------
 * Nothing, sparse prolongation / aggregation matrix modified in place.
 *
 */
template<class I, class T>
void drake_matching_nodata(const I A_rowptr[],
                           const I A_colinds[], 
                           const T A_data[],
                           I Agg_rowptr[],
                           I Agg_colinds[],
                           I Agg_shape[],
                           const I &n,
                           T cost[] )
{
        
    // Plan - store M1, M2 as all -a to start, when nodes are aggregated, 
    // say x and y, set M1[x] = y and M1[y] = x. 
    std::vector<I> M1(n,-1);     
    std::vector<I> M2(n,-1);

    // Empty initial weights.
    T W1 = 0;
    T W2 = 0;

    // Form two matchings, M1, M2, starting from last node in DOFs. 
    for (I row=(n-1); row>=0; row--) {
        I x = row;
        while (true) {       
            // Get new edge in matching, M1. Break loop if node x has no
            // edges to unaggregated nodes.
            if (M1[x] != -1) {
                break;
            }    
            I y = add_edge(A_rowptr, A_colinds, A_data, M1, W1, x, cost);
            if (y == -1) {
                break;
            }

            // Get new edge in matching, M2. Break loop if node y has no
            // edges to unaggregated nodes.
            if (M2[y] != -1) {
                break;
            }
            x = add_edge(A_rowptr, A_colinds, A_data, M2, W2, y, cost);
            if (x == -1) {
                break;
            }
        }
    }

    int *M = NULL; 
    if (std::abs(W1) >= std::abs(W2)) {
        M = &M1[0];
    }
    else {
        M = &M2[0];
    }

    // Form sparse structure of aggregation matrix 
    I Nc = 0;
    Agg_rowptr[0] = 0;
    for (I i=0; i<n; i++) {

        cost[0] += 1.0; // No real FLOPs, just +1 / iteration?

        // Set row pointer value for next row
        Agg_rowptr[i+1] = i+1;

        // Node has not been aggregated --> singleton
        if (M[i] == -1) {

            // Add singleton to sparse structure
            Agg_colinds[i] = Nc;

            // Mark node as stored (-2), increase coarse grid count
            M[i] = -2;
            Nc += 1;
        }
        // Node has been aggregated, mark pair in aggregation matrix
        else if (M[i] > -1) {

            // Reference to each node in pair for ease of notation
            const I &p2 = M[i];
            const I &p1 = i;

            // Set rows p1, p2 to have column Nc
            Agg_colinds[p1] = Nc;
            Agg_colinds[p2] = Nc;

            // Mark both nodes as stored (-2), and increase coarse grid count
            // Order is important, must modify M[p2] before M[p1], because 
            // p2 is a reference to M[p1].
            M[p2] = -2;
            M[p1] = -2;
            Nc += 1;
        }
        // Node has already been added to sparse aggregation matrix
        else { 
            continue;
        }
    }

    // Save shape of aggregation matrix
    Agg_shape[0] = n;
    Agg_shape[1] = Nc;
    M = NULL;
}


template<class I, class T>
void drake_matching(const I A_rowptr[], const int A_rowptr_size,
                    const I A_colinds[], const int A_colinds_size,
                    const T A_data[], const int A_data_size,
                    const T B[], const int B_size,
                    I Agg_rowptr[], const int Agg_rowptr_size,
                    I Agg_colinds[], const int Agg_colinds_size,
                    T Agg_data[], const int Agg_data_size,
                    I Agg_shape[], const int Agg_shape_size,
                    T cost[], const int cost_size )
{
    I n = A_rowptr_size-1;
    drake_matching_data(A_rowptr, A_colinds, A_data,
                        Agg_rowptr, Agg_colinds, Agg_data,
                        Agg_shape, B, n, cost);
}


template<class I, class T>
void drake_matching(const I A_rowptr[], const int A_rowptr_size,
                    const I A_colinds[], const int A_colinds_size,
                    const T A_data[], const int A_data_size,
                    I Agg_rowptr[], const int Agg_rowptr_size,
                    I Agg_colinds[], const int Agg_colinds_size,
                    I Agg_shape[], const int Agg_shape_size,
                    T cost[], const int cost_size )
{
    I n = A_rowptr_size-1;
    drake_matching_nodata(A_rowptr, A_colinds, A_data,
                          Agg_rowptr, Agg_colinds, Agg_shape,
                          n, cost);
}

/* Function to filter matrix A using the hard minimum approach in
 * Notay (2010). Filters matrix and stores result in sparse CSR 
 * format in input reference vectors. 
 * 
 * Input:
 * ------
 * A_rowptr : const {int array}
 *      Row pointer for sparse array in CSR format.
 * A_colinds : const {int array}
 *      Column indices for sparse array in CSR format.
 * A_data : const {float array}
 *      Data values for sparse array in CSR format.
 * rowptr : {vector<int>}
 *      Empty vector to store row pointer for filtered CSR matrix.
 * colinds : {vector<int>}
 *      Empty vector to store column indices for filtered CSR matrix.
 * data : {vector<float>}
 *      Empty vector to store data for filtered CSR matrix.
 * beta : const {float}
 *      Threshold for filtering out 'strong' connections  
 *          a_ij < -beta * max_{a_ij < 0} |a_ij|
 * n : const {int} 
 *      Problem size
 *
 * Returns
 * -------
 * Nothing, sparse structure for filtered matrix modified in place.
 *
 */
template<class I, class T>
void notay_filter(const I A_rowptr[],
                  const I A_colinds[],
                  const T A_data[],
                  std::vector<I> &rowptr,
                  std::vector<I> &colinds,
                  std::vector<T> &data,
                  const T &beta,
                  const I &n)
{
    rowptr.push_back(0);
    for (I i=0; i<n; i++) {
        // Filtering threshold, -beta * max_{a_ij < 0} |a_ij|.
        T row_thresh = 0;
        for (I j=A_rowptr[i]; j<A_rowptr[i+1]; j++) {
            if (A_data[j] < row_thresh) {
                row_thresh = A_data[j];
            }
        }
        row_thresh *= beta;

        // Construct sparse row of filtered matrix
        I row_size = 0;
        for (I j=A_rowptr[i]; j<A_rowptr[i+1]; j++) {
            if (A_data[j] < row_thresh) {
                colinds.push_back(A_colinds[j]);
                data.push_back(A_data[j]);
                row_size += 1;
            }
        }
        rowptr.push_back(rowptr[i]+row_size);
    }
}


/* Function to approximate to perform pairwise aggregation on matrix A,
 * as in Notay (2010). A target near null space vector is provided and
 * a tentative prolongator constructed and normalized as in Panayot and
 * Pasqua (2013).
 *
 * Notes: 
 * ------
 *      - Not implemented for complex matrices due to Notay's use of
 *        a hard minimum (not absolute value) for strong connections.
 *      - Threshold beta must be 0 <= beta < 1
 *
 * Input:
 * ------
 * A_rowptr : const {int array}
 *      Row pointer for sparse array in CSR format.
 * A_colinds : const {int array}
 *      Column indices for sparse array in CSR format.
 * A_data : const {float array}
 *      Data values for sparse array in CSR format.
 * Agg_rowptr : {int array}
 *      Empty length(n+1) row pointer for sparse array in CSR format.
 * Agg_colinds : {int array}
 *      Empty length(n) column indices for sparse array in CSR format.
 * Agg_data : {float array}
 *      Empty length(n) data values for sparse array in CSR format.
 * Agg_shape : {int array, size 2} 
 *      Shape array for sparse matrix constructed in function.
 * beta : const {float}
 *      Threshold for filtering out 'strong' connections  
 *          a_ij < -beta * max_{a_ij < 0} |a_ij|
 * B : const {float array}
 *      Target near null space components provided by user.
 * n : const {int} 
 *      Problem size
 *
 * Returns
 * -------
 * Nothing, sparse prolongation / aggregation matrix modified in place.
 *
 */
template<class I, class T>
void notay_pairwise_data(const I A_rowptr[],
                         const I A_colinds[],
                         const T A_data[],
                         I Agg_rowptr[],
                         I Agg_colinds[],
                         T Agg_data[],
                         I Agg_shape[],
                         const T B[],
                         const I &n,
                         T cost[] )
{
    // Construct vector, m, to track if each node has been aggregated (-1),
    // and its number of unaggregated neighbors otherwise. Save node with
    // minimum number of neighbors as starting node. 
    std::vector<I> m(n);
    I start_ind = -1;
    I min_neighbor = std::numeric_limits<int>::max();
    for (I i=0; i<n; i++) {
        m[i] = A_rowptr[i+1] - A_rowptr[i];
        if (m[i] < min_neighbor) {
            min_neighbor = m[i];
            start_ind = i;
        }
    }
    cost[0] += n;

    // Loop until all nodes have been aggregated 
    I Nc = 0;
    I num_aggregated = 0;
    T max_single = 0.0;
    std::vector<I> singletons;
    Agg_rowptr[0] = 0;
    while (num_aggregated < n) {

        // Find unaggregated neighbor with strongest (negative) connection
        I neighbor = -1;
        T min_val = 0;
        for (I j=A_rowptr[start_ind]; j<A_rowptr[start_ind+1]; j++) {
            // Check for self loop, make sure node is unaggregated 
            if ( (start_ind != A_colinds[j] ) && (m[A_colinds[j]] >= 0) ) {
                // Find hard minimum weight of neighbor nodes 
                if (A_data[j] < min_val) {
                    neighbor = A_colinds[j];
                    min_val = A_data[j];
                }
            }
            cost[0] += 1.0;
        }

        // Form new aggregate as vector of length 1 or 2 and mark
        // nodes as aggregated. 
        std::vector<I> new_agg;
        new_agg.push_back(start_ind);
        m[start_ind] = -1;
        if (neighbor >= 0) {
            new_agg.push_back(neighbor);
            m[neighbor] = -1;
        }
        // If target bad guy provided, check for largest singleton entry 
        singletons.push_back(start_ind);
        if (std::abs(B[start_ind]) > max_single) {
            max_single = std::abs(B[start_ind]);
        }

        // Find new starting node
        start_ind = -1;
        min_neighbor = std::numeric_limits<int>::max();
        // For each node in aggregate
        for (auto it=new_agg.begin(); it!=new_agg.end(); it++) {

            // For each node strongly connected to current node
            for (I j=A_rowptr[*it]; j<A_rowptr[(*it)+1]; j++) {
                I &neighborhood = m[A_colinds[j]];
            
                // Check if node has not been aggregated
                if (neighborhood >= 0) {
                    // Decrease neighborhood size by one
                    neighborhood -= 1;

                    // Find neighboring node with smallest neighborhood 
                    if (neighborhood < min_neighbor) {
                        min_neighbor = neighborhood;
                        start_ind = A_colinds[j];
                    }
                }
                cost[0] += 1.0;
            }
        }

        // If no start node was found and there are nodes left, find
        // unaggregated node with least connections out of all nodes.
        if ( (start_ind == -1) && (num_aggregated < (n-new_agg.size())) ) {
            for (I i=0; i<n; i++) {
                if ( (m[i] >= 0) && (m[i] < min_neighbor) ) {
                    min_neighbor = m[i];
                    start_ind = i;
                }
                cost[0] += 1.0;
            }
        }

        // If target B provided, get norm restricted to this aggregate
        // in case of pair, or set norm = 1 in case of singleton
        T agg_norm = 1.0;
        if (new_agg.size() > 1) {
            for (auto it=new_agg.begin(); it!=new_agg.end(); it++) {
                agg_norm += B[*it] * B[*it];
                cost[0] += 1.0;
            }
            agg_norm = std::sqrt(agg_norm);
            cost[0] += 1.0;
        }

        // Update sparse structure for aggregation matrix with new aggregate.
        for (auto it=new_agg.begin(); it!=new_agg.end(); it++) {
            // Set all nodes in this aggregate to column Nc
            Agg_colinds[*it] = Nc;
            // Set corresponding data value, normalize if B provided.
            Agg_data[*it] = B[*it] / agg_norm;
            cost[0] += 1.0;
            // Increase row pointer by one for each node aggregated
            Agg_rowptr[num_aggregated+1] = num_aggregated+1; 
            // Increase count of aggregated nodes
            num_aggregated += 1;
        }

        // Increase coarse grid count
        Nc += 1;
    }

    // Normalize singleton data value, s_k <-- s_k / max_k |s_k|.
    if (max_single > 0) {
        for (auto it=singletons.begin(); it!=singletons.end(); it++) {
            Agg_data[*it] /= max_single;
        }
    }

    // Save shape of aggregation matrix
    Agg_shape[0] = n;
    Agg_shape[1] = Nc;
}


/* Function to approximate to perform pairwise aggregation on matrix A,
 * as in Notay (2010). Only the row pointer and column indices are 
 * constructed for the tentative prolongator, no data entries. 
 *
 * Notes: 
 * ------
 *      - Not implemented for complex matrices due to Notay's use of
 *        a hard minimum (not absolute value) for strong connections.
 *      - Threshold beta must be 0 <= beta < 1
 *
 * Input:
 * ------
 * A_rowptr : const {int array}
 *      Row pointer for sparse array in CSR format.
 * A_colinds : const {int array}
 *      Column indices for sparse array in CSR format.
 * A_data : const {float array}
 *      Data values for sparse array in CSR format.
 * Agg_rowptr : {int array}
 *      Empty length(n+1) row pointer for sparse array in CSR format.
 * Agg_colinds : {int array}
 *      Empty length(n) column indices for sparse array in CSR format.
 * Agg_shape : {int array, size 2} 
 *      Shape array for sparse matrix constructed in function.
 * beta : const {float}
 *      Threshold for filtering out 'strong' connections  
 *          a_ij < -beta * max_{a_ij < 0} |a_ij|
 * n : const {int} 
 *      Problem size
 *
 * Returns
 * -------
 * Nothing, sparse prolongation / aggregation matrix modified in place.
 *
 */
template<class I, class T>
void notay_pairwise_nodata(const I A_rowptr[],
                           const I A_colinds[],
                           const T A_data[],
                           I Agg_rowptr[],
                           I Agg_colinds[],
                           I Agg_shape[],
                           const I &n,
                           T cost[] )
{
    // Construct vector, m, to track if each node has been aggregated (-1),
    // and its number of unaggregated neighbors otherwise. Save node with
    // minimum number of neighbors as starting node. 
    std::vector<I> m(n);
    I start_ind = -1;
    I min_neighbor = std::numeric_limits<int>::max();
    for (I i=0; i<n; i++) {
        m[i] = A_rowptr[i+1] - A_rowptr[i];
        if (m[i] < min_neighbor) {
            min_neighbor = m[i];
            start_ind = i;
        }
    }
    cost[0] += n;

    // Loop until all nodes have been aggregated 
    I Nc = 0;
    I num_aggregated = 0;
    Agg_rowptr[0] = 0;
    while (num_aggregated < n) {

        // Find unaggregated neighbor with strongest (negative) connection
        I neighbor = -1;
        T min_val = 0;
        for (I j=A_rowptr[start_ind]; j<A_rowptr[start_ind+1]; j++) {
            // Check for self loop, make sure node is unaggregated 
            if ( (start_ind != A_colinds[j] ) && (m[A_colinds[j]] >= 0) ) {
                // Find hard minimum weight (<0) of neighbor nodes 
                if (A_data[j] < min_val) {
                    neighbor = A_colinds[j];
                    min_val = A_data[j];
                }
            }
            cost[0] += 1;
        }

        // Form new aggregate as vector of length 1 or 2 and mark
        // nodes as aggregated. 
        std::vector<I> new_agg;
        new_agg.push_back(start_ind);
        m[start_ind] = -1;
        if (neighbor >= 0) {
            new_agg.push_back(neighbor);
            m[neighbor] = -1;
        }

        // Find new starting node
        start_ind = -1;
        min_neighbor = std::numeric_limits<int>::max();
        // For each node in aggregate
        for (auto it=new_agg.begin(); it!=new_agg.end(); it++) {

            // For each node strongly connected to current node
            for (I j=A_rowptr[*it]; j<A_rowptr[(*it)+1]; j++) {
                I &neighborhood = m[A_colinds[j]];
            
                // Check if node has not been aggregated
                if (neighborhood >= 0) {
                    // Decrease neighborhood size by one
                    neighborhood -= 1;

                    // Find neighboring node with smallest neighborhood 
                    if (neighborhood < min_neighbor) {
                        min_neighbor = neighborhood;
                        start_ind = A_colinds[j];
                    }
                }
                cost[0] += 1;
            }
        }

        // If no start node was found and there are nodes left, find
        // unaggregated node with least connections out of all nodes.
        if ( (start_ind == -1) && (num_aggregated < (n-new_agg.size())) ) {
             for (I i=0; i<n; i++) {
                if ( (m[i] >= 0) && (m[i] < min_neighbor) ) {
                    min_neighbor = m[i];
                    start_ind = i;
                }
                cost[0] += 1.0;
            }
        }

        // Update sparse structure for aggregation matrix with new aggregate.
        for (auto it=new_agg.begin(); it!=new_agg.end(); it++) {
            // Set all nodes in this aggregate to column Nc
            Agg_colinds[*it] = Nc;
            // Increase row pointer by one for each node aggregated
            Agg_rowptr[num_aggregated+1] = num_aggregated+1; 
            // Increase count of aggregated nodes
            num_aggregated += 1;
        }

        // Increase coarse grid count
        Nc += 1;
    }
    
    // Save shape of aggregation matrix
    Agg_shape[0] = n;
    Agg_shape[1] = Nc;
}


template<class I, class T>
void notay_pairwise(const I A_rowptr[], const int A_rowptr_size,
                    const I A_colinds[], const int A_colinds_size,
                    const T A_data[], const int A_data_size,
                    const T B[], const int B_size,
                    I Agg_rowptr[], const int Agg_rowptr_size,
                    I Agg_colinds[], const int Agg_colinds_size,
                    T Agg_data[], const int Agg_data_size,
                    I Agg_shape[], const int Agg_shape_size,
                    T cost[], const int cost_size,
                    const T beta = 0)
{
    I n = A_rowptr_size-1;
    // If filtering threshold != 0
    if (beta != 0) {
        // Preallocate storage - at least n entries in each vector
        std::vector<I> rowptr;
        std::vector<I> colinds;
        std::vector<T> data;
        rowptr.reserve(n+1);   
        colinds.reserve(n);   
        data.reserve(n);
        // Filter matrix to only keep entries less than beta times
        // the minimum (negative) element in each row, i.e.
        //      { a_ij : a_ij < beta * min_j a_ij }
        notay_filter(A_rowptr, A_colinds, A_data,
                     rowptr, colinds, data, beta, n);
        cost[0] += A_data_size;

        // Pairwise matching on filtered matrix
        notay_pairwise_data(&rowptr[0], &colinds[0], &data[0],
                            Agg_rowptr, Agg_colinds, Agg_data,
                            Agg_shape, B, n, cost);
    }
    // If filtering threshold = 0, do pairwise aggregation on A
    else {
        notay_pairwise_data(A_rowptr, A_colinds, A_data,
                            Agg_rowptr, Agg_colinds, Agg_data,
                            Agg_shape, B, n, cost);
    }
}


// TODO : seg-fault with beta != 0.

template<class I, class T>
void notay_pairwise(const I A_rowptr[], const int A_rowptr_size,
                    const I A_colinds[], const int A_colinds_size,
                    const T A_data[], const int A_data_size,
                    I Agg_rowptr[], const int Agg_rowptr_size,
                    I Agg_colinds[], const int Agg_colinds_size,
                    I Agg_shape[], const int Agg_shape_size,
                    T cost[], const int cost_size,
                    const T beta = 0)
{
    I n = A_rowptr_size-1;
    // If filtering threshold != 0
    if (beta != 0) {
        // Preallocate storage - at least n entries in each vector
        std::vector<I> rowptr;
        std::vector<I> colinds;
        std::vector<T> data;
        rowptr.reserve(n+1);   
        colinds.reserve(n);   
        data.reserve(n);
        // Filter matrix to only keep entries less than beta times
        // the minimum (negative) element in each row, i.e.
        //      { a_ij : a_ij < beta * min_j a_ij }
        notay_filter(A_rowptr, A_colinds, A_data,
                     rowptr, colinds, data, beta, n);
        cost[0] += A_data_size;

        // Pairwise matching on filtered matrix
        notay_pairwise_nodata(&rowptr[0], &colinds[0], &data[0],
                              Agg_rowptr, Agg_colinds, Agg_shape, n,
                              cost);
    }
    // If filtering threshold = 0, do pairwise aggregation on A
    else {
        notay_pairwise_nodata(A_rowptr, A_colinds, A_data,
                              Agg_rowptr, Agg_colinds, Agg_shape, n,
                              cost);
    } 
}


/* Function to compute weights of a graph for an approximate matching. 
 * Weights are chosen to construct as well-conditioned of a fine grid
 * as possible based on a given smooth vector:
 *
 *      W_{ij} = 1 - 2a_{ij}b_ib_j / (a_{ii}w_i^2 + a_{jj}w_j^2)
 *
 * Notes: 
 * ------
 *      - weight matrix has same row-pointer and column indices as
 *		  A, so only the weight data is stored in an array.
 *		- If B is not provided, it is assumed to be the constant
 *		  vector.
 *
 * Input:
 * ------
 * A_rowptr : const {int array}
 *      Row pointer for sparse array in CSR format.
 * A_colinds : const {int array}
 *      Column indices for sparse array in CSR format.
 * A_data : const {float array}
 *      Data values for sparse array in CSR format.
 * weights : {float array}
 *      Empty length(n) data array for computed weights
 * B : {float array}, optional
 *      Target algebraically smooth vector to compute weights with.
 *
 * Returns
 * -------
 * Nothing, weights modified in place.
 * 
 */
template<class I, class T>
void compute_weights(const I A_rowptr[], const int A_rowptr_size,
                     const I A_colinds[], const int A_colinds_size,
                     const T A_data[], const int A_data_size,
                      	   T weights[], const int weights_size,
                     const T B[], const int B_size,
                           T cost[], const int cost_size)
{
	I n = A_rowptr_size-1;
	std::vector<T> diag(n);
    T temp_cost = 0.0;

	// Get diagonal elements of matrix
	for (I i=0; i<n; i++) {
		for (I ind=A_rowptr[i]; ind<A_rowptr[i+1]; ind++) {
			if(i == A_colinds[ind]) {
				diag[i] = A_data[ind];
			}
		}
	}

	// Compute matrix weights,
	// 		w{ij} = 1 - 2a_{ij}B_iB_j / (a_{ii}B_i^2 + a_{jj}B_j^2)
	for (I i=0; i<n; i++) {
		for (I ind=A_rowptr[i]; ind<A_rowptr[i+1]; ind++) {
			I j=A_colinds[ind];
			weights[ind] = 1.0 - (2*A_data[ind]*B[i]*B[j]) / (diag[i]*B[i]*B[i] + diag[j]*B[j]*B[j]);
            temp_cost += 3.0;
		}
	}
    temp_cost += n; // Can precompute a_{ii}B_i^2
    cost[0] += temp_cost;   
}

template<class I, class T>
void compute_weights(const I A_rowptr[], const int A_rowptr_size,
                     const I A_colinds[], const int A_colinds_size,
                     const T A_data[], const int A_data_size,
                      	   T weights[], const int weights_size,
                           T cost[], const int cost_size)
{
	I n = A_rowptr_size-1;
	std::vector<T> diag(n);
    T temp_cost = 0.0;

	// Get diagonal elements of matrix
	for (I i=0; i<n; i++) {
		for (I ind=A_rowptr[i]; ind<A_rowptr[i+1]; ind++) {
			if(i == A_colinds[ind]) {
				diag[i] = A_data[ind];
			}
		}
	}

	// Compute matrix weights. B is assumed constant, and
	//		w{ij} = 1 - 2a_{ij} / (a_{ii} + a_{jj})
	for (I i=0; i<n; i++) {
		for (I ind=A_rowptr[i]; ind<A_rowptr[i+1]; ind++) {
			I j=A_colinds[ind];
			weights[ind] = 1.0 - 2*A_data[ind] / (diag[i] + diag[j]);
            temp_cost += 2.0;
		}
	}	
    cost[0] += temp_cost;	
}

#endif
