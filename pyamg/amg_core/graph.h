#ifndef GRAPH_H
#define GRAPH_H

#include <algorithm>
#include <stack>
#include <cassert>
#include <limits>
#include <vector>
#include <iostream>

// Usage
// printv(d, d_size, "d");
template<class T>
void printv(T *v, int n, char* name)
{
  std::cout << name << " = [";
  for(int i=0; i<n; i++){
    std::cout << v[i];
    if(i < (n-1)){
      std::cout << ", ";
    }
  }
  std::cout << "]" << std::endl << std::endl;
}


// Internal assert
inline void coreassert(const bool istrue, const std::string &errormsg){
    if (!istrue){
        throw std::runtime_error("pyamg-error (amg_core) -- " + errormsg);
    }
}

/*
 * Serial maximal independent set.
 *
 * Compute a maximal independent set for a graph stored in CSR format
 * using a greedy serial algorithm
 *
 * Parameters
 * ----------
 * num_rows : int
 *     Number of rows in A (number of vertices).
 * Ap : array
 *     CSR row pointer.
 * Aj : array
 *     CSR index array.
 * active : float
 *     Value used for active vertices.
 * C : float
 *     Value used to mark non-MIS vertices.
 * F : float
 *     Value used to mark MIS vertices.
 * x : array
 *     State of each vertex.
 *
 * Returns
 * -------
 * int
 *     The number of nodes in the MIS.
 *
 * Notes
 * -----
 * Only the vertices with values with x[i] == active are considered
 * when determining the MIS.  Upon return, all active vertices will
 * be assigned the value C or F depending on whether they are in the
 * MIS or not.
 *
 */
template<class I, class T>
I maximal_independent_set_serial(const I num_rows,
                                 const I Ap[], const int Ap_size,
                                 const I Aj[], const int Aj_size,
                                 const T active,
                                 const T  C,
                                 const T  F,
                                       T  x[], const int  x_size)
{
    I N = 0;

    for(I i = 0; i < num_rows; i++){
        if(x[i] != active) continue;

        x[i] = C;
        N++;

        for(I jj = Ap[i]; jj < Ap[i+1]; jj++){
            const I j = Aj[jj];
            if(x[j] == active) {
                x[j] = F;
            }
        }

    }

    return N;
}

/*
 * Parallel maximal independent set.
 *
 * Compute a maximal independent set for a graph stored in CSR format
 * using a variant of Luby's parallel MIS algorithm.
 *
 * Parameters
 * ----------
 * num_rows : int
 *     Number of rows in A (number of vertices).
 * Ap : array
 *     CSR row pointer.
 * Aj : array
 *     CSR index array.
 * active : float
 *     Value used for active vertices.
 * C : float
 *     Value used to mark non-MIS vertices.
 * F : float
 *     Value used to mark MIS vertices.
 * x : array, output
 *     State of each vertex.
 * y : array
 *     Random values for each vertex.
 * max_iters : int
 *     Maximum number of iterations By default max_iters=-1 and no limit is imposed.
 *
 * Returns
 * -------
 * int
 *     The number of nodes in the MIS.
 *
 * Notes
 * -----
 * Only the vertices with values with x[i] == active are considered
 * when determining the MIS.  Upon return, all active vertices will
 * be assigned the value C or F depending on whether they are in the
 * MIS or not.
 *
 */
template<class I, class T, class R>
I maximal_independent_set_parallel(const I num_rows,
                                   const I Ap[], const int Ap_size,
                                   const I Aj[], const int Aj_size,
                                   const T active,
                                   const T  C,
                                   const T  F,
                                         T  x[], const int  x_size,
                                   const R  y[], const int  y_size,
                                   const I  max_iters)
{
    I N = 0;
    I num_iters = 0;

    bool active_nodes = true;

    while(active_nodes && (max_iters == -1 || num_iters < max_iters)){
        active_nodes = false;

        num_iters++;

        for(I i = 0; i < num_rows; i++){
            const R yi = y[i];

            if(x[i] != active) continue;

            const I row_start = Ap[i];
            const I row_end   = Ap[i+1];

            I jj;

            for(jj = row_start; jj < row_end; jj++){
                const I j  = Aj[jj];
                const T xj = x[j];

                if(xj == C) {
                    x[i] = F;                      //neighbor is MIS
                    break;
                }

                if(xj == active){
                    const R yj = y[j];
                    if(yj > yi)
                        break;                     //neighbor is larger
                    else if (yj == yi && j > i)
                        break;                     //tie breaker goes to neighbor
                }
            }

            if(jj == row_end){
                for(jj = row_start; jj < row_end; jj++){
                    const I j  = Aj[jj];
                    if(x[j] == active)
                        x[j] = F;
                }
                N++;
                x[i] = C;
            } else {
                active_nodes = true;
            }
        }
    } // end while

    return N;
}

/*
 *  Compute a vertex coloring for a graph stored in CSR format.
 *
 *  The coloring is computed by removing maximal independent sets
 *  of vertices from the graph.  Specifically, at iteration i an
 *  independent set of the remaining subgraph is constructed and
 *  assigned color i.
 *
 *  Returns the K, the number of colors used in the coloring.
 *  On return x[i] \in [0,1, ..., K - 1] will contain the color
 *  of the i-th vertex.
 *
 */
template<class I, class T>
T vertex_coloring_mis(const I num_rows,
                      const I Ap[], const int Ap_size,
                      const I Aj[], const int Aj_size,
                            T  x[], const int  x_size)
{
    std::fill( x, x + num_rows, -1);

    I N = 0;
    T K = 0;

    while(N < num_rows){
        N += maximal_independent_set_serial(num_rows,Ap,Ap_size,Aj,Aj_size,-1-K,K,-2-K,x,x_size);
        K++;
    }

    return K;
}


/*
 *  Applies the first fit heuristic to a graph coloring.
 *
 *  For each vertex with color K the vertex is assigned the *first*
 *  available color such that no neighbor of the vertex has that
 *  color.  This heuristic is used to reduce the number of color used
 *  in the vertex coloring.
 *
 */
template<class I, class T>
void vertex_coloring_first_fit(const I num_rows,
                               const I Ap[], const int Ap_size,
                               const I Aj[], const int Aj_size,
                                     T  x[], const int  x_size,
                               const T  K)
{
    for(I i = 0; i < num_rows; i++){
        if(x[i] != K) continue;
        std::vector<bool> mask(K,false);
        for(I jj = Ap[i]; jj < Ap[i+1]; jj++){
            const I j = Aj[jj];
            if(  i == j  ) continue; //ignore diagonal
            if( x[j] < 0 ) continue; //ignore uncolored vertices
            mask[x[j]] = true;
        }
        x[i] = std::find(mask.begin(), mask.end(), false) - mask.begin();
    }
}



/*
 * Compute a vertex coloring of a graph using the Jones-Plassmann algorithm.
 *
 * Parameters
 * ----------
 * num_rows : int
 *     Number of rows in A (number of vertices).
 * Ap : array
 *     CSR row pointer.
 * Aj : array
 *     CSR index array.
 * x : array, inplace
 *     Color of each vertex.
 * z : array
 *     Initial random values for each vertex.
 *
 * Notes
 * -----
 *     Arrays x and z will be overwritten.
 *
 * References
 * ----------
 * .. [Jones92] Mark T. Jones and Paul E. Plassmann
 *    A Parallel Graph Coloring Heuristic
 *    SIAM Journal on Scientific Computing 14:3 (1993) 654--669
 *    http://citeseer.ist.psu.edu/jones92parallel.html
 */
template<class I, class T, class R>
T vertex_coloring_jones_plassmann(const I num_rows,
                                  const I Ap[], const int Ap_size,
                                  const I Aj[], const int Aj_size,
                                        T  x[], const int  x_size,
                                        R  z[], const int  z_size)
{
    std::fill( x, x + num_rows, -1);

    for(I i = 0; i < num_rows; i++){
        z[i] += Ap[i+1] - Ap[i];
    }

    I N = 0;
    T K = 0; //iteration number

    while(N < num_rows){
        N += maximal_independent_set_parallel(num_rows,Ap,Ap_size,Aj,Aj_size,-1,K,-2,x,x_size,z,z_size,1);
        for(I i = 0; i < num_rows; i++){
            if(x[i] == -2)
                x[i] = -1;
        }
        vertex_coloring_first_fit(num_rows,Ap,Ap_size,Aj,Aj_size,x,x_size,K);
        K++;
    }

    return *std::max_element(x, x + num_rows);
}


/*
 * Compute a vertex coloring of a graph using parallel Largest-Degree-First (LDF).
 *
 * Parameters
 * ----------
 * num_rows : int
 *     Number of rows in A (number of vertices).
 * Ap : array
 *     CSR row pointer.
 * Aj : array
 *     CSR index array.
 * x : array
 *     Color of each vertex.
 * y : array
 *     Initial random values for each vertex.
 *
 * References
 * ----------
 * .. [LDF] J. R. Allwright and R. Bordawekar and P. D. Coddington and K. Dincer and C. L. Martin
 *    A Comparison of Parallel Graph Coloring Algorithms
 *    DRAFT SCCS-666
 *    http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.45.4650
 *
 */
template<class I, class T, class R>
T vertex_coloring_LDF(const I num_rows,
                      const I Ap[], const int Ap_size,
                      const I Aj[], const int Aj_size,
                            T  x[], const int  x_size,
                      const R  y[], const int  y_size)
{
    std::fill( x, x + num_rows, -1);

    std::vector<R> weights(num_rows);

    I N = 0;
    T K = 0; //iteration number

    while(N < num_rows){
        // weight is # edges in induced subgraph + random value
        for(I i = 0; i < num_rows; i++){
            if(x[i] != -1) continue;
            I num_neighbors = 0;
            for(I jj = Ap[i]; jj < Ap[i+1]; jj++){
                I j = Aj[jj];
                if(x[j] == -1 && i != j)
                    num_neighbors++;
            }
            weights[i] = y[i] + num_neighbors;
        }

        N += maximal_independent_set_parallel(num_rows,Ap,Ap_size,Aj,Aj_size,-1,K,-2,x,x_size,&weights[0],num_rows,1);
        for(I i = 0; i < num_rows; i++){
            if(x[i] == -2)
                x[i] = -1;
        }
        vertex_coloring_first_fit(num_rows,Ap,Ap_size,Aj,Aj_size,x,x_size,K);
        K++;
    }

    return *std::max_element(x, x + num_rows);
}

/*
 * Floyd-Warshall on a subgraph or cluster of nodes in A.
 *
 * Parameters
 * ----------
 * num_nodes : int
 *     Number of nodes (number of rows in A).
 * Ap : array
 *     CSR row pointer for A, (num_nodes, 1).
 * Aj : array
 *     CSR column index for A, (num_edges, 1).
 * Ax : array
 *     CSR data array (edge weights), (num_edges, 1).
 * D : array
 *     FW distance array, (max_size, max_size).
 * P : array
 *     FW predecessor array, (max_size, max_size).
 * C : array
 *     FW global index for current cluster, (N, 1).
 * L : array
 *     FW local index for current cluster, (num_nodes, 1).
 * m : array
 *     Cluster index, (num_nodes, 1).
 * a : array
 *     Center of current cluster.
 * N : int
 *     Size of current cluster.
 *
 * Notes
 * -----
 *     - There are no checks within this kernel
 *     - There is no initialization within this kernel
 *     - Ax > 0 is assumed
 *     - Only a slice of C is passed to Floyd–Warshall.  See center_nodes.
 *     - C[i] is the global index of i for i=0, ..., N in the current cluster
 *     - N = |C|
 *     - L = local indices, nx1 (-1 if not in the cluster)
 *     - assumes a fully connected (directed) graph
 *
 * References
 * ----------
 * .. [1] Graph Center:   https://en.wikipedia.org/wiki/Graph_center
 * .. [2] Floyd-Warshall: https://en.wikipedia.org/wiki/Floyd–Warshall_algorithm
 * .. [3] Graph Distance: https://en.wikipedia.org/wiki/Distance_(graph_theory)
 *
 */
template<class I, class T>
void floyd_warshall(const I num_nodes,
                    const I Ap[], const int Ap_size,
                    const I Aj[], const int Aj_size,
                    const T Ax[], const int Ax_size,
                          T D[],  const int D_size,
                          I P[],  const int P_size,
                    const I C[],  const int C_size,
                    const I L[],  const int L_size,
                    const I m[], const int m_size,
                    const I a,
                    const I N
                    )
{
  const double tol = 1e-14; // precision tolerance
  // initialize D and P
  for(I _i = 0; _i < N; _i++){              // each node in the cluster, local index
    I i = C[_i];                            // global index
    for(I jj = Ap[i]; jj < Ap[i+1]; jj++){  // each neighbor
      I j = Aj[jj];                         // global index
      I _j = L[j];                          // local index

      if(m[j] == a){                        // check to see if neighbor is in cluster a
        I _ij = _i * N + _j;                // row major indexing into D, P
        D[_ij] = Ax[jj];                    // edge weight
        P[_ij] = i;                         // predecessor
      }
    }
  }
  for(I _i = 0; _i < N; _i++){              // each node in the cluster, local index
    I i = C[_i];                            // global index
    I _ii = _i * N + _i;                    // row major indexing into D, P
    D[_ii] = 0;
    P[_ii] = i;
  }

  for(I k = 0; k < N; k++){
    for(I i = 0; i < N; i++){
      for(I j = 0; j < N; j++){
        I ij = i * N + j;
        I ik = i * N + k;
        I kj = k * N + j;
        if(D[ij] > (D[ik] + D[kj] + tol)){
          D[ij] = D[ik] + D[kj];
          P[ij] = P[kj];
        }
      }
    }
  }
}

// Update center nodes for a cluster
//
// Parameters
// ----------
//   num_nodes  : (IN) number of nodes (number of rows in A)
//   Ap[]       : (IN) CSR row pointer for A                              (num_nodes x 1)
//   Aj[]       : (IN) CSR column index for A                             (num_edges x 1)
//   Ax[]       : (IN) CSR data array (edge weights)                      (num_edges x 1)
// Cptr[]       : (INOUT) ptr to start of indices in C for each cluster   (num_clusters x 1)
//    D[]       : (INOUT) FW distance array                               (max_size x max_size)
//    P[]       : (INOUT) FW predecessor array                            (max_size x max_size)
//    C[]       : (INOUT) FW global index for current cluster             (num_nodes x 1)
//    L[]       : (INOUT) FW local index for current cluster              (num_nodes x 1)
//    q         : (INOUT) FW work array for D**2                          (max_size x max_size)
//    c         : (INOUT) cluster center                                  (num_clusters x 1)
//    d         : (INOUT) distance to cluster center                      (num_nodes x 1)
//    m         : (INOUT) cluster index                                   (num_nodes x 1)
//    p         : (INOUT) predecessor on shortest path to center          (num_nodes x 1)
//    pc        : (INOUT) predecessor count                               (num_nodes x 1)
//    s         : (INOUT) cluster size                                    (num_clusters x 1)
//
// Returns
// -------
// changed : flag to indicate a change in arrays D or P
//
// Notes
// -----
// - sort into clusters first O(n)
//     s: [4           2     4               ....
//  Cptr: [0           4     6              11 ...
//         |           |     |              |
//         v           v     v              v
//     C: [87 99 4  6  82 13 15 9  12 55 66 77 ...]
//               ^  ^           ^
//               |  |________   |_____
//               |_____      |        |
//                     |     |        |
//     L: [            2     3        1
//         ^  ^  ^  ^  ^  ^  ^  ^  ^  ^  ^
//         |  |  |  |  |  |  |  |  |  |  |  ...
//         0  1  2  3  4  5  6  7  8  9  10 ...
// - pass pointer to start of each C[start,...., start+N]
// - N is the cluster size
template<class I, class T>
bool center_nodes(const I num_nodes,
                  const I Ap[], const int Ap_size,
                  const I Aj[], const int Aj_size,
                  const T Ax[], const int Ax_size,
                      I Cptr[], const int Cptr_size,// to set up FW
                         T D[],  const int D_size,  // for FW
                         I P[],  const int P_size,  // for FW
                         I C[],  const int C_size,  // for FW
                         I L[],  const int L_size,  // for FW
                         T q[],  const int q_size,  // to hold D**2
                         I c[],  const int c_size,  // from BF
                         T d[],  const int d_size,  // from BF
                         I m[],  const int m_size,  // from BF
                         I p[],  const int p_size,  // from BF
                         I pc[], const int pc_size, // from BF
                         I s[],  const int s_size)  // from BF
{
  I num_clusters = c_size;
  bool changed = false; // return a change on d or p
  const double tol = 1e-14; // precision tolerance

  // point the first empty slot in cluster block of C
  I Clast = 0;
  for(I a=0; a<num_clusters; a++){
    Cptr[a] = Clast;
    Clast += s[a];
  }
  // fill in the global index into the next spot in the cluster
  for(I i=0; i<num_nodes; i++){
    I a = m[i];                      // get the cluster id
    I nextspot = Cptr[a];            // get the next spot
    C[nextspot] = i;                 // set the global index
    Cptr[a]++;                       // update the next spot
  }
  // reset pointer to the first empty slot in cluster block of C
  Clast = 0;
  for(I a=0; a<num_clusters; a++){
    Cptr[a] = Clast;
    Clast += s[a];
  }
  // set L, local indices for each global C
  for(I a=0; a<num_clusters; a++){
    for(I _j=0; _j<s[a]; _j++){
      L[C[Cptr[a]+_j]] = _j;            // set the local index for the node
    }
  }

  // for each cluster a
  for(I a=0; a<num_clusters; a++){
    // call Floyd–Warshall for cluster a
    // D: (max_N, max_N)
    // P: (max_N, max_N)
    // C: (max_N,)
    // L: (num_nodes,)
    // initialize
    std::fill(D, D+D_size, std::numeric_limits<T>::infinity());
    std::fill(P, P+P_size, -1);

    I N = s[a];
    floyd_warshall(num_nodes, Ap, Ap_size, Aj, Aj_size, Ax, Ax_size,
                   D,  D_size, P,  P_size, C+Cptr[a], s[a], L,  L_size,
                   m,  m_size, a, N);

    // sum of square distances to the other nodes
    for(I _i=0; _i<N; _i++){
      q[_i] = 0;
      for(I _j=0; _j<N; _j++){
        I _ij = _i * N + _j;
        q[_i] += D[_ij]*D[_ij];
      }
    }

    I i = c[a];                   // global index of the cluster center
    for(I _j=0; _j<N; _j++){
      if(q[_j] < q[L[i]] - tol) { // is j (strictly) better?
        i = C[Cptr[a] + _j];      // global index of every node in the cluster
      }
    }
    if(i != c[a]){                // if we've found a new center, then...
      c[a] = i;                   // i is the new center
      I _i = L[i];
      for(I _j=0; _j<N; _j++){    // update metrics for all nodes in the cluster
        I j = C[Cptr[a] + _j];    // global index of every node in the cluster
        I _ij = _i * N + _j;

        d[j] = D[_ij];            // new distance from i->j

        pc[p[j]]--;               // update predecessor count (old j)
        p[j] = P[_ij];            // set predecessor
        pc[p[j]]++;               // update predecessor count (new j)

        changed = true;
      }
    }
  }
  return changed;
}

/*
 * Apply one iteration of Bellman-Ford iteration on a distance graph stored in CSR format.
 *
 * Parameters
 * ----------
 * num_nodes : int
 *     Number of nodes (number of rows in A).
 * Ap : array
 *     CSR row pointer.
 * Aj : array
 *     CSR index array.
 * Ax : array
 *     CSR data array (edge lengths).
 * c : array
 *     Cluster center.
 * d : array, inplace
 *     Distance to nearest center.
 * m : array, inplace
 *     Cluster index for each node.
 * p : array, inplace
 *     Predecssor on the shortest path to center.
 *
 * Notes
 * -----
 * - There are no checks within this kernel.
 * - Ax is assumed to be positive
 *
 * Initializations::
 *
 *  - d[i] = 0 if i is a center, else inf
 *  - m[i] = 0 .. num_clusters if in a cluster, else -1
 *  - p[i] = -1
 *
 * See Also
 * --------
 * pyamg.graph.bellman_ford
 *
 * References
 * ----------
 * .. [1] Bellman-Ford Wikipedia: http://en.wikipedia.org/wiki/Bellman-Ford_algorithm
 *
 */
template<class I, class T>
void bellman_ford(const I num_nodes,
                  const I Ap[], const int Ap_size,
                  const I Aj[], const int Aj_size,
                  const T Ax[], const int Ax_size,
                  const I c[],  const int c_size,
                        T d[],  const int d_size,
                        I m[],  const int m_size,
                        I p[],  const int p_size)
{
  bool done = false;

  while (!done) {
    done = true;
    for(I i = 0; i < num_nodes; i++){
      for(I jj = Ap[i]; jj < Ap[i+1]; jj++){
        const I j = Aj[jj];
        const T Aij = Ax[jj];
        if(d[i] + Aij < d[j]){
          d[j] = d[i] + Aij;
          m[j] = m[i];
          p[j] = i;
          done = false; // found a change, keep going
        }
      }
    }
  }
}


//  Bellman-Ford with a heuristic to balance cluster sizes
//
//  This version is modified to break distance ties by assigning nodes
//  to the cluster with the fewest points, while preserving cluster
//  connectivity. This results in more balanced cluster sizes.
//
//  Parameters
//  ----------
//   num_nodes  : (IN) number of nodes (number of rows in A)
//   Ap[]       : (IN) CSR row pointer for A                              (num_nodes x 1)
//   Aj[]       : (IN) CSR column index for A                             (num_edges x 1)
//   Ax[]       : (IN) CSR data array (edge weights)                      (num_edges x 1)
//    c         : (INOUT) cluster center                                  (num_clusters x 1)
//    d         : (INOUT) distance to cluster center                      (num_nodes x 1)
//    m         : (INOUT) cluster index                                   (num_nodes x 1)
//    p         : (INOUT) predecessor on shortest path to center          (num_nodes x 1)
//    pc        : (INOUT) number of predecessors                          (num_nodes x 1)
//    s         : (INOUT) cluster size                                    (num_clusters x 1)
//
//  Notes
//  -----
//  - There are no checks within this kernel.
//  - Ax > 0 is assumed
//
//  Initializations
//  ---------------
//  d[i] = 0 if i is a center, else 0
//  m[i] = 0, ..., nclusters if i is in a cluster, else -1
//  p = -1
//  pc = 0
//  s = 1
//
//  See Also
//  --------
//  pyamg.graph.bellman_ford
template<class I, class T>
bool bellman_ford_balanced(const I num_nodes,
                           const I Ap[], const int Ap_size,
                           const I Aj[], const int Aj_size,
                           const T Ax[], const int Ax_size,
                           const I  c[], const int c_size,
                                 T  d[], const int  d_size,
                                 I  m[], const int  m_size,
                                 I  p[], const int  p_size,
                                 I pc[], const int pc_size,
                                 I  s[], const int  s_size,
                           const bool tiebreaking)
{
  bool done;            // did we make any changes during this iteration?
  bool changed = false; // indicate a change for the return
  bool swap;            // should we swap node i to the same clusters as node j?
  int iter = 0;         // iteration count for safety check
  const double tol = 1e-14; // precision tolerance

  do{
    done = true;
    for(I i = 0; i < num_nodes; i++){
      for(I jj = Ap[i]; jj < Ap[i+1]; jj++){ // all neighbors of node i

        if(m[i] < 0){ // if i is unassigned, continue
          continue;
        }

        const I j = Aj[jj];
        const T Aij = Ax[jj];
        swap = false;

        //if(d[i] + Aij < d[j]){  // standard Bellman-Ford
        if(d[j] - (d[i] + Aij) > 2*tol){  // if both are finite and close
          swap = true;
        }

        if(m[j] > -1 && tiebreaking){             // if j is unassigned, do not consider the tie
          if(std::abs(d[i] + Aij - d[j]) < tol){  // if both are finite and close
            if((s[m[i]] + 1) < s[m[j]]){          // if the size of cluster j is larger
              if(pc[j] == 0){                     // if the predecessor count is zero
                swap = true;
              }
            }
          }
        }

        if(swap){
          if(m[j] >= 0){     // if part of a cluster
            s[m[j]]--;       // update cluster size (removing j)
            pc[p[j]]--;      // update predecessor count (removing j)
          }

          m[j] = m[i];       // swap node i to the cluster of node j
          d[j] = d[i] + Aij; // use the distance through node j
          p[j] = i;          // mark the predecessor

          s[m[j]]++;         // update cluster size (adding j)
          pc[p[j]]++;        // update predecessor count (adding j)

          done = false;
          changed = true;
        }
      }
    }

    // safety check, regular unweighted BF is actually O(|V|.|E|)
    if (++iter > num_nodes*num_nodes){
      throw std::runtime_error("pyamg-error (amg_core) -- too many iterations!");
    }
  } while(!done);
  return changed;
}

/*
 * Find the most interior nodes.
 *
 * Parameters
 * ----------
 * num_nodes : int
 *     Number of nodes (number of rows in A).
 * Ap : array
 *     CSR row pointer for adjacency matrix A, (num_nodes, 1).
 * Aj : array
 *     CSR index array, (num_edges, 1).
 * Ax : array
 *     CSR data array (edge lengths), (num_edges, 1).
 * c : array, num_cluster
 *     Cluster centers, (num_clusters, 1).
 * d : array, num_nodes
 *     Distance to nearest seed, (num_nodes, 1).
 * m : array, num_nodes
 *     Cluster index for each node, (num_nodes, 1).
 * p : array
 *     Predecessor on shortest path to center, (num_nodes, 1).
 *
 * Notes
 * -----
 * - There are no checks within this kernel.
 * - Ax is assumed to be positive
 *
 * References
 * ----------
 * .. [Bell2008] Nathan Bell, Algebraic Multigrid for Discrete Differential Forms
 *    PhD thesis (UIUC), August 2008.
 *
 */
template<class I, class T>
bool most_interior_nodes(const I num_nodes,
                   const I Ap[], const int Ap_size,
                   const I Aj[], const int Aj_size,
                   const T Ax[], const int Ax_size,
                         I  c[], const int  c_size,
                         T  d[], const int  d_size,
                         I  m[], const int  m_size,
                         I  p[], const int  p_size
                        )
{
  // find boundaries
  // for each edge,
  //   if i and j are in difference clusters,
  //   then i is a boundary node
  std::fill(d, d+d_size, std::numeric_limits<T>::infinity());
  for(I i = 0; i < num_nodes; i++){
    for(I jj = Ap[i]; jj < Ap[i+1]; jj++){
      I j = Aj[jj];
      if( m[i] != m[j] ){
        d[i] = 0;
        break;
      }
    }
  }

  bool changed = false; // indicate a change for the return

  // find the distance to the closest boundary point as marked in d
  // c is unused
  // m should be invariant under this operation
  bellman_ford(num_nodes, Ap, Ap_size, Aj, Aj_size, Ax, Ax_size, c, c_size,
               d, d_size, m, m_size, p, p_size);

  // determine the new centers: the node furthest from a boundary
  for(I i = 0; i < num_nodes; i++){
    const I a = m[i];

    if (a == -1) // node belongs to no cluster
      continue;

    if( d[c[a]] < d[i] ) {
      c[a] = i;
      changed = true;   // center changed
    }
  }
  return changed;
}


/*
 * Propagate (key,value) pairs across a graph in CSR format.
 *
 * Each vertex in the graph looks at all neighboring vertices
 * and selects the (key,value) pair such that the value is
 * greater or equal to every other neighboring value.  If
 * two (key,value) pairs have the same value, the one with
 * the higher index is chosen
 *
 * This method is used within a parallel MIS-k method to
 * propagate the local maximia's information to neighboring
 * vertices at distance K > 1 away.
 *
 */
template<typename IndexType, typename ValueType>
void csr_propagate_max(const IndexType  num_rows,
                       const IndexType  Ap[],
                       const IndexType  Aj[],
                       const IndexType  i_keys[],
                             IndexType  o_keys[],
                       const ValueType  i_vals[],
                             ValueType  o_vals[])
{
    for(IndexType i = 0; i < num_rows; i++){

        IndexType k_max = i_keys[i];
        ValueType v_max = i_vals[i];

        for(IndexType jj = Ap[i]; jj < Ap[i+1]; jj++){
            const IndexType j   = Aj[jj];
            const IndexType k_j = i_keys[j];
            const ValueType v_j = i_vals[j];

            if( k_j == k_max ) continue;
            if( v_j < v_max ) continue;
            if( v_j > v_max || k_j > k_max ){
                k_max = k_j;
                v_max = v_j;
            }
        }

        o_keys[i] = k_max;
        o_vals[i] = v_max;
    }
}

/*
 * Compute MIS-k.
 *
 * Parameters
 * ----------
 * num_rows : int
 *     Number of rows in A (number of vertices).
 * Ap : array
 *     CSR row pointer.
 * Aj : array
 *     CSR index array.
 * k : int
 *     Minimum separation between MIS vertices.
 * x : array, inplace
 *     State of each vertex (1 if in the MIS, 0 otherwise).
 * y : array
 *     Random values used during parallel MIS algorithm.
 * max_iters : int
 *     Maximum number of iterations to use (default, no limit).
 *
 * Returns
 * -------
 * None
 *     In place.
 *
 * Notes
 * -----
 * Compute a distance-k maximal independent set for a graph stored
 * in CSR format using a parallel algorithm.  An MIS-k is a set of
 * vertices such that all vertices in the MIS-k are separated by a
 * path of at least K+1 edges and no additional vertex can be added
 * to the set without destroying this property.  A standard MIS
 * is therefore a MIS-1.
 *
 */
template<class I, class T, class R>
void maximal_independent_set_k_parallel(const I num_rows,
                                        const I Ap[], const int Ap_size,
                                        const I Aj[], const int Aj_size,
                                        const I  k,
                                              T  x[], const int  x_size,
                                        const R  y[], const int  y_size,
                                        const I  max_iters)
{
    std::vector<bool> active(num_rows,true);

    std::vector<I> i_keys(num_rows);
    std::vector<I> o_keys(num_rows);
    std::vector<R> i_vals(num_rows);
    std::vector<R> o_vals(num_rows);

    for(I i = 0; i < num_rows; i++){
        i_keys[i] = i;
        i_vals[i] = y[i];
        x[i] = 0;
    }

    for(I iter = 0; max_iters == -1 || iter < max_iters; iter++){
        for(I i = 0; i < k; i++){
            csr_propagate_max(num_rows, Ap, Aj, &(i_keys[0]), &(o_keys[0]), &(i_vals[0]), &(o_vals[0]));
            std::swap(i_keys, o_keys);
            std::swap(i_vals, o_vals);
        }

        for(I i = 0; i < num_rows; i++){
            if( i_keys[i] == i && active[i]){
                x[i] = 1; // i is a MIS-k node
            }

            i_keys[i] = i;
            i_vals[i] = x[i];
        }

        I rank = 0;
        //while(rank < k && 2*(k - rank) > k){
        //    csr_propagate_max(num_rows, Ap, Aj, &(i_keys[0]), &(o_keys[0]), &(i_vals[0]), &(o_vals[0]));
        //    std::swap(i_keys, o_keys);
        //    std::swap(i_vals, o_vals);
        //    rank++;
        //}

        while(rank < k){
            csr_propagate_max(num_rows, Ap, Aj, &(i_keys[0]), &(o_keys[0]), &(i_vals[0]), &(o_vals[0]));
            std::swap(i_keys, o_keys);
            std::swap(i_vals, o_vals);
            rank++;
        }

        bool work_left = false;

        for(I i = 0; i < num_rows; i++){
            if(i_vals[i] == 1){
                active[i] =  false;
                i_vals[i] = -1;
            } else {
                i_vals[i] = y[i];
                work_left = true;
            }
            i_keys[i] = i;
        }

        if( !work_left )
            return;
    }

}

/*
 * Breadth first search.
 *
 * Compute a breadth first search of a graph in CSR format
 * beginning at a given seed vertex.
 *
 * Parameters
 * ----------
 * num_rows : int
 *     Number of rows in A (number of vertices).
 * Ap : array
 *     CSR row pointer.
 * Aj : array
 *     CSR index array.
 * order : array, num_rows, inplace
 *     Records the order in which vertices were searched.
 * level : array, num_rows, inplace
 *     Records the level set of the searched vertices (i.e. the minimum distance to the seed).
 *
 * Returns
 * -------
 * None
 *     In place.
 *
 * Notes
 * -----
 * The values of the level must be initialized to -1.
 *
 */
template <class I>
void breadth_first_search(const I Ap[], const int Ap_size,
                          const I Aj[], const int Aj_size,
                          const I seed,
                                I order[], const int order_size,
                                I level[], const int level_size)
{
    // initialize seed
    order[0]    = seed;
    level[seed] = 0;

    I N = 1;
    I level_begin = 0;
    I level_end   = N;

    I current_level = 1;

    while(level_begin < level_end){
        // for each node of the last level
        for(I ii = level_begin; ii < level_end; ii++){
            const I i = order[ii];

            // add all unmarked neighbors to the queue
            for(I jj = Ap[i]; jj < Ap[i+1]; jj++){
                const I j = Aj[jj];
                if(level[j] == -1){
                    order[N] = j;
                    level[j] = current_level;
                    N++;
                }
            }
        }

        level_begin = level_end;
        level_end   = N;
        current_level++;
    }

}


/*
 * Compute the connected components of a graph stored in CSR format.
 *
 * Parameters
 * ----------
 * num_rows : int
 *     Number of rows in A (number of vertices).
 * Ap : array
 *     CSR row pointer.
 * Aj : array
 *     CSR index array.
 * components : array, num_rows
 *     Component labels.
 *
 * Returns
 * -------
 * None
 *     In place.
 *
 * Notes
 * -----
 * Vertices belonging to each component are marked with a unique integer
 * in the range [0,K), where K is the number of components.
 *
 */
template <class I>
I connected_components(const I num_nodes,
                       const I Ap[], const int Ap_size,
                       const I Aj[], const int Aj_size,
                             I components[], const int components_size)
{
    std::fill(components, components + num_nodes, -1);
    std::stack<I> DFS;
    I component = 0;

    for(I i = 0; i < num_nodes; i++)
    {
        if(components[i] == -1)
        {
            DFS.push(i);
            components[i] = component;

            while (!DFS.empty())
            {
                I top = DFS.top();
                DFS.pop();

                for(I jj = Ap[top]; jj < Ap[top + 1]; jj++){
                    const I j = Aj[jj];
                    if(components[j] == -1){
                        DFS.push(j);
                        components[j] = component;
                    }
                }
            }

            component++;
        }
    }

    return component;
}

#endif
