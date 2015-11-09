#ifndef GRAPH_H
#define GRAPH_H

#include <algorithm>
#include <stack>
#include <cassert>
#include <limits>
#include <vector>

/*
 *  Compute a maximal independent set for a graph stored in CSR format
 *  using a greedy serial algorithm
 *
 *  Parameters
 *      num_rows   - number of rows in A (number of vertices)
 *      Ap[]       - CSR row pointer
 *      Aj[]       - CSR index array
 *      active     - value used for active vertices        (input)
 *       C         - value used to mark non-MIS vertices   (output)
 *       F         - value used to mark MIS vertices       (output)
 *      x[]        - state of each vertex
 *
 *
 *  Returns:
 *      The number of nodes in the MIS.
 *
 *  Notes:
 *      Only the vertices with values with x[i] == active are considered
 *      when determining the MIS.  Upon return, all active vertices will
 *      be assigned the value C or F depending on whether they are in the
 *      MIS or not.
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
 *  Compute a maximal independent set for a graph stored in CSR format
 *  using a variant of Luby's parallel MIS algorithm
 *
 *  Parameters
 *      num_rows   - number of rows in A (number of vertices)
 *      Ap[]       - CSR row pointer
 *      Aj[]       - CSR index array
 *      active     - value used for active vertices        (input)
 *       C         - value used to mark non-MIS vertices   (output)
 *       F         - value used to mark MIS vertices       (output)
 *      x[]        - state of each vertex
 *      y[]        - random values for each vertex
 *      max_iters  - maximum number of iterations
 *                   by default max_iters=-1 and no limit
 *                   is imposed
 *
 *  Returns:
 *      The number of nodes in the MIS.
 *
 *  Notes:
 *      Only the vertices with values with x[i] == active are considered
 *      when determining the MIS.  Upon return, all active vertices will
 *      be assigned the value C or F depending on whether they are in the
 *      MIS or not.
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

    //std::cout << std::endl << "Luby's finished in " << num_iters << " iterations " << std::endl;

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
 * Compute a vertex coloring of a graph using the Jones-Plassmann algorithm
 *
 *  Parameters
 *      num_rows   - number of rows in A (number of vertices)
 *      Ap[]       - CSR row pointer
 *      Aj[]       - CSR index array
 *      x[]        - color of each vertex
 *      y[]        - initial random values for each vertex
 *
 *  Notes:
 *      Arrays x and y will be overwritten
 *
 *  References:
 *      Mark T. Jones and Paul E. Plassmann
 *      A Parallel Graph Coloring Heuristic
 *      SIAM Journal on Scientific Computing 14:3 (1993) 654--669
 *      http://citeseer.ist.psu.edu/jones92parallel.html
 *
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
 * Compute a vertex coloring of a graph using the parallel
 * Largest-Degree-First (LDF) algorithm
 *
 *  Parameters
 *      num_rows   - number of rows in A (number of vertices)
 *      Ap[]       - CSR row pointer
 *      Aj[]       - CSR index array
 *      x[]        - color of each vertex
 *      y[]        - initial random values for each vertex
 *
 *   References:
 *     J. R. Allwright and R. Bordawekar and P. D. Coddington and K. Dincer and C. L. Martin
 *     A Comparison of Parallel Graph Coloring Algorithms
 *     DRAFT SCCS-666
 *     http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.45.4650
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
 * Apply one iteration of Bellman-Ford iteration on a distance
 * graph stored in CSR format.
 *
 *  Parameters
 *      num_rows   - number of rows in A (number of vertices)
 *      Ap[]       - CSR row pointer
 *      Aj[]       - CSR index array
 *      Ax[]       - CSR data array (edge lengths)
 *      x[]        - (current) distance to nearest center
 *      y[]        - (current) index of nearest center
 *
 *  References:
 *      http://en.wikipedia.org/wiki/Bellman-Ford_algorithm
 */
template<class I, class T>
void bellman_ford(const I num_rows,
                  const I Ap[], const int Ap_size,
                  const I Aj[], const int Aj_size,
                  const T Ax[], const int Ax_size,
                        T  x[], const int  x_size,
                        I  z[], const int  z_size)
{
    for(I i = 0; i < num_rows; i++){
        T xi = x[i];
        I zi = z[i];
        for(I jj = Ap[i]; jj < Ap[i+1]; jj++){
            const I j = Aj[jj];
            const T d = Ax[jj] + x[j];
            if(d < xi){
                xi = d;
                zi = z[j];
            }
        }
        x[i] = xi;
        z[i] = zi;
    }
}


/*
 * Perform Lloyd clustering on a distance graph
 *
 *  Parameters
 *      num_rows       - number of rows in A (number of vertices)
 *      Ap[]           - CSR row pointer
 *      Aj[]           - CSR index array
 *      Ax[]           - CSR data array (edge lengths)
 *      x[num_rows]    - distance to nearest seed
 *      y[num_rows]    - cluster membership
 *      z[num_centers] - cluster centers
 *
 *  References
 *      Nathan Bell
 *      Algebraic Multigrid for Discrete Differential Forms
 *      PhD thesis (UIUC), August 2008
 *
 */
template<class I, class T>
void lloyd_cluster(const I num_rows,
                   const I Ap[], const int Ap_size,
                   const I Aj[], const int Aj_size,
                   const T Ax[], const int Ax_size,
                   const I num_seeds,
                         T  x[], const int  x_size,
                         I  w[], const int  w_size,
                         I  z[], const int  z_size)
{
    for(I i = 0; i < num_rows; i++){
        x[i] = std::numeric_limits<T>::max();
        w[i] = -1;
    }
    for(I i = 0; i < num_seeds; i++){
        I seed = z[i];
        assert(seed >= 0 && seed < num_rows);
        x[seed] = 0;
        w[seed] = i;
    }

    std::vector<T> old_distances(num_rows);

    // propagate distances outward
    do{
        std::copy(x, x+num_rows, old_distances.begin());
        bellman_ford(num_rows, Ap, Ap_size, Aj, Aj_size, Ax, Ax_size, x, x_size, w, w_size);
    } while ( !std::equal( x, x+num_rows, old_distances.begin() ) );

    //find boundaries
    for(I i = 0; i < num_rows; i++){
        x[i] = std::numeric_limits<T>::max();
    }
    for(I i = 0; i < num_rows; i++){
        for(I jj = Ap[i]; jj < Ap[i+1]; jj++){
            I j = Aj[jj];
            if( w[i] != w[j] ){
                x[i] = 0;
                break;
            }
        }
    }

    // propagate distances inward
    do{
        std::copy(x, x+num_rows, old_distances.begin());
        bellman_ford(num_rows, Ap, Ap_size, Aj, Aj_size, Ax, Ax_size, x, x_size, w, w_size);
    } while ( !std::equal( x, x+num_rows, old_distances.begin() ) );


    // compute new seeds
    for(I i = 0; i < num_rows; i++){
        const I seed = w[i];

        if (seed == -1) //node belongs to no cluster
            continue;

        assert(seed >= 0 && seed < num_seeds);

        if( x[z[seed]] < x[i] )
            z[seed] = i;
    }
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
 *  Compute a distance-k maximal independent set for a graph stored
 *  in CSR format using a parallel algorithm.  An MIS-k is a set of
 *  vertices such that all vertices in the MIS-k are separated by a
 *  path of at least K+1 edges and no additional vertex can be added
 *  to the set without destroying this property.  A standard MIS
 *  is therefore a MIS-1.
 *
 *  Parameters
 *      num_rows   - number of rows in A (number of vertices)
 *      Ap[]       - CSR row pointer
 *      Aj[]       - CSR index array
 *      k          - minimum separation between MIS vertices
 *      x[]        - state of each vertex (1 if in the MIS, 0 otherwise)
 *      y[]        - random values used during parallel MIS algorithm
 *      max_iters  - maximum number of iterations to use (default, no limit)
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
 *  Compute a breadth first search of a graph in CSR format
 *  beginning at a given seed vertex.
 *
 *  Parameters
 *      num_rows         - number of rows in A (number of vertices)
 *      Ap[]             - CSR row pointer
 *      Aj[]             - CSR index array
 *      order[num_rows]  - records the order in which vertices were searched
 *      level[num_rows]  - records the level set of the searched vertices (i.e. the minimum distance to the seed)
 *
 *  Notes:
 *      The values of the level must be initialized to -1
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
 *  Compute the connected components of a graph stored in CSR format.
 *
 *  Vertices belonging to each component are marked with a unique integer
 *  in the range [0,K), where K is the number of components.
 *
 *  Parameters
 *      num_rows             - number of rows in A (number of vertices)
 *      Ap[]                 - CSR row pointer
 *      Aj[]                 - CSR index array
 *      components[num_rows] - component labels
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
