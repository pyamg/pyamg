#ifndef GRAPH_H
#define GRAPH_H

#include <iostream>
#include <algorithm>

/*
 *  Compute a maximal independent set for a graph stored in CSR format
 *  using a greedy serial algorithm
 *
 *  Parameters
 *      num_rows   - number of rows in A (number of vertices)
 *      Ap[]       - CSR row pointer
 *      Aj[]       - CSR index array
 *      active     - value used for active vertices        (input)
 *       C         - value used to mark non-MIS vertices   (ouput)
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
                                 const I Ap[], 
                                 const I Aj[], 
                                 const T active,
                                 const T  C,
                                 const T  F,
                                       T  x[])
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
 *       C         - value used to mark non-MIS vertices   (ouput)
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
                                   const I Ap[], 
                                   const I Aj[],
                                   const T active,
                                   const T  C,
                                   const T  F,
                                         T  x[],
                                   const R  y[],
                                   const I  max_iters=-1)
{
    I N = 0;
    I num_iters = 0;

    bool work = true;

    while(work && (max_iters == -1 || num_iters < max_iters)){
        work = false;

        num_iters++;
        
        for(I i = 0; i < num_rows; i++){
            const R yi = y[i];

            if(x[i] != active) continue;
            
            work = true;

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
                
                if(x[j] == active){
                    const R yj = y[j];
                    if(yj > yi)
                        break;                     //neighbor is larger 
                    else if (yj == yi && j > i)
                        break;                     //tie breaker goes to neighbor
                }
            }
   
            if(jj == row_end){
                N++;
                x[i] = C;
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
 *  of vertices from the graph.
 *
 *  Returns the K, the number of colors used in the coloring.
 *  On return x[i] \in [0,1, ..., K - 1] will contain the color
 *  of the i-th vertex.
 *
 */
template<class I, class T>
T vertex_coloring_mis(const I num_rows,
                      const I Ap[], 
                      const I Aj[], 
                            T  x[])
{
    std::fill( x, x + num_rows, -1);

    I N = 0;
    T K = 0;

    while(N < num_rows){
        N += maximal_independent_set_serial(num_rows,Ap,Aj,-1-K,K,-2-K,x);
        K++;
    }

    return K;
}


    
template<class I, class T>
void bellman_ford(const I num_rows,
                  const I Ap[], 
                  const I Aj[], 
                  const T Ax[],
                        T  x[],
                        I  y[])
{
    for(I i = 0; i < num_rows; i++){
        T xi = x[i];
        I yi = y[i];
        for(I jj = Ap[i]; jj < Ap[i+1]; jj++){
            const I j = Aj[jj];
            const T d = Ax[jj] + x[j];
            if(d < xi){
                xi = d;
                yi = y[j];
            }
        }
        x[i] = xi;
        y[i] = yi;
    }
}

// x[num_rows]     - distance to nearest seed
// y[num_rows]     - cluster membership
// z[num_centers]  - cluster centers
template<class I, class T>
void lloyd_cluster(const I num_rows,
                   const I Ap[], 
                   const I Aj[], 
                   const T Ax[],
                   const I num_seeds,
                         T  x[],
                         I  y[],
                         I  z[])
{
    for(I i = 0; i < num_rows; i++){
        x[i] = std::numeric_limits<T>::max();
        y[i] = -1;
    }
    for(I i = 0; i < num_seeds; i++){
        I seed = z[i];
        x[seed] = 0;
        y[seed] = i;
    }

    std::vector<T> old_distances(num_rows);

    // propagate distances outward
    do{
        std::copy(x, x+num_rows, old_distances.begin());
        bellman_ford(num_rows, Ap, Aj, Ax, x, y);
    } while ( !std::equal( x, x+num_rows, old_distances.begin() ) );

    //find boundaries
    for(I i = 0; i < num_rows; i++){
        x[i] = std::numeric_limits<T>::max();
    }
    for(I i = 0; i < num_rows; i++){
        for(I jj = Ap[i]; jj < Ap[i+1]; jj++){
            I j = Aj[jj];
            if( y[i] != y[j] ){
                x[i] = 0;
                break;
            }
        }
    }

    // propagate distances inward
    do{
        std::copy(x, x+num_rows, old_distances.begin());
        bellman_ford(num_rows, Ap, Aj, Ax, x, y);
    } while ( !std::equal( x, x+num_rows, old_distances.begin() ) );


    // compute new centers
    for(I i = 0; i < num_rows; i++){
        const I cluster = y[i];

        if (cluster == -1) //node belongs to no cluster
            continue;
        
        assert(cluster > 0 && cluster < num_seeds);
        if( x[z[cluster]] < x[i] )
            z[cluster] = i;
    }
}

#endif

