#ifndef GRAPH_H
#define GRAPH_H

#include <algorithm>

/*
 *  Compute a maximal independent set for a
 *  graph stored in CSR format
 *
 *  Returns the N, the number of nodes in the MIS.
 *
 *  If x[i] != 0 on input then the i-th vertex is ignored and
 *  effectively removed from the graph.
 *
 *  In the output x[i] = K if the i-th node is a member of the MIS.
 *  Otherwise the value is unchanged.
 *
 *
 */
template<class I, class T>
I maximal_independent_set(const I num_rows,
                          const I Ap[], 
                          const I Aj[], 
                          const T  K,
                                T  x[])
{
    I N = 0;
    
    for(I i = 0; i < num_rows; i++){
        if(x[i]) continue;

        const I row_start = Ap[i];
        const I row_end   = Ap[i+1];

        I jj;
        for(jj = row_start; jj < row_end; jj++){
            if(x[Aj[jj]] == K) break;
        }

        if(jj == row_end){
            N++;
            x[i] = K;
        }
    }

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
    std::fill( x, x + num_rows, 0);

    I N = 0;
    T K = 0;

    while(N < num_rows){
        N += maximal_independent_set(num_rows,Ap,Aj,K+1,x);
        K++;
    }

    for(I i = 0; i < num_rows; i++)
        x[i]--;

    return K;
}


#endif

