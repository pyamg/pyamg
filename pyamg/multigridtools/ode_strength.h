#ifndef ODE_STRENGTH_H
#define ODE_STRENGTH_H

#include <iostream>
#include <vector>
#include <iterator>
#include <assert.h>

/*
 *
 * Return a filtered strength-of-connection matrix by applying a drop tolerance
 *  Strength values are assumed to be "distance"-like, i.e. the smaller the 
 *  value the stronger the connection
 *
 *    An off-diagonal entry A[i,j] is a strong connection iff
 *
 *            S[i,j] <= epsilon * min( S[i,k] )   where k != i
 *  
 *   Also, set the diagonal to 1.0, as each node is perfectly close to itself
 */          

template<class I, class T>
void apply_distance_filter(const I n_row,
                           const T epsilon,
                           const I Sp[],    const I Sj[], T Sx[])
{
    //Loop over rows
    for(I i = 0; i < n_row; i++)
    {
        T min_offdiagonal = std::numeric_limits<T>::max();

        I row_start = Sp[i];
        I row_end   = Sp[i+1];
    
        //Find min for row i
        for(I jj = row_start; jj < row_end; jj++){
            if(Sj[jj] != i){
                min_offdiagonal = std::min(min_offdiagonal,Sx[jj]);
            }
        }

        //Apply drop tol to row i
        T threshold = epsilon*min_offdiagonal;
        for(I jj = row_start; jj < row_end; jj++){
            if(Sx[jj] >= threshold){
                if(Sj[jj] != i){
                    Sx[jj] = 0.0;
                }
            }
            //Set diagonal to 1.0
            if(Sj[jj] == i){
                    Sx[jj] = 1.0;
            }
        }
    }
}


#endif

