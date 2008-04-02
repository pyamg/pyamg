#ifndef SMOOTHED_AGGREGATION_H
#define SMOOTHED_AGGREGATION_H

#include <iostream>
#include <vector>
#include <iterator>
#include <algorithm>

#include <assert.h>
#include <cmath>

#define DEBUG

template<class I, class T>
void symmetric_strength_of_connection(const I n_row, 
                                      const T epsilon,
                                      const I Ap[], const I Aj[], const T Ax[],
                                            I Sp[],       I Sj[],       T Sx[])
{
    //Sp,Sj form a CSR representation where the i-th row contains
    //the indices of all the strong connections from node i
    std::vector<T> diags(n_row);

    //compute diagonal values
    for(I i = 0; i < n_row; i++){
        T diag = 0;
        for(I jj = Ap[i]; jj < Ap[i+1]; jj++){
            if(Aj[jj] == i){
                diag += Ax[jj]; //gracefully handle duplicates
            }
        }    
        diags[i] = std::abs(diag);
    }

    I nnz = 0;
    Sp[0] = 0;

    for(I i = 0; i < n_row; i++){

        T eps_Aii = epsilon*epsilon*diags[i];

        for(I jj = Ap[i]; jj < Ap[i+1]; jj++){
            const I   j = Aj[jj];
            const T Aij = Ax[jj];

            if(i == j){continue;}  //skip diagonal

            //  |A(i,j)| >= epsilon * sqrt(|A(i,i)|*|A(j,j)|) 
            if(Aij*Aij >= eps_Aii * diags[j]){    
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
 * Parameters:
 *   n_row         - # of rows in A
 *   Ap[n_row + 1] - CSR row pointer
 *   Aj[nnz]       - CSR column indices
 *    x[n_row]     - aggregate numbers for each node
 *
 * Returns:
 *  The number of aggregates (== max(x[:]) + 1 )
 *
 * Notes:
 *    It is assumed that A is symmetric.
 *    A may contain diagonal entries (self loops)
 *    Unaggregated nodes are marked with a -1
 *    
 */
    
template <class I>
I standard_aggregation(const I n_row,
                       const I Ap[], 
                       const I Aj[],
                             I  x[])
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






//template<class T>
//void sa_smoother(const int n_row,
//		 const T   omega,
//		 const int Ap[], const int Aj[], const T Ax[],
//		 const int Sp[], const int Sj[], const T Sx[],
//		 std::vector<int> * Bp, std::vector<int> * Bj, std::vector<T> * Bx){
//
//
//  //compute filtered diagonal
//  std::vector<T> diags(n_row,0);
//  
//  for(int i = 0; i < n_row; i++){
//    int row_start = Ap[i];
//    int row_end   = Ap[i+1];
//    for(int jj = row_start; jj < row_end; jj++){
//      diags[i] += Ax[jj];
//    }    
//  }
//  for(int i = 0; i < n_row; i++){
//    int row_start = Sp[i];
//    int row_end   = Sp[i+1];
//    for(int jj = row_start; jj < row_end; jj++){
//      diags[i] -= Sx[jj];
//    }    
//  }
//  
//#ifdef DEBUG
//  for(int i = 0; i < n_row; i++){ assert(diags[i] > 0); }
//#endif
//
//
//  //compute omega Jacobi smoother
//  Bp->push_back(0);
//  for(int i = 0; i < n_row; i++){
//    int row_start = Sp[i];
//    int row_end   = Sp[i+1];
//    const T row_scale = -omega/diags[i];
//
//    Bx->push_back(1.0);
//    Bj->push_back( i );
//    
//    for(int jj = row_start; jj < row_end; jj++){
//      Bx->push_back(row_scale*Sx[jj]);
//      Bj->push_back(Sj[jj]);
//    }    
//    Bp->push_back(Bj->size());
//  }
//}



#endif
