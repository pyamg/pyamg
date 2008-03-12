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
void sa_strong_connections(const I n_row, 
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


    
    
template <class I>
I sa_get_aggregates(const I n_row,
                    const I Ap[], const I Aj[],
                          I Bj[])
{
    // Bj[n] == -1 means i-th node has not been aggregated
    std::fill(Bj, Bj+n_row, -1);

    std::vector<bool> aggregated(n_row, false);

    I num_aggregates = 0;

    //Pass #1
    for(I i = 0; i < n_row; i++){
        if(aggregated[i]){ continue; } //already marked

        const I row_start = Ap[i];
        const I row_end   = Ap[i+1];

        //Determine whether all neighbors of this node are free (not already aggregates)
        bool free_neighborhood = true;
        for(I jj = row_start; jj < row_end; jj++){
            if( aggregated[Aj[jj]] ){
                free_neighborhood = false;
                break;
            }
        }    

        if(!free_neighborhood){ continue; } //bail out

        //Make an aggregate out of this node and its strong neigbors
        Bj[i]         = num_aggregates;
        aggregated[i] = true;
        for(I jj = row_start; jj < row_end; jj++){
            const I j = Aj[jj];
            Bj[j]         = num_aggregates;
            aggregated[j] = true;
        }
        num_aggregates++;
    }


    //Pass #2
    for(I i = 0; i < n_row; i++){
        if(aggregated[i]){ continue; } //already marked

        for(I jj = Ap[i]; jj < Ap[i+1]; jj++){
            const I j = Aj[jj];
        
            if(aggregated[j]){
                Bj[i] = Bj[j];
                break;
            }

        }   
    }

    // now Bj[n] == -1 means i-th node has not been aggregated

    //Pass #3
    for(I i = 0; i < n_row; i++){
        if(Bj[i] != -1){ continue; } //already marked

        const I row_start = Ap[i];
        const I row_end   = Ap[i+1];

        Bj[i] = num_aggregates;

        for(I jj = row_start; jj < row_end; jj++){
            const I j = Aj[jj];

            if(Bj[j] != -1){ //unmarked neighbors
                Bj[j] = num_aggregates;
            }
        }  
        num_aggregates++;
    }

#ifdef DEBUG
    for(I i = 0; i < n_row; i++){ assert(Bj[i] >= 0 && Bj[i] < num_aggregates); }
#endif

    return num_aggregates;
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
