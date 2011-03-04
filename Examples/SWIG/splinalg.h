#ifndef SPLINALG_H
#define SPLINALG_H

#include <math.h>

/*
 *  Perform a CSR lower triangular forward solve A x = b
 *
 *  Indices need not be sorted
 *  Does not check for zero diagonal.
 *
 *  Parameters
 *      Ap[]       - CSR row pointer
 *      Aj[]       - CSR index array
 *      Ax[]       - CSR data array
 *      x[]        - approximate solution
 *      b[]        - right hand side
 *      n          - matrix size
 *  
 *  Returns:
 *      Nothing, x will be modified in place
 *
 */
template<class I, class T>
void forwardsolve(const I Ap[], 
                  const I Aj[], 
                  const T Ax[],
                        T  x[],
                  const T  b[],
                  const I n)
{
    T tmp, diag;

    x[0] = b[0]/Ax[0];
    for(I i=1; i<n; i++){
      diag = 0.0;
      tmp = b[i];
      for(I j=Ap[i]; j<Ap[i+1]; j++){
        if(i==Aj[j]){
          diag = Ax[j];
        }
        else{
          tmp -= Ax[j]*x[Aj[j]]; 
        }
      }
      x[i] = tmp/diag;
  }
}

/*
 *  Perform a CSR upper triangular backward solve A x = b
 *
 *  Indices need not be sorted
 *  Does not check for zero diagonal.
 *
 *  Parameters
 *      Ap[]       - CSR row pointer
 *      Aj[]       - CSR index array
 *      Ax[]       - CSR data array
 *      x[]        - approximate solution
 *      b[]        - right hand side
 *      n          - matrix size
 *  
 *  Returns:
 *      Nothing, x will be modified in place
 *
 */
template<class I, class T>
void backsolve(const I Ap[], 
                  const I Aj[], 
                  const T Ax[],
                        T  x[],
                  const T  b[],
                  const I n)
{
    T tmp, diag;

    x[n-1] = b[n-1]/Ax[n-1];
    for(I i=n-1; i>=0; i--){
      diag = 0.0;
      tmp = b[i];
      for(I j=Ap[i]; j<Ap[i+1]; j++){
        if(i==Aj[j]){
          diag = Ax[j];
        }
        else{
          tmp -= Ax[j]*x[Aj[j]];
        }
      }
      x[i] = tmp/diag;
    }
}

#endif
