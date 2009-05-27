#ifndef RELAXATION_H
#define RELAXATION_H

#include "linalg.h"

/*
 *  Perform one iteration of Gauss-Seidel relaxation on the linear
 *  system Ax = b, where A is stored in CSR format and x and b
 *  are column vectors.
 *
 *  The unknowns are swept through according to the slice defined
 *  by row_start, row_end, and row_step.  These options are used
 *  to implement standard forward and backward sweeps, or sweeping
 *  only a subset of the unknowns.  A forward sweep is implemented
 *  with gauss_seidel(Ap, Aj, Ax, x, b, 0, N, 1) where N is the 
 *  number of rows in matrix A.  Similarly, a backward sweep is 
 *  implemented with gauss_seidel(Ap, Aj, Ax, x, b, N, -1, -1).
 *
 *  Parameters
 *      Ap[]       - CSR row pointer
 *      Aj[]       - CSR index array
 *      Ax[]       - CSR data array
 *      x[]        - approximate solution
 *      b[]        - right hand side
 *      row_start  - beginning of the sweep
 *      row_stop   - end of the sweep (i.e. one past the last unknown)
 *      row_step   - stride used during the sweep (may be negative)
 *  
 *  Returns:
 *      Nothing, x will be modified in place
 *
 */
template<class I, class T, class F>
void gauss_seidel(const I Ap[], 
                  const I Aj[], 
                  const T Ax[],
                        T  x[],
                  const T  b[],
                  const I row_start,
                  const I row_stop,
                  const I row_step)
{
    for(I i = row_start; i != row_stop; i += row_step) {
        I start = Ap[i];
        I end   = Ap[i+1];
        T rsum = 0;
        T diag = 0;
        
        for(I jj = start; jj < end; jj++){
            I j = Aj[jj];
            if (i == j)
                diag  = Ax[jj];
            else
                rsum += Ax[jj]*x[j];
        }
        
        //TODO raise error? inform user?
        if (diag != 0){
            x[i] = (b[i] - rsum)/diag;
        }
    }
}


/*
 *  Perform one iteration of Gauss-Seidel relaxation on the linear
 *  system Ax = b, where A is stored in Block CSR format and x and b
 *  are column vectors.  This method applies pointwise relaxation
 *  to the BSR as opposed to "block relaxation".
 *
 *  Refer to gauss_seidel for additional information regarding
 *  row_start, row_stop, and row_step.
 *
 *  Parameters
 *      Ap[]       - BSR row pointer
 *      Aj[]       - BSR index array
 *      Ax[]       - BSR data array
 *      x[]        - approximate solution
 *      b[]        - right hand side
 *      row_start  - beginning of the sweep (block row index)
 *      row_stop   - end of the sweep (i.e. one past the last unknown)
 *      row_step   - stride used during the sweep (may be negative)
 *      blocksize  - BSR blocksize (blocks must be square)
 *  
 *  Returns:
 *      Nothing, x will be modified in place
 *
 */
template<class I, class T, class F>
void block_gauss_seidel(const I Ap[], 
                        const I Aj[], 
                        const T Ax[],
                              T  x[],
                        const T  b[],
                        const I row_start,
                        const I row_stop,
                        const I row_step,
                        const I blocksize)
{
    const I B2 = blocksize * blocksize;
    
    for(I i = row_start; i != row_stop; i += row_step) {
        I start = Ap[i];
        I end   = Ap[i+1];

        for(I bi = 0; bi < blocksize; bi++){
            T rsum = 0;
            T diag = 0;

            for(I jj = start; jj < end; jj++){
                I j = Aj[jj];
                const T * block_row = Ax + B2*jj + blocksize*bi;
                const T * block_x   = x + blocksize * j;

                if (i == j){
                    //diagonal block
                    diag = block_row[bi];
                    for(I bj = 0; bj < bi; bj++){
                        rsum += block_row[bj] * block_x[bj];
                    }
                    for(I bj = bi+1; bj < blocksize; bj++){
                        rsum += block_row[bj] * block_x[bj];
                    }
                } else {
                    for(I bj = 0; bj < blocksize; bj++){
                        rsum += block_row[bj] * block_x[bj];
                    }
                }
            }

            //TODO raise error? inform user?
            if (diag != 0){
                x[blocksize*i + bi] = (b[blocksize*i + bi] - rsum)/diag;
            }
        }
    }
}


/*
 *  Perform one iteration of Jacobi relaxation on the linear
 *  system Ax = b, where A is stored in CSR format and x and b
 *  are column vectors.  Damping is controlled by the omega
 *  parameter.
 *
 *  Refer to gauss_seidel for additional information regarding
 *  row_start, row_stop, and row_step.
 *
 *  Parameters
 *      Ap[]       - CSR row pointer
 *      Aj[]       - CSR index array
 *      Ax[]       - CSR data array
 *      x[]        - approximate solution
 *      b[]        - right hand side
 *      temp[]     - temporary vector the same size as x
 *      row_start  - beginning of the sweep
 *      row_stop   - end of the sweep (i.e. one past the last unknown)
 *      row_step   - stride used during the sweep (may be negative)
 *      omega      - damping parameter
 *  
 *  Returns:
 *      Nothing, x will be modified in place
 *
 */
template<class I, class T, class F>
void jacobi(const I Ap[], 
            const I Aj[], 
            const T Ax[],
                  T  x[],
            const T  b[],
                  T temp[],
            const I row_start,
            const I row_stop,
            const I row_step,
            const T omega[])
{
    T one = 1.0;
    T omega2 = omega[0];

    for(I i = row_start; i != row_stop; i += row_step) {
        temp[i] = x[i];
    }
    
    for(I i = row_start; i != row_stop; i += row_step) {
        I start = Ap[i];
        I end   = Ap[i+1];
        T rsum = 0;
        T diag = 0;
        
        for(I jj = start; jj < end; jj++){
            I j = Aj[jj];
            if (i == j)
                diag  = Ax[jj];
            else
                rsum += Ax[jj]*temp[j];
        }
        
        //TODO raise error? inform user?
        if (diag != 0){ 
            x[i] = (one - omega2) * temp[i] + omega2 * ((b[i] - rsum)/diag);
        }
    }
}

/*
 *  Perform one iteration of Gauss-Seidel relaxation on the linear
 *  system Ax = b, where A is stored in CSR format and x and b
 *  are column vectors.
 *
 *  Unlike gauss_seidel, which is restricted to updating a slice
 *  of the unknowns (defined by row_start, row_start, and row_step),
 *  this method updates unknowns according to the rows listed in  
 *  an index array.  This allows and arbitrary set of the unknowns 
 *  to be updated in an arbitrary order, as is necessary for the
 *  relaxation steps in the Compatible Relaxation method.
 *
 *  In this method the slice arguments are used to define the subset
 *  of the index array Id which is to be considered.
 *
 *  Parameters
 *      Ap[]       - CSR row pointer
 *      Aj[]       - CSR index array
 *      Ax[]       - CSR data array
 *      x[]        - approximate solution
 *      b[]        - right hand side
 *      Id[]       - index array representing the 
 *      row_start  - beginning of the sweep (in array Id)
 *      row_stop   - end of the sweep (in array Id)
 *      row_step   - stride used during the sweep (may be negative)
 *  
 *  Returns:
 *      Nothing, x will be modified in place
 *
 */
template<class I, class T>
void gauss_seidel_indexed(const I Ap[], 
                          const I Aj[], 
                          const T Ax[],
                                T  x[],
                          const T  b[],
                          const I Id[],
                          const I row_start,
                          const I row_stop,
                          const I row_step)
{
  for(I i = row_start; i != row_stop; i += row_step) {
    I inew = Id[i];
    I start = Ap[inew];
    I end   = Ap[inew+1];
    T rsum  = 0;
    T diag  = 0;

    for(I jj = start; jj < end; ++jj){
      I j = Aj[jj];
      if (inew == j){
        diag = Ax[jj];
      }
      else{
        rsum += Ax[jj]*x[j];
      }
    }

    if (diag != 0){
      x[inew] = (b[inew] - rsum)/diag;
    }
  }
}

/*
 * Perform NE Jacobi on the linear system A x = b
 * This effectively carries out weighted-Jacobi on A A^T x = A^T b
 * (also known as Cimmino's relaxation)
 *
 * Parameters
 * ----------
 * Ap : {int array}
 *  index pointer for CSR matrix A
 * Aj : {int array}
 *  column indices for CSR matrix A
 * Ax : {array}
 *  value array for CSR matrix A
 * x : {array}
 *  current guess to the linear system
 * b : {array}
 *  right hand side
 * Tx : {array}
 *  scaled residual
 *  D_A^{-1} (b - Ax)
 * temp : {array}
 *  work space
 * row_start,stop,step : {int}
 *  controls which rows to iterate over
 * omega : {array}
 *  size one array that contains the weighted-jacobi 
 *  parameter.  An array must be used to pass in omega to
 *  account for the case where omega may be complex
 *
 * Returns
 * -------
 * x is modified in place in an additive, not overwiting fashion
 *
 * Notes
 * -----
 * Primary calling routine is jacobi_ne in relaxation.py
 */
template<class I, class T, class F>
void jacobi_ne(const I Ap[], 
               const I Aj[], 
               const T Ax[],
                     T  x[],
               const T  b[],
               const T Tx[],
                     T temp[],
               const I row_start,
               const I row_stop,
               const I row_step,
               const T omega[])
{
    //rename
    const T * delta = Tx;
    const T omega2 = omega[0];

    for(I i = row_start; i < row_stop; i+=row_step)
    {   temp[i] = 0.0; }

    for(I i = row_start; i < row_stop; i+=row_step)
    {
        I start = Ap[i];
        I end   = Ap[i+1];
        for(I j = start; j < end; j++)
        {   temp[Aj[j]] += omega2*conjugate(Ax[j])*delta[i]; }
    }

    for(I i = row_start; i < row_stop; i+=row_step)
    {   x[i] += temp[i]; }
}

/*
 * Perform NE Gauss-Seidel on the linear system A x = b
 * This effectively carries out Gauss-Seidel on A A.H x = b
 * 
 * Parameters
 * ----------
 * Ap : {int array}
 *  index pointer for CSR matrix A
 * Aj : {int array}
 *  column indices for CSR matrix A
 * Ax : {array}
 *  value array for CSR matrix A
 * x : {array}
 *  current guess to the linear system
 * b : {array}
 *  right hand side
 * Tx : {array}
 *  inverse(diag(A A.H))
 * omega : {float}
 *  relaxation parameter 
 *  (if not 1.0, then algorithm becomes SOR)
 * row_start,stop,step : {int}
 *  controls which rows to iterate over
 *
 * Returns
 * -------
 * x is modified in place in an additive, not overwiting fashion
 *
 * Notes
 * -----
 * Primary calling routine is gass_seidel_ne in relaxation.py
 */
template<class I, class T, class F>
void gauss_seidel_ne(const I Ap[], 
                     const I Aj[], 
                     const T Ax[],
                           T  x[],
                     const T  b[],
                     const I row_start,
                     const I row_stop,
                     const I row_step,
                     const T Tx[],
                     const F omega)
{
    //rename
    const T * D_inv = Tx;
    
    for(I i = row_start; i != row_stop; i+=row_step)
    {
        I start = Ap[i];
        I end   = Ap[i+1];
        
        //First calculate "delta", the scaled residual term
        T delta = 0.0;
        for(I j = start; j < end; j++)
        {   delta += Ax[j]*x[Aj[j]]; }
        delta = (b[i] - delta)*D_inv[i]*omega;

        for(I j = start; j < end; j++)
        {   x[Aj[j]] += conjugate(Ax[j])*delta; }
    }

}


/*
 * Perform NR Gauss-Seidel on the linear system A x = b
 * This effectively carries out Gauss-Seidel on A.H A x = A.H b
 * 
 * Parameters
 * ----------
 * Ap : {int array}
 *  index pointer for CSC matrix A
 * Aj : {int array}
 *  row indices for CSC matrix A
 * Ax : {array}
 *  value array for CSC matrix A
 * x : {array}
 *  current guess to the linear system
 * z : {array}
 *  initial residual
 * Tx : {array}
 *  inverse(diag(A.H A))
 * omega : {float}
 *  relaxation parameter 
 *  (if not 1.0, then algorithm becomes SOR)
 * col_start,stop,step : {int}
 *  controls which rows to iterate over
 *
 * Returns
 * -------
 * x is modified in place in an additive, not overwiting fashion
 *
 * Notes
 * -----
 * Primary calling routine is nr_gass_seidel in relaxation.py
 */
template<class I, class T, class F>
void gauss_seidel_nr(const I Ap[], 
                     const I Aj[], 
                     const T Ax[],
                           T  x[],
                           T  z[],
                     const I col_start,
                     const I col_stop,
                     const I col_step,
                     const T Tx[],
                     const F omega)
{
    //rename
    const T * D_inv = Tx;
    T * r = z;
    
    for(I i = col_start; i != col_stop; i+=col_step)
    {
        I start = Ap[i];
        I end   = Ap[i+1];
        
        // delta = < A e_i, r > 
        T delta = 0.0;
        for(I j = start; j < end; j++)
        {   delta += conjugate(Ax[j])*r[Aj[j]]; }
        
        // delta /=  omega*(A.H A)_{ii}
        delta *= (D_inv[i]*omega);

        // update entry in x forcing < A.H b - A.H A x, e_i > = 0
        x[i] += delta;
        
        // r -= delta A e_i
        for(I j = start; j < end; j++)
        {   r[Aj[j]] -= delta*Ax[j]; }
    }

}

/*
 *  Perform one iteration of block Jacobi relaxation on the linear
 *  system Ax = b, where A is stored in BSR format and x and b
 *  are column vectors.  Damping is controlled by the omega
 *  parameter.
 *
 *  Refer to gauss_seidel for additional information regarding
 *  row_start, row_stop, and row_step.
 *
 *  Parameters
 *      Ap[]       - BSR row pointer
 *      Aj[]       - BSR index array
 *      Ax[]       - BSR data array, blocks assumed square
 *      x[]        - approximate solution
 *      b[]        - right hand side
 *      Tx[]       - Inverse of each diagonal block of A stored
 *                   as a (n/blocksize, blocksize, blocksize) array
 *      temp[]     - temporary vector the same size as x
 *      row_start  - beginning of the sweep
 *      row_stop   - end of the sweep (i.e. one past the last unknown)
 *      row_step   - stride used during the sweep (may be negative)
 *      omega      - damping parameter
 *      blocksize  - dimension of sqare blocks in BSR matrix A
 *  
 *  Returns:
 *      Nothing, x will be modified in place
 *
 */
template<class I, class T, class F>
void block_jacobi(const I Ap[], 
                  const I Aj[], 
                  const T Ax[],
                        T  x[],
                  const T  b[],
                  const T Tx[],
                        T temp[],
                  const I row_start,
                  const I row_stop,
                  const I row_step,
                  const T omega[],
                  const I blocksize)
{
    // Rename
    const T * Dinv = Tx;

    T one = 1.0;
    T zero = 0.0;
    T omega2 = omega[0];
    T *rsum = new T[blocksize];
    T *v = new T[blocksize];
    I blocksize_sq = blocksize*blocksize;

    // Copy x to temp vector
    for(I i = row_start*blocksize; i != row_stop*blocksize; i += row_step*blocksize) {
        std::copy(&(x[i]), &(x[i+blocksize]), &(temp[i]));
    }
    
    // Begin block Jacobi sweep
    for(I i = row_start; i != row_stop; i += row_step) {
        I start = Ap[i];
        I end   = Ap[i+1];
        std::fill(&(rsum[0]), &(rsum[blocksize]), zero);
        
        // Carry out a block dot product between block row i and x
        for(I jj = start; jj < end; jj++){
            I j = Aj[jj];
            if (i == j)
                //diagonal, do nothing
                continue;
            else
                gemm(&(Ax[jj*blocksize_sq]), blocksize, blocksize, 'F', 
                     &(temp[j*blocksize]),   blocksize, 1,         'F', 
                     &(v[0]),                blocksize, 1,         'F');
                for(I k = 0; k < blocksize; k++) {
                    rsum[k] += v[k]; }
        }
        
        // x[i*blocksize:(i+1)*blocksize] = (one - omega2) * temp[i*blocksize:(i+1)*blocksize] + omega2 * 
        //          (Dinv[i*blocksize_sq : (i+1)*blocksize_sq]*(b[i*blocksize:(i+1)*blocksize] - rsum[0:blocksize]));
        I iblocksize = i*blocksize;
        for(I k = 0; k < blocksize; k++) {
            rsum[k] = b[iblocksize + k] - rsum[k]; }
        
        gemm(&(Dinv[i*blocksize_sq]), blocksize, blocksize, 'F', 
             &(rsum[0]),              blocksize, 1,         'F', 
             &(v[0]),                 blocksize, 1,         'F');

        for(I k = 0; k < blocksize; k++) {
            x[iblocksize + k] = (one - omega2)*temp[iblocksize + k] + omega2*v[k]; }
    }

    delete[] v;
    delete[] rsum;
}

#endif
