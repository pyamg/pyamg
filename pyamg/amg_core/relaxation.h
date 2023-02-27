#ifndef RELAXATION_H
#define RELAXATION_H

#include "linalg.h"

/*
 * Perform one iteration of Gauss-Seidel relaxation on the linear
 * system Ax = b, where A is stored in CSR format and x and b
 * are column vectors.
 *
 * Parameters
 * ----------
 * Ap : array
 *     CSR row pointer
 * Aj : array
 *     CSR index array
 * Ax : array
 *     CSR data array
 * x : array, inplace
 *     approximate solution
 * b : array
 *     right hand side
 * row_start : int
 *     beginning of the sweep
 * row_stop : int
 *     end of the sweep (i.e. one past the last unknown)
 * row_step : int
 *     stride used during the sweep (may be negative)
 *
 * Returns
 * -------
 * Nothing, x will be modified inplace
 *
 * Notes
 * -----
 * The unknowns are swept through according to the slice defined
 * by row_start, row_end, and row_step.  These options are used
 * to implement standard forward and backward sweeps, or sweeping
 * only a subset of the unknowns.  A forward sweep is implemented
 * with gauss_seidel(Ap, Aj, Ax, x, b, 0, N, 1) where N is the
 * number of rows in matrix A.  Similarly, a backward sweep is
 * implemented with gauss_seidel(Ap, Aj, Ax, x, b, N, -1, -1).
 */
template<class I, class T, class F>
void gauss_seidel(const I Ap[], const int Ap_size,
                  const I Aj[], const int Aj_size,
                  const T Ax[], const int Ax_size,
                        T  x[], const int  x_size,
                  const T  b[], const int  b_size,
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

        if (diag != (F) 0.0){
            x[i] = (b[i] - rsum)/diag;
        }
    }
}


/*
 * Perform one iteration of Gauss-Seidel relaxation on the linear
 * system Ax = b, where A is stored in Block CSR format and x and b
 * are column vectors.  This method applies point-wise relaxation
 * to the BSR as opposed to \"block relaxation\".
 *
 * Refer to gauss_seidel for additional information regarding
 * row_start, row_stop, and row_step.
 *
 * Parameters
 * ----------
 * Ap : array
 *     BSR row pointer
 * Aj : array
 *     BSR index array
 * Ax : array
 *     BSR data array
 * x : array, inplace
 *     approximate solution
 * b : array
 *     right hand side
 * row_start : int
 *     beginning of the sweep (block row index)
 * row_stop : int
 *     end of the sweep (i.e. one past the last unknown)
 * row_step : int
 *     stride used during the sweep (may be negative)
 * blocksize : int
 *     BSR blocksize (blocks must be square)
 *
 * Returns
 * -------
 * Nothing, x will be modified inplace
 *
 */
template<class I, class T, class F>
void bsr_gauss_seidel(const I Ap[], const int Ap_size,
                      const I Aj[], const int Aj_size,
                      const T Ax[], const int Ax_size,
                            T  x[], const int  x_size,
                      const T  b[], const int  b_size,
                      const I row_start,
                      const I row_stop,
                      const I row_step,
                      const I blocksize)
{
    I B2 = blocksize*blocksize;
    T *rsum = new T[blocksize];
    T *Axloc = new T[blocksize];
    //T zero = 0.0;

    // Determine if this is a forward, or backward sweep
    I step, step_start, step_end;
    if (row_step < 0){
        step = -1;
        step_start = blocksize-1;
        step_end = -1;
    }
    else{
        step = 1;
        step_start = 0;
        step_end = blocksize;
    }

    for(I i = row_start; i != row_stop; i += row_step) {
        I start = Ap[i];
        I end   = Ap[i+1];
        I diag_ptr = -1;


        // initialize rsum to b, then later subtract A*x
        for(I k = 0; k < blocksize; k++) {
            rsum[k] = b[i*blocksize+k]; }

        // loop over row i
        for(I jj = start; jj < end; jj++){
            // extract column entry
            I j = Aj[jj];
            // absolute column entry for the start of this block
            I col = j*blocksize;

            if (i == j){    //point to where in Ax the diagonal block starts
                diag_ptr = jj*B2; }
            else {
                // do a dense multiply of this block times x and accumulate in rsum
                gemm(&(Ax[jj*B2]),  blocksize, blocksize, 'F',
                     &(x[col]),     blocksize,   1,       'F',
                     &(Axloc[0]),   blocksize,   1,       'F',
                     'T');
                for(I m = 0; m < blocksize; m++) {
                    rsum[m] -= Axloc[m]; }
            }
        }

        // Carry out point-wise GS over the diagonal block,
        // all the other blocks have been factored into rsum.
        if (diag_ptr != -1) {
            for(I k = step_start; k != step_end; k+=step){
                T diag = 1.0;
                for(I kk = step_start; kk != step_end; kk+=step){
                    if(k == kk){
                        // diagonal entry
                        diag = Ax[k*blocksize + kk + diag_ptr]; }
                    else{
                        // off-diag entry
                        rsum[k] -= Ax[k*blocksize + kk + diag_ptr]*x[i*blocksize+kk]; }
                }
                if (diag != (F) 0.0){
                    x[i*blocksize+k] = rsum[k]/diag; }
            }
        }

    } // end outer-most for loop

    delete[] rsum;
    delete[] Axloc;
}// end function


/*
 * Perform one iteration of Jacobi relaxation on the linear
 * system Ax = b, where A is stored in CSR format and x and b
 * are column vectors.  Damping is controlled by the omega
 * parameter.
 *
 * Refer to gauss_seidel for additional information regarding
 * row_start, row_stop, and row_step.
 *
 * Parameters
 * ----------
 * Ap : array
 *     CSR row pointer
 * Aj : array
 *     CSR index array
 * Ax : array
 *     CSR data array
 * x : array, inplace
 *     approximate solution
 * b : array
 *     right hand side
 * temp, array
 *     temporary vector the same size as x
 * row_start : int
 *     beginning of the sweep
 * row_stop : int
 *     end of the sweep (i.e. one past the last unknown)
 * row_step : int
 *     stride used during the sweep (may be negative)
 * omega : float
 *     damping parameter
 *
 * Returns
 * -------
 * Nothing, x will be modified inplace
 *
 */
template<class I, class T, class F>
void jacobi(const I Ap[], const int Ap_size,
            const I Aj[], const int Aj_size,
            const T Ax[], const int Ax_size,
                  T  x[], const int  x_size,
            const T  b[], const int  b_size,
                  T temp[], const int temp_size,
            const I row_start,
            const I row_stop,
            const I row_step,
            const T omega[], const int omega_size)
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

        if (diag != (F) 0.0){
            x[i] = (one - omega2) * temp[i] + omega2 * ((b[i] - rsum)/diag);
        }
    }
}


/*
 *  Perform one iteration of Jacobi relaxation on the linear
 *  system Ax = b for a given set of row indices, where A is
 *  stored in CSR format and x and b are column vectors.
 *  Damping is controlled by the omega parameter.
 *
 *  Parameters
 *      Ap[]       - CSR row pointer
 *      Aj[]       - CSR index array
 *      Ax[]       - CSR data array
 *      x[]        - approximate solution
 *      b[]        - right hand side
 *      temp[]     - temporary vector the same size as x
 *      indices[]  - list of row indices to perform Jacobi on, e.g. F-points
 *      omega      - damping parameter
 *
 *  Returns:
 *      Nothing, x will be modified in place
 *
 */
template<class I, class T, class F>
void jacobi_indexed(const I Ap[], const int Ap_size,
                    const I Aj[], const int Aj_size,
                    const T Ax[], const int Ax_size,
                          T  x[], const int  x_size,
                    const T  b[], const int  b_size,
                    const I indices[], const int indices_size,
                    const T omega[], const int omega_size)
{
    T one = 1.0;
    T omega2 = omega[0];
    std::vector<T> temp(x_size);

    // Set temp = x
    for(I i=0; i<x_size; i++) {
        temp[i] = x[i];
    }

    // Perform Jacobi on each row in indices[]
    for(I i=0; i<indices_size; i++) {
        I row = indices[i];
        I start = Ap[row];
        I end   = Ap[row+1];
        T rsum = 0;
        T diag = 0;

        // Compute A*x for this row, get diagonal element of A
        for(I jj = start; jj < end; jj++){
            I col = Aj[jj];
            if (row == col) {
                diag  = Ax[jj];
            }
            else {
                rsum += Ax[jj]*temp[col];
            }
        }

        // Check for nonzero diagonal and update ith index
        if (diag != (F) 0.0){
            x[row] = (one - omega2) * temp[row] + omega2 * ((b[row] - rsum)/diag);
        }
        else {
            std::cout << "Warning : zero diagonal encountered in Jacobi; ignored.\n";
        }
    }
}


/*
 * Perform one iteration of Jacobi relaxation on the linear
 * system Ax = b, where A is stored in Block CSR format and x and b
 * are column vectors.  This method applies point-wise relaxation
 * to the BSR as opposed to \"block relaxation\".
 *
 * Refer to jacobi for additional information regarding
 * row_start, row_stop, and row_step.
 *
 * Parameters
 * ----------
 * Ap : array
 *     BSR row pointer
 * Aj : array
 *     BSR index array
 * Ax : array
 *     BSR data array
 * x : array, inplace
 *     approximate solution
 * b : array
 *     right hand side
 * temp : array, inplace
 *     temporary vector the same size as x
 * row_start : int
 *     beginning of the sweep (block row index)
 * row_stop : int
 *     end of the sweep (i.e. one past the last unknown)
 * row_step : int
 *     stride used during the sweep (may be negative)
 * blocksize : int
 *     BSR blocksize (blocks must be square)
 * omega : float
 *     damping parameter
 *
 * Returns
 * -------
 * Nothing, x will be modified inplace
 *
 */
template<class I, class T, class F>
void bsr_jacobi(const I Ap[], const int Ap_size,
                const I Aj[], const int Aj_size,
                const T Ax[], const int Ax_size,
                      T  x[], const int  x_size,
                const T  b[], const int  b_size,
                      T temp[], const int temp_size,
                const I row_start,
                const I row_stop,
                const I row_step,
                const I blocksize,
                const T omega[], const int omega_size)
{
    I B2 = blocksize*blocksize;
    T *rsum = new T[blocksize];
    T *Axloc = new T[blocksize];
    //T zero = 0.0;
    T one = 1.0;
    T omega2 = omega[0];

    // Determine if this is a forward, or backward sweep
    I step, step_start, step_end;
    if (row_step < 0){
        step = -1;
        step_start = blocksize-1;
        step_end = -1;
    }
    else{
        step = 1;
        step_start = 0;
        step_end = blocksize;
    }

    // copy x to temp
    for(I i = 0; i < abs(row_stop-row_start)*blocksize; i += step) {
        temp[i] = x[i];
    }

    for(I i = row_start; i != row_stop; i += row_step) {
        I start = Ap[i];
        I end   = Ap[i+1];
        I diag_ptr = -1;


        // initialize rsum to b, then later subtract A*x
        for(I k = 0; k < blocksize; k++) {
            rsum[k] = b[i*blocksize+k]; }

        // loop over row i
        for(I jj = start; jj < end; jj++){
            // extract column entry
            I j = Aj[jj];
            // absolute column entry for the start of this block
            I col = j*blocksize;

            if (i == j){    //point to where in Ax the diagonal block starts
                diag_ptr = jj*B2; }
            else {
                // do a dense multiply of this block times x and accumulate in rsum
                gemm(&(Ax[jj*B2]),  blocksize, blocksize, 'F',
                     &(temp[col]),  blocksize,   1,       'F',
                     &(Axloc[0]),   blocksize,   1,       'F',
                     'T');
                for(I m = 0; m < blocksize; m++) {
                    rsum[m] -= Axloc[m]; }
            }
        }

        // Carry out point-wise jacobi over the diagonal block,
        // all the other blocks have been factored into rsum.
        if (diag_ptr != -1) {
            for(I k = step_start; k != step_end; k+=step){
                T diag = 1.0;
                for(I kk = step_start; kk != step_end; kk+=step){
                    if(k == kk){
                        // diagonal entry
                        diag = Ax[k*blocksize + kk + diag_ptr]; }
                    else{
                        // off-diag entry
                        rsum[k] -= Ax[k*blocksize + kk + diag_ptr]*temp[i*blocksize+kk]; }
                }
                if (diag != (F) 0.0){
                    x[i*blocksize+k] = (one - omega2) * temp[i*blocksize+k] + omega2 * rsum[k]/diag; }
            }
        }

    } // end outer-most for loop

    delete[] rsum;
    delete[] Axloc;
}// end function



/*
 *  Perform one iteration of Jacobi relaxation on the linear
 *  system Ax = b for a given set of row indices, where A is
 *  stored in Block CSR format and x and b are column vectors.
 *  This method applies point-wise relaxation to the BSR matrix
 *  for a given set of row block indices, as opposed to "block
 *  relaxation".
 *
 *  Parameters
 *  ----------
 *  Ap : array
 *      BSR row pointer
 *  Aj : array
 *      BSR index array
 *  Ax : array
 *      BSR data array
 *  x : array
 *      approximate solution
 *  b : array
 *      right hand side
 *  indices : array
 *      list of row indices to perform Jacobi on, e.g., F-points.
 *      Note, it is assumed that indices correspond to blocks in A.
 *  blocksize : int
 *      BSR blocksize (blocks must be square)
 *  omega : float
 *      damping parameter
 *
 *  Returns
 *  -------
 *      Nothing, x will be modified in place
 *
 */
template<class I, class T, class F>
void bsr_jacobi_indexed(const I Ap[], const int Ap_size,
                        const I Aj[], const int Aj_size,
                        const T Ax[], const int Ax_size,
                              T  x[], const int  x_size,
                        const T  b[], const int  b_size,
                        const I indices[], const int indices_size,
                        const I blocksize,
                        const T omega[], const int omega_size)
{
    I B2 = blocksize*blocksize;
    T *rsum = new T[blocksize];
    T *Axloc = new T[blocksize];
    T omega2 = omega[0];
    T one = 1.0;

    // copy x to temp
    std::vector<T> temp(x_size);
    for(I i=0; i<x_size; i++) {
        temp[i] = x[i];
    }

    for(I i=0; i<indices_size; i++) {

        I row = indices[i];
        I start = Ap[row];
        I end   = Ap[row+1];
        I diag_ptr = -1;

        // initialize rsum to b, then later subtract A*x
        for(I k = 0; k < blocksize; k++) {
            rsum[k] = b[row*blocksize+k];
        }

        // loop over this row
        for(I jj=start; jj<end; jj++) {
            // extract column entry
            I j = Aj[jj];
            // absolute column entry for the start of this block
            I col = j*blocksize;

            if (row == j){    //point to where in Ax the diagonal block starts
                diag_ptr = jj*B2;
            }
            else {
                // do a dense multiply of this block times x and accumulate in rsum
                gemm(&(Ax[jj*B2]),  blocksize, blocksize, 'F',
                     &(temp[col]),  blocksize,   1,       'F',
                     &(Axloc[0]),   blocksize,   1,       'F',
                     'T');
                for(I m = 0; m < blocksize; m++) {
                    rsum[m] -= Axloc[m];
                }
            }
        }

        // Carry out point-wise jacobi over the diagonal block,
        // all the other blocks have been factored into rsum.
        if (diag_ptr != -1) {
            for(I k=0; k<blocksize; k++) {
                T diag = 1.0;
                for(I kk=0; kk<blocksize; kk++) {
                    if(k == kk){
                        // diagonal entry
                        diag = Ax[k*blocksize + kk + diag_ptr];
                    }
                    else{
                        // off-diag entry
                        rsum[k] -= Ax[k*blocksize + kk + diag_ptr] * temp[row*blocksize + kk];
                    }
                }

                // Check for nonzero diagonal and update ith index
                if (diag != (F) 0.0){
                    x[row*blocksize + k] = (one - omega2) * temp[row*blocksize + k] + omega2 * rsum[k]/diag;
                }
                else {
                    std::cout << "Warning : zero diagonal encountered in relaxation; ignored.\n";
                }
            }
        }
    }

    delete[] rsum;
    delete[] Axloc;
}


/*
 * Perform one iteration of Gauss-Seidel relaxation on the linear
 * system Ax = b, where A is stored in CSR format and x and b
 * are column vectors.
 *
 * Parameters
 * ----------
 * Ap : array
 *     CSR row pointer
 * Aj : array
 *     CSR index array
 * Ax : array
 *     CSR data array
 * x : array, inplace
 *     approximate solution
 * b : array
 *     right hand side
 * Id : array
 *     index array representing the
 * row_start : int
 *     beginning of the sweep (in array Id)
 * row_stop : int
 *     end of the sweep (in array Id)
 * row_step : int
 *     stride used during the sweep (may be negative)
 *
 * Returns
 * -------
 * Nothing, x will be modified inplace
 *
 * Notes
 * -----
 * Unlike gauss_seidel, which is restricted to updating a slice
 * of the unknowns (defined by row_start, row_start, and row_step),
 * this method updates unknowns according to the rows listed in
 * an index array.  This allows and arbitrary set of the unknowns
 * to be updated in an arbitrary order, as is necessary for the
 * relaxation steps in the Compatible Relaxation method.
 *
 * In this method the slice arguments are used to define the subset
 * of the index array Id which is to be considered.
 */
template<class I, class T, class F>
void gauss_seidel_indexed(const I Ap[], const int Ap_size,
                          const I Aj[], const int Aj_size,
                          const T Ax[], const int Ax_size,
                                T  x[], const int  x_size,
                          const T  b[], const int  b_size,
                          const I Id[], const int Id_size,
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

    if (diag != (F) 0.0){
      x[inew] = (b[inew] - rsum)/diag;
    }
  }
}

/*
 * Perform NE Jacobi on the linear system A x = b
 * This effectively carries out weighted-Jacobi on A^TA x = A^T b
 * (also known as Cimmino's relaxation)
 *
 * Parameters
 * ----------
 * Ap : array
 *     index pointer for CSR matrix A
 * Aj : array
 *     column indices for CSR matrix A
 * Ax : array
 *     value array for CSR matrix A
 * x : array, inplace
 *     current guess to the linear system
 * b : array
 *     right hand side
 * Tx : array
 *     scaled residual D_A^{-1} (b - Ax)
 * temp : array
 *     work space
 * row_start : int
 *     controls which rows to start on
 * row_stop : int
 *     controls which rows to stop on
 * row_step : int
 *     controls which rows to iterate over
 * omega : array
 *     size one array that contains the weighted-jacobi
 *     parameter.  An array must be used to pass in omega to
 *     account for the case where omega may be complex
 *
 * Returns
 * -------
 * x is modified inplace in an additive, not overwriting fashion
 *
 * Notes
 * -----
 * Primary calling routine is jacobi_ne in relaxation.py
 */
template<class I, class T, class F>
void jacobi_ne(const I Ap[], const int Ap_size,
               const I Aj[], const int Aj_size,
               const T Ax[], const int Ax_size,
                     T  x[], const int  x_size,
               const T  b[], const int  b_size,
               const T Tx[], const int Tx_size,
                     T temp[], const int temp_size,
               const I row_start,
               const I row_stop,
               const I row_step,
               const T omega[], const int omega_size)
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
 * This effectively carries out Gauss-Seidel on A A.H y = b,
 * where x = A.h y.
 *
 * Parameters
 * ----------
 * Ap : array
 *     index pointer for CSR matrix A
 * Aj : array
 *     column indices for CSR matrix A
 * Ax : array
 *     value array for CSR matrix A
 * x : array
 *     current guess to the linear system
 * b : array
 *     right hand side
 * Tx : array
 *     inverse(diag(A A.H))
 * omega : float
 *     relaxation parameter (if not 1.0, then algorithm becomes SOR)
 * row_start,stop,step : int
 *     controls which rows to iterate over
 *
 * Returns
 * -------
 * x is modified inplace in an additive, not overwriting fashion
 *
 * Notes
 * -----
 * Primary calling routine is gass_seidel_ne in relaxation.py
 */
template<class I, class T, class F>
void gauss_seidel_ne(const I Ap[], const int Ap_size,
                     const I Aj[], const int Aj_size,
                     const T Ax[], const int Ax_size,
                           T  x[], const int  x_size,
                     const T  b[], const int  b_size,
                     const I row_start,
                     const I row_stop,
                     const I row_step,
                     const T Tx[], const int Tx_size,
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
 * Ap : array
 *     index pointer for CSC matrix A
 * Aj : array
 *     row indices for CSC matrix A
 * Ax : array
 *     value array for CSC matrix A
 * x : array
 *     current guess to the linear system
 * z : array
 *     initial residual
 * Tx : array
 *     inverse(diag(A.H A))
 * omega : float
 *     relaxation parameter (if not 1.0, then algorithm becomes SOR)
 * col_start,stop,step : int
 *     controls which rows to iterate over
 *
 * Returns
 * -------
 * x is modified inplace in an additive, not overwriting fashion
 *
 * Notes
 * -----
 * Primary calling routine is gauss_seidel_nr in relaxation.py
 */
template<class I, class T, class F>
void gauss_seidel_nr(const I Ap[], const int Ap_size,
                     const I Aj[], const int Aj_size,
                     const T Ax[], const int Ax_size,
                           T  x[], const int  x_size,
                           T  z[], const int  z_size,
                     const I col_start,
                     const I col_stop,
                     const I col_step,
                     const T Tx[], const int Tx_size,
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
 * Perform one iteration of block Jacobi relaxation on the linear
 * system Ax = b, where A is stored in BSR format and x and b
 * are column vectors.  Damping is controlled by the omega
 * parameter.
 *
 * Refer to gauss_seidel for additional information regarding
 * row_start, row_stop, and row_step.
 *
 * Parameters
 * ----------
 * Ap : array
 *     BSR row pointer
 * Aj : array
 *     BSR index array
 * Ax : array
 *     BSR data array, blocks assumed square
 * x : array, inplace
 *     approximate solution
 * b : array
 *     right hand side
 * Tx : array
 *     Inverse of each diagonal block of A stored
 *     as a (n/blocksize, blocksize, blocksize) array
 * temp : array
 *     temporary vector the same size as x
 * row_start : int
 *     beginning of the sweep
 * row_stop : int
 *     end of the sweep (i.e. one past the last unknown)
 * row_step : int
 *     stride used during the sweep (may be negative)
 * omega : float
 *     damping parameter
 * blocksize int
 *     dimension of sqare blocks in BSR matrix A
 */
template<class I, class T, class F>
void block_jacobi(const I Ap[], const int Ap_size,
                  const I Aj[], const int Aj_size,
                  const T Ax[], const int Ax_size,
                        T  x[], const int  x_size,
                  const T  b[], const int  b_size,
                  const T Tx[], const int Tx_size,
                        T temp[], const int temp_size,
                  const I row_start,
                  const I row_stop,
                  const I row_step,
                  const T omega[], const int omega_size,
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
            if (i == j) {
                //diagonal, do nothing
                continue;
            }
            else {
                gemm(&(Ax[jj*blocksize_sq]), blocksize, blocksize, 'F',
                     &(temp[j*blocksize]),   blocksize, 1,         'F',
                     &(v[0]),                blocksize, 1,         'F',
                     'T');
                for(I k = 0; k < blocksize; k++) {
                    rsum[k] += v[k]; }
            }
        }

        // x[i*blocksize:(i+1)*blocksize] = (one - omega2) * temp[i*blocksize:(i+1)*blocksize] + omega2 *
        //          (Dinv[i*blocksize_sq : (i+1)*blocksize_sq]*(b[i*blocksize:(i+1)*blocksize] - rsum[0:blocksize]));
        I iblocksize = i*blocksize;
        for(I k = 0; k < blocksize; k++) {
            rsum[k] = b[iblocksize + k] - rsum[k]; }

        gemm(&(Dinv[i*blocksize_sq]), blocksize, blocksize, 'F',
             &(rsum[0]),              blocksize, 1,         'F',
             &(v[0]),                 blocksize, 1,         'F',
             'T');

        for(I k = 0; k < blocksize; k++) {
            x[iblocksize + k] = (one - omega2)*temp[iblocksize + k] + omega2*v[k]; }
    }

    delete[] v;
    delete[] rsum;
}


/*
 *  Perform one iteration of block Jacobi relaxation on the linear
 *  system Ax = b for a given set of (block) row indices. A is
 *  stored in BSR format and x and b are column vectors. Damping
 *  is controlled by the parameter omega.
 *
 *  Parameters
 *  ----------
 *  Ap : array
 *      BSR row pointer
 *  Aj : array
 *      BSR index array
 *  Ax : array
 *      BSR data array, blocks assumed square
 *  x : array
 *      approximate solution
 *  b : array
 *      right hand side
 *  Tx : array
 *      Inverse of each diagonal block of A stored
 *      as a (n/blocksize, blocksize, blocksize) array
 *  indices : array
 *      Indices
 *  omega : float
 *      damping parameter
 *  blocksize : int
 *      dimension of square blocks in BSR matrix A
 *
 *  Returns
 *  -------
 *      Nothing, x will be modified in place
 *
 */
template<class I, class T, class F>
void block_jacobi_indexed(const I Ap[], const int Ap_size,
                          const I Aj[], const int Aj_size,
                          const T Ax[], const int Ax_size,
                                T  x[], const int  x_size,
                          const T  b[], const int  b_size,
                          const T Tx[], const int Tx_size,
                          const I indices[], const int indices_size,
                          const T omega[], const int omega_size,
                          const I blocksize)
{
    // Rename
    const T * Dinv = Tx;
    T zero = 0.0;
    T one = 1.0;
    T omega2 = omega[0];
    T *rsum = new T[blocksize];
    T *v = new T[blocksize];
    I blocksize_sq = blocksize*blocksize;

    // Copy x to temp vector
    std::vector<T> temp(x_size);
    for(I i=0; i<x_size; i++) {
        temp[i] = x[i];
    }

    // Begin block Jacobi sweep
    for(I i=0; i<indices_size; i++) {
        I row = indices[i];
        I start = Ap[row];
        I end   = Ap[row+1];
        std::fill(&(rsum[0]), &(rsum[blocksize]), zero);

        // Carry out a block dot product between block row i and x
        for(I jj = start; jj < end; jj++) {
            I j = Aj[jj];
            if (row == j) {
                //diagonal, do nothing
                continue;
            }
            else {
                gemm(&(Ax[jj*blocksize_sq]), blocksize, blocksize, 'F',
                     &(temp[j*blocksize]),   blocksize, 1,         'F',
                     &(v[0]),                blocksize, 1,         'F',
                     'T');
                for(I k=0; k<blocksize; k++) {
                    rsum[k] += v[k];
                }
            }
        }

        // Compute b - Ax within block
        for(I k = 0; k < blocksize; k++) {
            rsum[k] = b[row*blocksize + k] - rsum[k];
        }

        // Apply D^{-1} for this block row to b - Ax
        gemm(&(Dinv[row*blocksize_sq]), blocksize, blocksize, 'F',
             &(rsum[0]),              blocksize, 1,         'F',
             &(v[0]),                 blocksize, 1,         'F',
             'T');

        // Update each element in ith block
        for(I k = 0; k < blocksize; k++) {
            x[row*blocksize + k] = (one - omega2)*temp[row*blocksize + k] + omega2*v[k];
        }
    }

    delete[] v;
    delete[] rsum;
}


/*
 * Perform one iteration of block Gauss-Seidel relaxation on
 * the linear system Ax = b, where A is stored in BSR format
 * and x and b are column vectors.
 *
 * Refer to gauss_seidel for additional information regarding
 * row_start, row_stop, and row_step.
 *
 * Parameters
 * ----------
 * Ap : array
 *     BSR row pointer
 * Aj : array
 *     BSR index array
 * Ax : array
 *     BSR data array, blocks assumed square
 * x : array, inplace
 *     approximate solution
 * b : array
 *     right hand side
 * Tx : array
 *     Inverse of each diagonal block of A stored
 *     as a (n/blocksize, blocksize, blocksize) array
 * row_start : int
 *     beginning of the sweep
 * row_stop : int
 *     end of the sweep (i.e. one past the last unknown)
 * row_step : int
 *     stride used during the sweep (may be negative)
 * blocksize : int
 *     dimension of square blocks in BSR matrix A
 */
template<class I, class T, class F>
void block_gauss_seidel(const I Ap[], const int Ap_size,
                        const I Aj[], const int Aj_size,
                        const T Ax[], const int Ax_size,
                              T  x[], const int  x_size,
                        const T  b[], const int  b_size,
                        const T Tx[], const int Tx_size,
                        const I row_start,
                        const I row_stop,
                        const I row_step,
                        const I blocksize)
{
    // Rename
    const T * Dinv = Tx;

    T zero = 0.0;
    T *rsum = new T[blocksize];
    T *v = new T[blocksize];
    I blocksize_sq = blocksize*blocksize;

    // Begin block Gauss-Seidel sweep
    for(I i = row_start; i != row_stop; i += row_step) {
        I start = Ap[i];
        I end   = Ap[i+1];
        std::fill(&(rsum[0]), &(rsum[blocksize]), zero);

        // Carry out a block dot product between block row i and x
        for(I jj = start; jj < end; jj++){
            I j = Aj[jj];
            if (i == j) {
                //diagonal, do nothing
                continue;
            }
            else {
                gemm(&(Ax[jj*blocksize_sq]), blocksize, blocksize, 'F',
                     &(x[j*blocksize]),      blocksize, 1,         'F',
                     &(v[0]),                blocksize, 1,         'F',
                     'T');
                for(I k = 0; k < blocksize; k++) {
                    rsum[k] += v[k]; }
            }
        }

        // x[i*blocksize:(i+1)*blocksize] = (Dinv[i*blocksize_sq : (i+1)*blocksize_sq]*(b[i*blocksize:(i+1)*blocksize] - rsum[0:blocksize]));
        I iblocksize = i*blocksize;
        for(I k = 0; k < blocksize; k++) {
            rsum[k] = b[iblocksize + k] - rsum[k]; }

        gemm(&(Dinv[i*blocksize_sq]), blocksize, blocksize, 'F',
             &(rsum[0]),              blocksize, 1,         'F',
             &(x[iblocksize]),        blocksize, 1,         'F',
             'T');
    }

    delete[] v;
    delete[] rsum;
}

/*
 * Extract diagonal blocks from A and insert into a linear array.
 * This is a helper function for overlapping_schwarz_csr.
 *
 * Parameters
 * ----------
 * Ap : array
 *     CSR row pointer
 * Aj : array
 *     CSR index array
 *     must be sorted for each row
 * Ax : array
 *     CSR data array, blocks assumed square
 * Tx : array, inplace
 *     Inverse of each diagonal block of A, stored in row major
 * Tp : array
 *     Pointer array into Tx indicating where the
 *     diagonal blocks start and stop
 * Sj : array
 *     Indices of each subdomain
 *     must be sorted over each subdomain
 * Sp : array
 *     Pointer array indicating where each subdomain
 *     starts and stops
 * nsdomains : int
 *     Number of subdomains
 * nrows : int
 *     Number of rows
 *
 * Returns
 * -------
 * Nothing, Tx will be modified inplace
 */
template<class I, class T, class F>
void extract_subblocks(const I Ap[], const int Ap_size,
                       const I Aj[], const int Aj_size,
                       const T Ax[], const int Ax_size,
                             T Tx[], const int Tx_size,
                       const I Tp[], const int Tp_size,
                       const I Sj[], const int Sj_size,
                       const I Sp[], const int Sp_size,
                       const I nsdomains,
                       const I nrows)
{
    // Initialize Tx to zero
    T zero = 0.0;
    std::fill(&(Tx[0]), &(Tx[Tp[nsdomains]]), zero);

    // Loop over each subdomain
    for(I i = 0; i < nsdomains; i++) {
        // Calculate the smallest and largest column index for this
        // diagonal block
        I lower = Sj[Sp[i]];
        I upper = Sj[Sp[i+1]-1];

        I Tx_offset = Tp[i];
        I row_length = Sp[i+1] - Sp[i];

        // Loop over subdomain i
        for(I j = Sp[i]; j < Sp[i+1]; j++) {
            // Peel off this row from A and insert into Tx
            I row = Sj[j];
            I start = Ap[row];
            I end = Ap[row+1];
            I local_col = 0;
            I placeholder = Sp[i];

            for(I k = start; k < end; k++) {
                I col = Aj[k];

                // Must decide if col is a member of this subdomain, and while
                // doing so, track the current local column number from 0 to
                // row_length
                if ((col >= lower) && (col <= upper) ) {
                    while(placeholder < Sp[i+1]){
                        if(Sj[placeholder] == col) {
                            //insert into Tx
                            Tx[Tx_offset + local_col] = Ax[k];
                            local_col++;
                            placeholder++;
                            break;
                        }
                        else if (Sj[placeholder] > col ){
                            break;
                        }
                        else{
                            local_col++;
                            placeholder++;
                        }
                    }
                }
            }

            Tx_offset += row_length;
        }
    }
}


/*
 * Perform one iteration of an overlapping Schwarz relaxation on
 * the linear system Ax = b, where A is stored in CSR format
 * and x and b are column vectors.
 *
 * Refer to gauss_seidel for additional information regarding
 * row_start, row_stop, and row_step.
 *
 * Parameters
 * ----------
 * Ap : array
 *     CSR row pointer
 * Aj : array
 *     CSR index array
 * Ax : array
 *     CSR data array, blocks assumed square
 * x : array, inplace
 *     approximate solution
 * b : array
 *     right hand side
 * Tx : array
 *     Inverse of each diagonal block of A, stored in row major
 * Tp : array
 *     Pointer array into Tx indicating where the diagonal blocks start and stop
 * Sj : array
 *     Indices of each subdomain
 *     must be sorted over each subdomain
 * Sp : array
 *     Pointer array indicating where each subdomain starts and stops
 * nsdomains
 *     Number of subdomains
 * nrows
 *     Number of rows
 * row_start : int
 *     The subdomains are processed in this order,
 * row_stop : int
 *     for(i = row_start, i != row_stop, i+=row_step)
 * row_step : int
 *     {...computation...}
 *
 * Returns
 * -------
 * Nothing, x will be modified inplace
 *
 */
template<class I, class T, class F>
void overlapping_schwarz_csr(const I Ap[], const int Ap_size,
                             const I Aj[], const int Aj_size,
                             const T Ax[], const int Ax_size,
                                   T  x[], const int  x_size,
                             const T  b[], const int  b_size,
                             const T Tx[], const int Tx_size,
                             const I Tp[], const int Tp_size,
                             const I Sj[], const int Sj_size,
                             const I Sp[], const int Sp_size,
                                   I nsdomains,
                                   I nrows,
                                   I row_start,
                                   I row_stop,
                                   I row_step)
{

    //T zero = 0.0;
    T *rsum = new T[nrows];
    T *Dinv_rsum = new T[nrows];

    // Initialize rsum and Dinv_rsum
    for(I k = 0; k < nrows; k++) {
        rsum[k] = 0.0;
        Dinv_rsum[k] = 0.0;
    }

    // Begin loop over the subdomains
    for(I domptr = row_start; domptr != row_stop; domptr+=row_step) {

        I counter = 0;
        I size_domain = Sp[domptr+1] - Sp[domptr];

        // Begin block calculation of the residual
        for(I j = Sp[domptr]; j < Sp[domptr+1]; j++) {
            // For this row, calculate the residual
            I row = Sj[j];
            I start = Ap[row];
            I end   = Ap[row+1];
            for(I jj = start; jj < end; jj++){
                rsum[counter] -= Ax[jj]*x[Aj[jj]];
            }

            // Account for the RHS
            rsum[counter] += b[row];
            counter++;
        }

        // Multiply block residual with block inverse of A
        gemm(&(Tx[Tp[domptr]]), size_domain, size_domain, 'F',
             &(rsum[0]),      size_domain,   1,         'F',
             &(Dinv_rsum[0]), size_domain,   1,         'F',
             'F');

        // Add to x
        counter = 0;
        for(I j = Sp[domptr]; j < Sp[domptr+1]; j++) {
            x[Sj[j]] += Dinv_rsum[counter];
            counter++;
            }

        // Set rsum and Dinv_rsum back to zero for the next iteration
        for(I k = 0; k < size_domain; k++) {
            rsum[k] = 0.0;
            Dinv_rsum[k] = 0.0;
        }


    }

    delete[] rsum;
    delete[] Dinv_rsum;
}


#endif
