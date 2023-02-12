#ifndef KRYLOV_H
#define KRYLOV_H

#include "linalg.h"

/* Apply Householder reflectors in B to z
 *
 * Implements the below python
 *
 * .. code-block:: python
 *
 *     for j in range(start,stop,step):
 *       z = z - 2.0*dot(conjugate(B[j,:]), v)*B[j,:]
 *
 * Parameters
 * ----------
 * z : array
 *     length n vector to be operated on
 * B : array
 *     n x m matrix of householder reflectors
 *     must be in row major form
 * n : int
 *     dimensionality of z
 * start, stop, step : int
 *     control the choice of vectors in B to use
 *
 * Returns
 * -------
 * z is modified in place to reflect the application of
 * the Householder reflectors, B[:,range(start,stop,step)]
 *
 * Notes
 * -----
 * Principle calling routine is gmres(...) and fgmres(...) in krylov.py
 */
template<class I, class T, class F>
void apply_householders(      T z[], const int z_size,
                        const T B[], const int B_size,
                        const I n,
                        const I start,
                        const I stop,
                        const I step)
{
    I index = start*n;
    I index_step = step*n;
    const T * Bptr;
    for(I i = start; i != stop; i+=step)
    {
        Bptr = &(B[index]);
        T alpha = dot_prod(Bptr, z, n);
        alpha *= -2;
        axpy(z, Bptr, alpha, n);
        index += index_step;
    }
}

/* For use after gmres is finished iterating and the least squares
 * solution has been found.  This routine maps the solution back to
 * the original space via the Householder reflectors.
 *
 * Apply Householder reflectors in B to z
 * while also adding in the appropriate value from y, so
 * that we follow the Horner-like scheme to map our least squares
 * solution in y back to the original space
 *
 * Implements the below python
 *
 * .. code-block:: python
 *
 *     for j in range(inner,-1,-1):
 *       z[j] += y[j]
 *       # Apply j-th reflector, (I - 2.0*w_j*w_j.T)*update
 *       z = z - 2.0*dot(conjugate(B[j,:]), update)*B[j,:]
 *
 * Parameters
 * ----------
 * z : array
 *     length n vector to be operated on
 * B : array
 *     n x m matrix of householder reflectors
 *     must be in row major form
 * y : array
 *     solution to the reduced system at the end of GMRES
 * n : int
 *     dimensionality of z
 * start, stop, step : int
 *     control the choice of vectors in B to use
 *
 * Returns
 * -------
 * z is modified in place to reflect the application of
 * the Householder reflectors, B[:,range(start,stop,step)],
 * and the inclusion of values in y.
 *
 * Notes
 * -----
 * Principle calling routine is gmres(...) and fgmres(...) in krylov.py
 *
 * References
 * ----------
 * See pages 164-167 in Saad, "Iterative Methods for Sparse Linear Systems"
 */
template<class I, class T, class F>
void householder_hornerscheme(      T z[], const int z_size,
                               const T B[], const int B_size,
                               const T y[], const int y_size,
                               const I n,
                               const I start,
                               const I stop,
                               const I step)
{
    I index = start*n;
    I index_step = step*n;
    const T * Bptr;
    for(I i = start; i != stop; i+=step)
    {
        z[i] += y[i];
        Bptr = &(B[index]);

        T alpha = dot_prod(Bptr, z, n);
        alpha *= -2;
        axpy(z, Bptr, alpha, n);

        index += index_step;
    }
}


/* Apply the first nrot Givens rotations in B to x
 *
 * Parameters
 * ----------
 * x : array
 *     n-vector to be operated on
 * B : array
 *     Each 4 entries represent a Givens rotation
 *     length nrot*4
 * n : int
 *     dimensionality of x
 * nrot : int
 *     number of rotations in B
 *
 * Returns
 * -------
 * x is modified in place to reflect the application of the nrot
 * rotations in B.  It is assumed that the first rotation operates on
 * degrees of freedom 0 and 1.  The second rotation operates on dof's 1 and 2,
 * and so on
 *
 * Notes
 * -----
 * Principle calling routine is gmres(...) and fgmres(...) in krylov.py
 */
template<class I, class T, class F>
void apply_givens(const T B[], const int B_size,
                        T x[], const int x_size,
                  const I n,
                  const I nrot)
{
    I ind1 = 0;
    I ind2 = 1;
    I ind3 = 2;
    I ind4 = 3;
    T x_temp;

    for(I rot=0; rot < nrot; rot++)
    {
        // Apply rotation
        x_temp = x[rot];
        x[rot]   = B[ind1]*x_temp + B[ind2]*x[rot+1];
        x[rot+1] = B[ind3]*x_temp + B[ind4]*x[rot+1];

        // Increment indices
        ind1 +=4;
        ind2 +=4;
        ind3 +=4;
        ind4 +=4;
    }
}

/* 
 * Parameters
 * ----------
 *        A : double array, length n*n
 *            Matrix stored in column- or row-major.
 *        b : double array, length n
 *            Right hand side of linear system
 *        x : double array, length n
 *            Preallocated array for solution
 *        n : int
 *            Number of rows and columns in A
 *        is_col_major : bool
 *            True if A is stored in column-major, false
 *            if A is stored in row-major
 *        maxiter : int, default 10
 *            Maximum GMRES iterations
 *        precondition : bool, default 1
 *            Use diagonal preconditioner
 *
 * Returns
 * -------
 *        Nothing, solution is stored in x[].
 *
 * Notes
 * -----
 * This method does not currently check residual for stopping criterion.
 *
 */
template<class I, class T>
void dense_GMRES(T A[], 
                 T b[],
                 T x[],
                 const I n,
                 const I is_col_major,
                 I maxiter = 10,
                 I precondition = 1)
{
    // If maxiter = 0, set to n
    if (maxiter == 0) {
        maxiter = n;
    }
    else {
        maxiter = std::min(maxiter,n);
    }

    // Function pointer for row or column major matrices. C_h is set for Hessenberg
    // matrix H, which has dimensions (maxiter+1) x maxiter
    I (*get_ind)(const I, const I, const I);
    I C_h;
    if (is_col_major) {
        get_ind = &col_major;
        C_h = maxiter+1;
    }
    else {
        get_ind = &row_major;
        C_h = maxiter;
    }

    // If n = 1 or 2, return direct solve
    if (n == 1) {
        x[0] = b[0] / A[0];
        return;
    }

    // Scale out diagonal of system as preconditioner
    if (precondition) {
        for (I i=0; i<n; i++) {
            T d_i = A[get_ind(i,i,n)];
            if (std::abs(d_i) < 1e-12) {
                std::cout << "Warning: zero diagonal; skipping.\n";
                continue;
            }
            d_i = 1.0 / d_i;
            b[i] *= d_i;
            for (I j=0; j<n; j++) {
                A[get_ind(i,j,n)] *= d_i;
            }
        }
    }

    // Preallocate space for vectors / matrices
    std::vector<T> V(maxiter*n, 0);
    std::vector<T> H(maxiter*(maxiter+1), 0);
    std::vector<T> g(n+1, 0);
    I rank = maxiter;

    // Set v0 =  M^{-1}b / ||M^{-1}b||, and g = ||v0||*e1. If ||M^{-1}b|| ~ 0, return zero solution.
    T normb = norm(b, n);
    if (normb < 1e-12) {
        for (I i=0; i<n; i++) {
            x[i] = 0.0;
        }
        return;
    }
    g[0] = normb;
    for (I i=0; i<n; i++) {
        V[i] = b[i] / normb;
    }

    // Loop over GMRES iterations
    for (I j=0; j<maxiter; j++) {

        // Form new search direction, w = M^{-1}Av_j (overwriting b with w)
        I v_ind = n * j;
        for (I l=0; l<n; l++) {
            b[l] = 0.0;
            for (I k=0; k<n; k++) {
                b[l] += A[get_ind(l,k,n)] * V[v_ind + k];
            }
        }

        // Modified Gram-Schmidt orthogonalization
        for (I i=0; i<=j; i++) {
            I v_ind = i * n;
            T temp = dot_prod(b, &V[v_ind], n);
            H[get_ind(i,j,C_h)] = temp;
            axpy(b, &V[v_ind], -temp, n);
        }

        // Check if residual approximately zero; if so, set matrix rank and exit loop
        normb = norm(b, n);
        if (normb < 1e-12) {
            rank = j+1;
            if (j < (maxiter-1)) {
                H[get_ind(j+1,j,C_h)] = 0.0;
            }
            break;
        }
        // Update Hessenberg matrix H and basis vectors V
        else {
            if (j < (maxiter-1)) {
                H[get_ind(j+1,j,C_h)] = normb;
                for (I i=0; i<n; i++) {
                    V[(j+1)*n + i] = b[i] / normb;
                }
            } 
        }
    }

    // Use Givens rotations to transform H from Hessenberg to triangular
    //  - Note: this can be done inside the loop as well, but there is something
    //    subtle I was missing. Moving in as is, we lose orthogonality of V. 
    for (int j=1; j<=maxiter; j++) {

        // Get 1st column of 2x2 block to rotate in H
        T h11 = H[get_ind(j-1,j-1,C_h)];
        T h21 = H[get_ind(j,j-1,C_h)];

        // If h21 (element to be rotated out) = 0, skip Givens rotation
        if (h21 == 0) {
            continue;
        }

        // Define constants in Givens rotation, G = [C1, S1; -S1, C1]
        T C1 = 1.0 / std::sqrt(h11*h11 + h21*h21);
        T S1 = h21 * C1;
        C1 *= h11;

        // Apply G to vector b
        T temp = g[j-1];
        g[j-1] = C1 * temp + S1 * g[j];
        g[j] = -S1 * temp + C1 * g[j];

        // Apply G to rows j and (j-1) of H, column by column
        for (I k=(j-1); k<maxiter; k++) {
            temp = H[get_ind(j-1,k,C_h)];
            H[get_ind(j-1,k,C_h)] = C1 * temp + S1 * H[get_ind(j,k,C_h)];
            H[get_ind(j,k,C_h)] = -S1 * temp + C1 * H[get_ind(j,k,C_h)];
        }
        H[get_ind(j,j-1,C_h)] = 0;
    }

    // Solve upper triangular system
    upper_tri_solve(&H[0], &g[0], b, maxiter+1, maxiter, is_col_major);

    // Multiply least squares solution of Hessenberg system by V
    for (I l=0; l<n; l++) {
        x[l] = 0.0;
        for (I k=0; k<rank; k++) {
            x[l] += V[col_major(l,k,n)] * b[k];
        }
    } 
}

#endif
