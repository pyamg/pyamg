#ifndef KRYLOV_H
#define KRYLOV_H

#include "linalg.h"

/* Apply |start-stop| Householder reflectors in B to z
 * B stores the Householder vectors in row major form
 *
 * Implements the below python
 *
 * for j in range(start,stop,step):
 *   z = z - 2.0*dot(conjugate(B[j,:]), v)*B[j,:]
 */
template<class I, class T, class F>
void apply_householders(T z[], const T B[], const I n, const I start, const I stop, const I step)
{
    I index = start*n;
    I index_step = step*n;
    const T * Bptr;
    for(I i = start; i != stop; i+=step)
    {
        Bptr = &(B[index]);
        T alpha = dot_prod(Bptr, z, n)*(-2.0);
        axpy(z, Bptr, alpha, n);
        index += index_step;
    }
}

/* For use after gmres is finished iterating and the least squares
 * solution has been found.  This routine maps the solution back to
 * the original space via the Householder reflectors.
 *
 * Apply |start-stop| Householder reflectors in B to z
 * while also adding in the appropriate value from y, so 
 * that we follow the Horner-like scheme to map our least squares
 * solution in y back to the original space
 * 
 * B stores the Householder vectors in row major form
 *
 * Implements the below python
 *
 *for j in range(inner,-1,-1):
 *  z[j] += y[j]
 *  # Apply j-th reflector, (I - 2.0*w_j*w_j.T)*upadate
 *  z = z - 2.0*dot(conjugate(B[j,:]), update)*B[j,:]
 */
template<class I, class T, class F>
void householder_hornerscheme(T z[], const T B[], const T y[], const I n, const I start, const I stop, const I step)
{
    I index = start*n;
    I index_step = step*n;
    const T * Bptr;
    for(I i = start; i != stop; i+=step)
    {
        z[i] += y[i];
        Bptr = &(B[index]);

        T alpha = dot_prod(Bptr, z, n)*(-2.0);
        axpy(z, Bptr, alpha, n);

        index += index_step;
    }
}


/* Apply the first nrot Given's rotations in B to x
 * x is an n-vector
 * B is a one dimensional array, but each 4 entries represent
 *   a Given's rotation
 */
template<class I, class T, class F>
void apply_givens(const T B[], T x[], const I n, const I nrot)
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

#endif
