#ifndef LINALG_H
#define LINALG_H

#include <math.h>
#include <limits>
#include <complex>
#include <iostream>

/*******************************************************************
 * Overloaded routines for real arithmetic for int, float and double
 *******************************************************************/

/* Sign-of Function overloaded for int, float and double
 * signof(x) =  1 if x > 0
 * signof(x) = -1 if x < 0
 * signof(0) =  1 if x = 0
 */
inline int signof(int a) { return (a<0 ? -1 : 1); }
inline float signof(float a) { return (a<0.0 ? -1.0 : 1.0); }
inline double signof(double a) { return (a<0.0 ? -1.0 : 1.0); }



/*******************************************************************
 *         Overloaded routines for complex arithmetic for
 *         pyamg's complex class, float and double
 *******************************************************************/

/*
 * Return the complex conjugate of a number
 */
inline float conjugate(const float& x)
    { return x; }
inline double conjugate(const double& x)
    { return x; }
inline std::complex<float> conjugate(const std::complex<float>& x)
{
    std::complex<float> tmp (x.real(), -x.imag());
    return tmp; }
inline std::complex<double> conjugate(const std::complex<double>& x)
{
    std::complex<double> tmp (x.real(), -x.imag());
    return tmp; }

/*
 * Return the real part of a number
 */
inline float real(const float& x)
    { return x; }
inline double real(const double& x)
    { return x; }
inline float real(const std::complex<float>& x)
    { return x.real(); }
inline double real(const std::complex<double>& x)
    { return x.real(); }

/*
 * Return the imaginary part of a number
 */
inline float imag(const float& x)
    { return 0.0; }
inline double imag(const double& x)
    { return 0.0; }
inline float imag(const std::complex<float>& x)
    { return x.imag(); }
inline double imag(const std::complex<double>& x)
    { return x.imag(); }

/*
 * Return the norm, i.e. the magnitude, of a single number
 */
inline float mynorm(const float& x)
    { return fabs(x); }
inline double mynorm(const double& x)
    { return fabs(x); }
inline float mynorm(const std::complex<float>& x)
    { return sqrt(x.real()*x.real() + x.imag()*x.imag()); }
inline double mynorm(const std::complex<double>& x)
    { return sqrt(x.real()*x.real() + x.imag()*x.imag()); }

/*
 * Return the norm squared of a single number, i.e.  save a square root
 */
inline float mynormsq(const float& x)
    { return (x*x); }
inline double mynormsq(const double& x)
    { return (x*x); }
inline float mynormsq(const std::complex<float>& x)
    { return (x.real()*x.real() + x.imag()*x.imag()); }
inline double mynormsq(const std::complex<double>& x)
    { return (x.real()*x.real() + x.imag()*x.imag()); }

/*
 * Return the input, but with the real part zeroed out
 */
inline float zero_real(float& x)
    { return 0.0; }
inline double zero_real(double& x)
    { return 0.0; }
inline std::complex<float> zero_real(std::complex<float>& x)
{
    std::complex<float> tmp (0.0, x.imag());
    return x; }
inline std::complex<double> zero_real(std::complex<double>& x)
{
    std::complex<double> tmp (0.0, x.imag());
    return x; }

/*
 * Return the input, but with the imag part zeroed out
 */
inline float zero_imag(float& x)
    { return x; }
inline double zero_imag(double& x)
    { return x; }
inline std::complex<float> zero_imag(std::complex<float>& x)
{
    std::complex<float> tmp (x.real(), 0.0);
    return x; }
inline std::complex<double> zero_imag(std::complex<double>& x)
{
    std::complex<double> tmp (x.real(), 0.0);
    return x; }

/* 
 * Return row-major index from 2d array index, A[row,col].
 */
template<class I>
inline I row_major(const I row, const I col, const I num_cols) 
{
    return row*num_cols + col;
}

/*
 * Return column-major index from 2d array index, A[row,col]. 
 */
template<class I>
inline I col_major(const I row, const I col, const I num_rows) 
{
    return col*num_rows + row;
}

/*******************************************************************
 *              Dense Linear Algebra Routines
 *      templated for pyamg's complex class, float and double
 *******************************************************************/

/* dot(x, y, n)
 *
 * Parameters
 * ----------
 * x : array
 *     n-vector
 * y : array
 *     n-vector
 * n : int
 *     size of x and y
 *
 * Returns
 * -------
 * float
 *     conjugate(x).T y
 *
 */
template<class I, class T>
inline T dot_prod(const T x[], const T y[], const I n)
{
    T sum = 0.0;
    for( I i = 0; i < n; i++)
    {   sum += conjugate(x[i])*y[i]; }
    return sum;
}


/* norm(x, n)
 *
 * Parameters
 * ----------
 * x : array
 *     n-vector
 * n : int
 *     size of x and y
 * normx : float
 *     output value
 *
 * Returns
 * -------
 * float
 *     normx = sqrt( <x, x> )
 *
 */
template<class I, class T, class F>
inline void norm(const T x[], const I n, F &normx)
{
    normx = sqrt(real(dot_prod(x,x,n)));
}

/* norm(x, n)
 *
 * Parameters
 * ----------
 * x : array
 *     n-vector
 * n : int
 *     size of x and y
 * normx : float
 *     output value
 *
 * Returns
 * -------
 * float
 *     sqrt( <x, x> )
 *
 */
template<class I, class T>
inline T norm(const T x[], const I n)
{
    return std::sqrt(dot_prod(x,x,n));
}

/* axpy(x, y, alpha, n)
 *
 * Parameters
 * ----------
 * x : array
 *     n-vector
 * y : array
 *     n-vector
 * n : int
 *     size of x and y
 * alpha : float
 *     value to scale with
 *
 * Returns
 * -------
 * x : float
 *     x = x + alpha*y
 */
template<class I, class T>
inline void axpy(T x[], const T y[], const T alpha, const I n)
{
    for( I i = 0; i < n; i++)
    {   x[i] += alpha*y[i]; }
}


/* Transpose Ax by overwriting Bx
 *
 * Parameters
 * ----------
 * Ax : array
 *      m x n dense array
 * Bx :array
 *      m x n dense array
 * m : int
 *      Dimensions of Ax
 * n : int
 *      Dimensions of Bx
 *
 * Returns
 * -------
 * Bx is overwritten with the transpose of Ax
 *
 * Notes
 * -----
 * There is a fair amount of hard-coding to make this routine very
 * fast for small (<10) square matrices, although it works for general
 * m x n matrices.
 *
 */
template<class I, class T>
inline void transpose(const T Ax[], T Bx[], const I m, const I n)
{
    // Almost all uses of this function are for
    // m==n, m,n<10.  Hence the attempts at speed.

    //Hard code the smallest examples for speed
    if( (m==1) && (n==1))
    {   Bx[0] = Ax[0]; }
    else if( (m==2) && (n==2))
    {
        Bx[0] = Ax[0];
        Bx[1] = Ax[2];
        Bx[2] = Ax[1];
        Bx[3] = Ax[3];
    }
    else if( (m==3) && (n==3))
    {
        Bx[0] = Ax[0];
        Bx[1] = Ax[3];
        Bx[2] = Ax[6];
        Bx[3] = Ax[1];
        Bx[4] = Ax[4];
        Bx[5] = Ax[7];
        Bx[6] = Ax[2];
        Bx[7] = Ax[5];
        Bx[8] = Ax[8];
    }
    // Do some hard coding for the other common examples
    else if ( (m == n) && (m < 11))
    {
        I j = 0;
        for(I i = 0; i < m*m; i+=m)
        {
            if(m == 4)
            {   Bx[i] = Ax[j]; Bx[i+1] = Ax[j+4]; Bx[i+2] = Ax[j+8]; Bx[i+3] = Ax[j+12]; }
            if(m == 5)
            {   Bx[i] = Ax[j]; Bx[i+1] = Ax[j+5]; Bx[i+2] = Ax[j+10]; Bx[i+3] = Ax[j+15];
                Bx[i+4] = Ax[j+20]; }
            if(m == 6)
            {   Bx[i] = Ax[j]; Bx[i+1] = Ax[j+6]; Bx[i+2] = Ax[j+12]; Bx[i+3] = Ax[j+18];
                Bx[i+4] = Ax[j+24];  Bx[i+5] = Ax[j+30]; }
            if(m == 7)
            {   Bx[i] = Ax[j]; Bx[i+1] = Ax[j+7]; Bx[i+2] = Ax[j+14]; Bx[i+3] = Ax[j+21];
                Bx[i+4] = Ax[j+28];  Bx[i+5] = Ax[j+35]; Bx[i+6] = Ax[j+42]; }
            if(m == 8)
            {   Bx[i] = Ax[j]; Bx[i+1] = Ax[j+8]; Bx[i+2] = Ax[j+16]; Bx[i+3] = Ax[j+24];
                Bx[i+4] = Ax[j+32];  Bx[i+5] = Ax[j+40]; Bx[i+6] = Ax[j+48]; Bx[i+7] = Ax[j+56]; }
            if(m == 9)
            {   Bx[i] = Ax[j]; Bx[i+1] = Ax[j+9]; Bx[i+2] = Ax[j+18]; Bx[i+3] = Ax[j+27];
                Bx[i+4] = Ax[j+36];  Bx[i+5] = Ax[j+45]; Bx[i+6] = Ax[j+54]; Bx[i+7] = Ax[j+63];
                Bx[i+8] = Ax[j+72];}
            if(m == 10)
            {   Bx[i] = Ax[j]; Bx[i+1] = Ax[j+10]; Bx[i+2] = Ax[j+20]; Bx[i+3] = Ax[j+30];
                Bx[i+4] = Ax[j+40];  Bx[i+5] = Ax[j+50]; Bx[i+6] = Ax[j+60]; Bx[i+7] = Ax[j+70];
                Bx[i+8] = Ax[j+80]; Bx[i+9] = Ax[j+90];}

            j++;
        }
    }
    // Finally, the general case
    else
    {
        I Bcounter = 0;
        for(I i = 0; i < n; i++)
        {
            I Acounter = i;
            for(I j = 0; j < m; j++)
            {
                //B[i,j] = A[j,i]
                Bx[Bcounter] = Ax[Acounter];
                Bcounter++;
                Acounter+=n;
            }
        }
    }

    return;
}


/* Calculate Ax*Bx = S
 *
 * Parameters
 * ----------
 * Ax : array
 *      Stored in row major
 * Arows : int
 *      Number of rows of A
 * Acols : int
 *      Number of columns of A
 * Atrans : char
 *      Not Used
 * Bx : array
 *      Stored in col major
 * Brows : int
 *      Number of rows of B
 * Bcols : int
 *      Number of columns of B
 * Btrans : char
 *      Supported, essentially Btrans='F' assumes
 *      B is in column major, and Brans='T' assumes
 *      B is in row major
 * Sx : array
 *      Output array, Contents are overwritten
 * Srows : int
 *      Number of rows of S
 * Scols : int
 *      Number of columns of S
 * Strans : char
 *      'T' gives S in col major (only works with Btrans='F')
 *      'F' gives S in row major
 * overwrite : {char}
 *      'T' overwrite S
 *      'F' accumulate to S
 *
 * Returns
 * -------
 * Modified inplace: Sx = Ax*Bx in column or row major, depending on Strans.
 *
 * Notes
 * -----
 * Supported matrix format combinations,
 * - Btrans = 'F' and Strans = 'T'
 * - Btrans = 'F' and Strans = 'F'
 * - Btrans = 'T' and Strans = 'F'
 *
 * All other combinations are not yet supported
 */
template<class I, class T>
inline void gemm(const T Ax[], const I Arows, const I Acols, const char Atrans,
          const T Bx[], const I Brows, const I Bcols, const char Btrans,
          T Sx[], const I Srows, const I Scols, const char Strans,
          const char overwrite)
{
    //Add checks for dimensions, but leaving them out speeds things up
    //Add functionality for transposes

    if(overwrite == 'T'){
        std::fill(Sx, Sx + Srows*Scols,  0); }

    if( (Strans == 'T') && (Btrans == 'F'))
    {
        // A is in row major, B is in column major, so compute
        // S(i,j) = A(i,:) B(:,j) by looping over the rows of A
        // and the columns of B.

        I s_counter = 0; I a_counter =0; I b_counter =0; I a_start = 0;
        for(I i = 0; i < Arows; i++)
        {
            s_counter = i;
            b_counter = 0;
            for(I j = 0; j < Bcols; j++)
            {
                a_counter = a_start;                                // a_counter cycles through rows of A
                for(I k = 0; k < Brows; k++)
                {
                    //S[i,j] += Ax[i,k]*B[k,j]
                    Sx[s_counter] += Ax[a_counter]*Bx[b_counter];
                    a_counter++; b_counter++;
                }
                s_counter+=Srows;
            }
            a_start += Acols;
        }
    }
    else if((Strans == 'F') && (Btrans == 'F'))
    {
        // A is in row major, B is in column major, so compute
        // S(i,j) = A(i,:) B(:,j) by looping over the rows of A
        // and the columns of B.

        I s_counter = 0; I a_counter =0; I b_counter =0; I a_start = 0;
        for(I i = 0; i < Arows; i++)
        {
            b_counter = 0;
            for(I j = 0; j < Bcols; j++)
            {
                a_counter = a_start;
                for(I k = 0; k < Brows; k++)
                {
                    //S[i,j] += A[i,k]*B[k,j]
                    Sx[s_counter] += Ax[a_counter]*Bx[b_counter];
                    a_counter++; b_counter++;
                }
                s_counter++;
            }
            a_start += Acols;
        }
    }
    else if((Strans == 'F') && (Btrans == 'T'))
    {
        // A is in row major, B is in row major, so compute
        // S(i,j) = A(i,:) B(:,j) with the SMMP algorithm

        I a_counter = 0;

        // Loop over rows of A
        for(I i = 0; i < Arows; i++)
        {
            // Loop over columns in row i of A
            for(I j = 0; j < Acols; j++)
            {
                I s_counter = i*Scols;
                I b_counter = j*Bcols;

                // Loop over columns in row j of B
                for(I k = 0; k < Bcols; k++)
                {
                    // Accumulate A[i,j]*B[j,k] --> S[i,k]
                    Sx[s_counter] += Ax[a_counter]*Bx[b_counter];
                    b_counter++;
                    s_counter++;
                }
                a_counter++;
            }
        }
    }
    else {
        std::cout << "Unsupported combination of row/column major for dense multiplication.\n";
    }
}


/*
 * Compute the SVD of a matrix, Ax, using the Jacobi method.
 * Compute Ax = U S V.H
 *
 * Parameters
 * ----------
 * Ax : array
 *     m x n dense matrix, stored in col major form
 * U : array
 *     m x n dense matrix initialized to 0.0
 *     Passed in as Tx
 * V : array
 *     n x n dense matrix initialized to 0.0
 *     Passed in as Bx
 * S : array
 *     n x 1 dense matrix initialized to 0.0
 *     Passed in as Sx
 * m,n : int
 *      Dimensions of Ax, m > n.
 *
 * Returns
 * -------
 * Returns Ax = U S V.H
 * U, V, S are modified in place
 *
 * V : array
 *      Orthogonal n x n matrix, V, stored in col major
 * U : array
 *      Orthogonal m x n matrix, U, stored in col major
 * S : array
 *      Singular values
 * int : int
 *      Function return value,
        ==  =====================
 *      -1  error
 *       0  successful
 *       1  did not converge
        ==  =====================
 *
 * Notes
 * -----
 * The Jacobi method is used to compute the SVD.  Conceptually,
 * the Jacobi method applies successive Jacobi rotations, Q_i to
 * the system, Q_i^H Ax.H Ax Q_i.  Despite the normal equations
 * appearing here, the actual method can be quite accurate.
 * However, the method is slower than Golub-Reinsch for all
 * but very small matrices.  For larger matrices, use
 * scipy.linalg.pinv or pyamg.util.linalg.pinv_array.
 *
 * References
 * ----------
 * De Rijk, "A One-Sided Jacobi Algorithm for computing the singular value
 * decomposition on a vector computer", SIAM J Sci and Statistical Comp,
 * Vol 10, No 2, p 359-371, March 1989.
 *
 */
template<class I, class T, class F>
I svd_jacobi(const T Ax[], T Tx[], T Bx[], F Sx[], const I m, const I n)
{
    // Not implemented for m < n matrices
    if( m < n)
    {   return -1; }

    // Rename
    const T * A = Ax;
    T * U = Tx;
    T * V = Bx;
    F * S = Sx;

    // Hard code fast 1x1 SVD
    if ( (n==1) && (m==1) )
    {
        F normA = mynorm(A[0]);

        V[0] = 1.0;
        S[0] = normA;
        if(normA == 0.0)
        {   U[0] = 1.0; }
        else
        {   U[0] = A[0]/normA; }

        return 0;
    }

    // Workspace
    I i, j, k;
    I nsq = n*n;
    F normx;

    // Initialize the rotation counter and the sweep counter.
    I count = 1;
    I sweep = 0;

    // Always do at least  30 sweeps
    I sweepmax = std::max(15*n, 30);

    F tolerance = sqrt((F)m)*std::numeric_limits<F>::epsilon();

    // Set V to the identity matrix
    for(i = 0; i < nsq; i++)
    {   V[i] = 0.0;}
    for(i = 0; i < nsq; i+= (n+1) )
    {   V[i] = 1.0;}

    // Copy A to U, note that the stop address &(A[m*n])
    // should go one past the final element to be copied
    std::copy(&(A[0]), &(A[m*n]), &(U[0]));

    // Store the column error estimates in S, for use during the orthogonalization
    for (j = 0; j < n; j++)
    {
        // S[j] = eps*norm(A[:,j])
        norm(&(U[j*m]), m, normx);
        S[j] = std::numeric_limits<F>::epsilon()*normx;
    }

    // Orthogonalize U by plane rotations.
    while( (count > 0) && (sweep <= sweepmax) )
    {
        // Initialize rotation counter.
        count = n*(n - 1)/2;
        I jm = 0;
        I jn = 0;

        for(j=0; j < n - 1; j++)
        {
            I km = (j+1)*m;
            I kn = (j+1)*n;

            for (k = j + 1; k < n; k++)
            {
                F cos, abserr_a, abserr_b;
                T sin, neg_conj_sin;
                I sorted, orthog, noisya, noisyb;

                F a; norm(&(U[jm]), m, a);              // || U[:,j] ||
                F b; norm(&(U[km]), m, b);              // || U[:,k] ||
                T d = dot_prod(&(U[jm]), &(U[km]), m);  // <U[:,j], U[:,k]>
                F norm_d = mynorm(d);

                // test for columns j,k orthogonal, or dominant errors
                abserr_a = S[j];
                abserr_b = S[k];

                sorted = (a >= b);
                orthog = (norm_d <= tolerance*a*b);

                // Test to see if col a or b has become noise
                noisya = (a < abserr_a);
                noisyb = (b < abserr_b);

                // no need for rotations
                if(sorted && (orthog || noisya || noisyb))
                {
                    // if count ever = 0, then everything is sorted and orthogonal
                    // (or possibly just noise)
                    count--;
                }

                // swap cols ||     Handle 0 matrix case
                else if(!sorted   || ( (norm_d == 0.0) && (a==b)  ) )
                {
                    // Apply rotation matrix,
                    // [ 0.0  1.0 ]
                    // [-1.0  0.0 ]
                    // Basically, swap columns in U and V with one sign flip

                    S[j] = abserr_b;
                    S[k] = abserr_a;

                    // apply rotation by right multiplication to U
                    I koffset = km;
                    for(I joffset = jm; joffset < (jm + m); joffset++)
                    {
                        // for i = 0:m-1
                        //   U[i,j] =   0.0*U[i,j] - 1.0*U[i,k]
                        //   U[i,k] =   1.0*U[i,j] + 0.0*U[i,k]
                        const T Uij = U[joffset];
                        const T Uik = U[koffset];
                        U[joffset] = -Uik;
                        U[koffset] =  Uij;
                        koffset++;
                    }

                    // apply rotation by right multiplication to V
                    koffset = kn;
                    for(I joffset = jn; joffset < (jn + n); joffset++)
                    {
                        // for i = 0:n-1
                        //   V[i,j] =   0.0*V[i,j] - 1.0*V[i,k]
                        //   V[i,k] =   1.0*V[i,j] + 0.0*V[i,k]
                        const T Vij = V[joffset];
                        const T Vik = V[koffset];
                        V[joffset] = -Vik;
                        V[koffset] =  Vij;
                        koffset++;
                    }
                }

                // Carry out Jacobi Rotations to orthogonalize column's j and k in U
                else
                {
                    // calculate rotation angles for
                    // jacobi_rot = [cos          sin]
                    //              [-conj(sin)   cos]
                    F tau = (b*b - a*a)/(2.0*norm_d);
                    F t = signof(tau)/(fabs(tau) + sqrt(1.0 + tau*tau));
                    cos = 1.0/(sqrt(1.0 + t*t));
                    sin = d*(t*cos/norm_d);
                    neg_conj_sin = -conjugate(sin);

                    F norm_sin = mynorm(sin);
                    S[j] = fabs(cos)*abserr_a + norm_sin*abserr_b;
                    S[k] =  norm_sin*abserr_a + fabs(cos)*abserr_b;

                    // apply rotation by right multiplication to U
                    I koffset = km;
                    for(I joffset = jm; joffset < (jm + m); joffset++)
                    {
                        // for i = 0:m-1
                        //   U[i,j] =   cos*U[i,j] + -conj(sin)*U[i,k]
                        //   U[i,k] =   sin*U[i,j] +       cos*U[i,k]
                        const T Uij = U[joffset];
                        const T Uik = U[koffset];
                        U[joffset] = Uij*cos + neg_conj_sin*Uik;
                        U[koffset] = sin*Uij + Uik*cos;
                        koffset++;
                    }

                    // apply rotation by right multiplication to V
                    koffset = kn;
                    for(I joffset = jn; joffset < (jn + n); joffset++)
                    {
                        // for i = 0:n-1
                        //   V[i,j] =   cos*V[i,j] + -conj(sin)*V[i,k]
                        //   V[i,k] =   sin*V[i,j] +       cos*V[i,k]
                        const T Vij = V[joffset];
                        const T Vik = V[koffset];
                        V[joffset] = Vij*cos + neg_conj_sin*Vik;
                        V[koffset] = sin*Vij + Vik*cos;
                        koffset++;
                    }
                }

                km += m;
                kn += n;
            } // end k loop

            jm += m;
            jn += n;
        } // end j loop

        //Sweep completed.
        sweep++;

    }// end while loop

    // Orthogonalization complete. Compute singular values.
    F sigma_tol=0.0;
    I Uoffset = 0;
    I iszero = n;
    for (j = 0; j < n; j++)
    {
        F curr_norm;
        norm(&(U[Uoffset]), m, curr_norm);              // || U[:,j] ||

        if(j == 0)
        {
            // For j==0, curr_norm is sigma_max
            F alpha = 50.0/sqrt(sqrt(std::numeric_limits<F>::epsilon()));
            sigma_tol = alpha*curr_norm*std::numeric_limits<F>::epsilon();
        }

        // Determine if singular value is zero
        if( curr_norm <= sigma_tol )
        {
            iszero--;                               // detect all zero matrix
            S[j] = 0.0;                             // singular
            for(i = Uoffset; i < (Uoffset + m); i++)
            {   U[i] = 0.0; }                       // annihilate U[:,j]
        }
        else
        {
            S[j] = curr_norm;                       // non-singular
            for(i = Uoffset; i < (Uoffset + m); i++)
            {   U[i] = U[i]/curr_norm; }            // normalize column U[:,j]
        }

        Uoffset += m;
    }

    if(iszero == 0)
    {
        // Set U and V to the identity matrix
        for(i = 0; i < nsq; i++)
        {   V[i] = 0.0;}
        for(i = 0; i < nsq; i+= (n+1) )
        {   V[i] = 1.0;}

        // U is already 0.0
        for(i = 0; i < n*m; i+= (m+1) )
        {   U[i] = 1.0;}

        return 0;
    }

    if(count > 0)
    {
        // reached sweep limit, i.e. did not converge
        return 1;
    }

    return 0;
}


/*
 * Solve a system with the SVD, i.e. use a robust Moore-Penrose
 * Pseudoinverse to multiply the RHS
 *
 * Parameters
 * ----------
 * A : array
 *     m x n dense column major array, m>n
 * m,n : int
 *     Dimensions of A, m > n
 * b : array
 *     RHS, m-vector
 * sing_vals : {float array}
 *     Holds the singular values upon return
 * work : array
 *     worksize array for temporary space for routine
 * worksize : int
 *     must be > m*n + n
 *
 * Returns
 * -------
 * A^{-1} b replaces b
 * sing_vals holds the singular values
 *
 * Notes
 * -----
 * forcing preallocation of sing_vals and work, allows for
 * efficient multiple calls to this routine
 *
 */
template<class I, class T, class F>
void svd_solve( T Ax[], I m, I n, T b[], F sing_vals[], T work[], I work_size)
{
    I mn = m*n;
    // Rename
    T * U = &(work[0]);
    T * V = &(work[mn]);
    T * x = &(work[2*mn]);
    const char trans = 'F';

    // calculate SVD
    I check = svd_jacobi(&(Ax[0]), &(U[0]), &(V[0]), &(sing_vals[0]), m, n);
    if (check == 1) {
        std::cout << "Warning: SVD iterations did not converge.\n";
    }
    else if (check != 0) {
        std::cout << "Warning: Error in computing SVD\n";
    }

    // Forming conjugate(U.T) in row major requires just
    // conjugating the current entries of U in col major
    for(I i = 0; i < m*n; i++)
    {   U[i] = conjugate(U[i]); }

    // A^{-1} b = V*Sinv*U.H*b, in 3 steps
    // Step 1, U.H*b
    gemm(&(U[0]), n, m, trans, &(b[0]), m, 1, trans,
         &(x[0]), n, 1, trans, 'T');

    // Step 2, scale x by Sinv
    for(I j = 0; j < n; j++)
    {
        if(sing_vals[j] != 0.0)
        {   x[j] = x[j]/sing_vals[j]; }
        else
        {   x[j] = 0.0; }
    }

    // Step 3, multiply by V
    // transpose V so that it is in row major for gemm
    transpose(&(V[0]), &(U[0]), n, n);
    gemm(&(U[0]), n, n, trans, &(x[0]), n, 1, trans,
         &(b[0]), n, 1, trans, 'T');

    return;
}

/* Replace each block of A with a Moore-Penrose pseudoinverse of that block.
 * Routine is designed to invert many small matrices at once.
 *
 * Parameters
 * ----------
 * AA : array
 *     (m, n, n) array, assumed to be "raveled" and in row major form
 * m,n : int
 *     dimensions of AA
 * TransA : char
 *     'T' or 'F'.  Decides whether to transpose each nxn block
 *     of A before inverting.  If using Python array, should be 'T'.
 *
 * Returns
 * -------
 * AA : array
 *     AA is modified in place with the pseduoinverse replacing each
 *     block of AA.  AA is returned in row-major form for Python
 *
 * Notes
 * -----
 * This routine is designed to be called once for a large m.
 * Calling this routine repeatably would not be efficient.
 *
 * This function offers substantial speedup over native Python
 * code for many small matrices, e.g. 5x5 and 10x10.  Tests have
 * indicated that matrices larger than 27x27 are faster if done
 * in native Python.
 *
 * Examples
 * --------
 * >>> from pyamg.amg_core import pinv_array
 * >>> from scipy import arange, ones, array, dot
 * >>> A = array([arange(1,5, dtype=float).reshape(2,2), ones((2,2),dtype=float)])
 * >>> Ac = A.copy()
 * >>> pinv_array(A, 2, 2, 'T')
 * >>> print "Multiplication By Inverse\n" + str(dot(A[0], Ac[0]))
 * >>> print "Multiplication by PseudoInverse\n" + str(dot(Ac[1], dot(A[1], Ac[1])))
 * >>>
 * >>> A = Ac.copy()
 * >>> pinv_array(A,2,2,'F')
 * >>> print "Changing flag to \'F\' results in different Inverse\n" + str(dot(A[0], Ac[0]))
 * >>> print "A holds the inverse of the transpose\n" + str(dot(A[0], Ac[0].T))
 *
 */
template<class I, class T, class F>
void pinv_array(T AA[], const int AA_size,
                const I m, const I n, const char TransA)
{
    I nsq = n*n;
    I Acounter = 0;
    T * Tran = new T[nsq];
    T * U = new T[nsq];
    T * V = new T[nsq];
    T * SinvUh = new T[nsq];
    F * S = new F[n];
    const char t = 'F';

    for(I i = 0; i < m; i++)
    {
        if(TransA == 'T')
        {   // transpose block of A so that it is in col major for SVD
            transpose(&(AA[Acounter]), &(Tran[0]), n, n);

            // calculate SVD
            svd_jacobi(&(Tran[0]), &(U[0]), &(V[0]), &(S[0]), n, n);
        }
        else
        {
            // calculate SVD
            svd_jacobi(&(AA[Acounter]), &(U[0]), &(V[0]), &(S[0]), n, n);
        }

        // invert S
        for(I j = 0; j < n; j++)
        {
            if(S[j] != 0.0)
            {   S[j] = 1.0/S[j]; }
        }

        // Sinv*conjugate(U.T), stored in column major form
        I counter = 0;
        for(I j = 0; j < n; j++)        // col of Uh
        {
            I Uoffset = j;
            for(I k = 0; k < n; k++)    // row of Uh
            {
                //Sinv(j) * conj(U(j,k)) ==> SinvUh(k,j)
                SinvUh[counter] = conjugate(U[Uoffset])*S[k];
                counter++;
                Uoffset+=n;
            }
        }

        // transpose V so that it is in row major for gemm
        transpose(&(V[0]), &(Tran[0]), n, n);

        // A^{-1} = V*SinvUh
        gemm(&(Tran[0]), n, n, t, &(SinvUh[0]), n, n, t,
             &(AA[Acounter]), n, n, t, 'T');

        Acounter += nsq;
    }

    delete[] Tran;
    delete[] U;
    delete[] V;
    delete[] S;
    delete[] SinvUh;

    return;
}

/*
 * Scale the columns of a CSC matrix *in place*
 *
 * ..
 *   A[:,i] *= X[i]
 *
 * References
 * ----------
 * https://github.com/scipy/scipy/blob/master/scipy/sparse/sparsetools/csr.h
 *
 */
template <class I, class T>
void csc_scale_columns(const I n_row,
                       const I n_col,
                       const I Ap[], const int Ap_size,
                       const I Aj[], const int Aj_size,
                             T Ax[], const int Ax_size,
                       const T Xx[], const int Xx_size)
{
    for(I i = 0; i < n_col; i++){
        for(I jj = Ap[i]; jj < Ap[i+1]; jj++){
            Ax[jj] *= Xx[i];
        }
    }
}

/*
 * Scale the rows of a CSC matrix *in place*
 *
 * ..
 *   A[i,:] *= X[i]
 *
 * References
 * ----------
 * https://github.com/scipy/scipy/blob/master/scipy/sparse/sparsetools/csr.h
 *
 */
template <class I, class T>
void csc_scale_rows(const I n_row,
                    const I n_col,
                    const I Ap[], const int Ap_size,
                    const I Aj[], const int Aj_size,
                          T Ax[], const int Ax_size,
                    const T Xx[], const int Xx_size)
{
    const I nnz = Ap[n_col];
    for(I i = 0; i < nnz; i++){
        Ax[i] *= Xx[Aj[i]];
    }
}


/*
 * Filter matrix rows by diagonal entry, that is set A_ij = 0 if::
 *
 *    |A_ij| < theta * |A_ii|
 *
 * Parameters
 * ----------
 * num_rows : int
 *     number of rows in A
 * theta : float
 *     stength of connection tolerance
 * Ap : array
 *     CSR row pointer
 * Aj : array
 *     CSR index array
 * Ax : array
 *     CSR data array
 *
 * Returns
 * -------
 * Nothing, Ax is modified in place
 */
template<class I, class T, class F>
void filter_matrix_rows(const I n_row,
                        const F theta,
                        const I Ap[], const int Ap_size,
                        const I Aj[], const int Aj_size,
                              T Ax[], const int Ax_size,
                        const bool lump)
{
    // Lump each row by setting A_ii += A_ij for all j s.t. |A_ij| < theta*|A_ii|,
    // and set A_ij = 0
    if (lump) {
        for(I i = 0; i < n_row; i++) {
            F diagonal = 0.0;

            const I row_start = Ap[i];
            const I row_end   = Ap[i+1];

            // Find diagonal of this row
            I diag_ind = -1;
            for(I jj = row_start; jj < row_end; jj++){
                if(Aj[jj] == i){
                    diag_ind = jj;
                    diagonal = mynorm(Ax[jj]);
                    break;
                }
            }

            // Set threshold for strong connections
            F threshold = theta*diagonal;
            for(I jj = row_start; jj < row_end; jj++){
                F norm_jj = mynorm(Ax[jj]);

                // Remove entry if below threshold
                if(norm_jj < threshold && Aj[jj] != i){
                    Ax[diag_ind] += Ax[jj];
                    Ax[jj] = 0.0;
                }
            }
        }
    }
    // Filter each row by setting explicit zeros when |A_ij| < theta*|A_ii|
    else {
        for(I i = 0; i < n_row; i++) {
            F diagonal = 0.0;

            const I row_start = Ap[i];
            const I row_end   = Ap[i+1];

            // Find diagonal of this row
            for(I jj = row_start; jj < row_end; jj++){
                if(Aj[jj] == i){
                    diagonal = mynorm(Ax[jj]);
                    break;
                }
            }

            // Set threshold for strong connections
            F threshold = theta*diagonal;
            for(I jj = row_start; jj < row_end; jj++){
                F norm_jj = mynorm(Ax[jj]);

                // Remove entry if below threshold
                if(norm_jj < threshold){
                    Ax[jj] = 0.0;
                }
            }
        }
    }
}

/* QR-decomposition using Householer transformations on dense
 * 2d array stored in either column- or row-major form. 
 * 
 * Parameters
 * ----------
 * A : double array
 *     2d matrix A stored in 1d column- or row-major.
 * m : &int
 *     Number of rows in A
 * n : &int
 *     Number of columns in A
 * is_col_major : bool
 *     True if A is stored in column-major, false
 *     if A is stored in row-major.
 *
 * Returns
 * -------
 * Q : vector<double>
 *     Matrix Q stored in same format as A.
 * R : in-place
 *     R is stored over A in place, in same format.
 * 
 * Notes
 * ------
 * Currently only set up for real-valued matrices. May easily
 * generalize to complex, but haven't checked.
 *
 */
template<class I, class T>
std::vector<T> QR(T A[],
                  const I &m,
                  const I &n,
                  const I is_col_major)
{
    // Funciton pointer for row or column major matrices
    I (*get_ind)(const I, const I, const I);
    const I *C;
    if (is_col_major) {
        get_ind = &col_major;
        C = &m;
    }
    else {
        get_ind = &row_major;
        C = &n;
    }

    // Initialize Q to identity
    std::vector<T> Q(m*m,0);
    for (I i=0; i<m; i++) {
        Q[get_ind(i,i,m)] = 1;
    }

    // Loop over columns of A using Householder reflections
    for (I j=0; j<n; j++) {

        // Break loop for short fat matrices
        if (m <= j) {
            break;
        }

        // Get norm of next column of A to be reflected. Choose sign
        // opposite that of A_jj to avoid catastrophic cancellation.
        // Skip loop if norm is zero, as that means column of A is all
        // zero.
        T normx = 0;
        for (I i=j; i<m; i++) {
            T temp = A[get_ind(i,j,*C)];
            normx += temp*temp;
        }
        normx = std::sqrt(normx);
        if (normx < 1e-12) {
            continue;
        }
        normx *= -1*signof(A[get_ind(j,j,*C)]);

        // Form vector v for Householder matrix H = I - tau*vv^T
        // where v = R(j:end,j) / scale, v[0] = 1.
        T scale = A[get_ind(j,j,*C)] - normx;
        T tau = -scale / normx;
        std::vector<T> v(m-j,0);
        v[0] = 1;
        for (I i=1; i<(m-j); i++) {
            v[i] = A[get_ind(j+i,j,*C)] / scale;    
        }

        // Modify R in place, R := H*R, looping over columns then rows
        for (I k=j; k<n; k++) {

            // Compute the kth element of v^T * R
            T vtR_k = 0;
            for (I i=0; i<(m-j); i++) {
                vtR_k += v[i] * A[get_ind(j+i,k,*C)];
            }

            // Correction for each row of kth column, given by 
            // R_ik -= tau * v_i * (vtR_k)_k
            for (I i=0; i<(m-j); i++) {
                A[get_ind(j+i,k,*C)] -= tau * v[i] * vtR_k;
            }
        }

        // Modify Q in place, Q = Q*H
        for (I i=0; i<m; i++) {

            // Compute the ith element of Q * v
            T Qv_i = 0;
            for (I k=0; k<(m-j); k++) {
                Qv_i += v[k] * Q[get_ind(i,k+j,m)];
            }

            // Correction for each column of ith row, given by
            // Q_ik -= tau * Qv_i * v_k
            for (I k=0; k<(m-j); k++) { 
                Q[get_ind(i,k+j,m)] -= tau * v[k] * Qv_i;
            }
        }
    }

    return Q;
}

/* Backward substitution solve on upper-triangular linear system,
 * Rx = rhs, where R is stored in column- or row-major form. 
 * 
 * Parameters
 * ----------
 * R : double array, length m*n
 *     Upper-triangular array stored in column- or row-major.
 * rhs : double array, length m
 *     Right hand side of linear system
 * x : double array, length n
 *     Preallocated array for solution
 * m : &int
 *     Number of rows in R
 * n : &int
 *     Number of columns in R
 * is_col_major : bool
 *     True if R is stored in column-major, false
 *     if R is stored in row-major.
 *
 * Returns
 * -------
 * Nothing, solution is stored in x[].
 *
 * Notes
 * -----
 * R need not be square, the system will be solved over the
 * upper-triangular block of size min(m,n). If remaining entries
 * insolution are unused, they will be set to zero. If a zero
 * is encountered on the ith diagonal, x[i] is set to zero. 
 *
 */        
template<class I, class T>
void upper_tri_solve(const T R[],
                     const T rhs[],
                     T x[],
                     const I m,
                     const I n,
                     const I is_col_major)
{
    // Function pointer for row or column major matrices
    I (*get_ind)(const I, const I, const I);
    const I *C;
    if (is_col_major) {
        get_ind = &col_major;
        C = &m;
    }
    else {
        get_ind = &row_major;
        C = &n;
    }

    // Backward substitution
    I rank = std::min(m,n);
    for (I i=(rank-1); i>=0; i--) {
        T temp = rhs[i];
        for (I j=(i+1); j<rank; j++) {
            temp -= R[get_ind(i,j,*C)]*x[j];
        }
        if (std::abs(R[get_ind(i,i,*C)]) < 1e-12) {
            x[i] = 0.0;
        }
        else {
            x[i] = temp / R[get_ind(i,i,*C)];            
        }
    }

    // If rank < size of rhs, set free elements in x to zero
    for (I i=m; i<n; i++) {
        x[i] = 0;
    }
}

/* Forward substitution solve on lower-triangular linear system,
 * Lx = rhs, where L is stored in column- or row-major form. 
 * 
 * Parameters
 * ----------
 * L : double array, length m*n
 *     Lower-triangular array stored in column- or row-major.
 * rhs : double array, length m
 *     Right hand side of linear system
 * x : double array, length n
 *     Preallocated array for solution
 * m : &int
 *     Number of rows in L
 * n : &int
 *     Number of columns in L
 * is_col_major : bool
 *     True if L is stored in column-major, false
 *     if L is stored in row-major.
 *
 * Returns
 * -------
 * Nothing, solution is stored in x[].
 *
 * Notes
 * -----
 * L need not be square, the system will be solved over the
 * lower-triangular block of size min(m,n). If remaining entries
 * in solution are unused, they will be set to zero. If a zero
 * is encountered on the ith diagonal, x[i] is set to zero. 
 *
 *
 */
template<class I, class T>
void lower_tri_solve(const T L[],
                     const T rhs[],
                     T x[],
                     const I &m,
                     const I &n,
                     const I is_col_major)
{
    // Function pointer for row or column major matrices
    I (*get_ind)(const I, const I, const I);
    const I *C;
    if (is_col_major) {
        get_ind = &col_major;
        C = &m;
    }
    else {
        get_ind = &row_major;
        C = &n;
    }

    // Forward substitution
    I rank = std::min(m,n);
    for (I i=0; i<rank; i++) {
        T temp = rhs[i];
        for (I j=0; j<i; j++) {
            temp -= L[get_ind(i,j,*C)]*x[j];
        }
        if (std::abs(L[get_ind(i,i,*C)]) < 1e-12) {
            x[i] = 0.0;
        }
        else{
            x[i] = temp / L[get_ind(i,i,*C)];
        }
    }

    // If rank < size of rhs, set free elements in x to zero
    for (I i=m; i<n; i++) {
        x[i] = 0;
    }
}

/* Method to solve the linear least squares problem.
 *
 * Parameters
 * ----------
 * A : double array, length m*n
 *     2d array stored in column- or row-major.
 * b : double array, length m
 *     Right hand side of unconstrained problem.
 * x : double array, length n
 *     Container for solution
 * m : &int
 *     Number of rows in A
 * n : &int
 *     Number of columns in A
 * is_col_major : bool
 *     True if A is stored in column-major, false
 *     if A is stored in row-major.
 *
 * Returns
 * -------
 * x : vector<double>
 *    Solution to constrained least sqaures problem.
 *
 * Notes
 * -----
 * If system is under determined, free entries are set to zero. 
 * Currently only set up for real-valued matrices. May easily
 * generalize to complex, but haven't checked.
 *
 */
template<class I, class T>
void least_squares(T A[],
                   T b[],
                   T x[],
                   const I &m,
                   const I &n,
                   const I is_col_major=0)
{
    // Function pointer for row or column major matrices
    I (*get_ind)(const I, const I, const I);
    if (is_col_major) {
        get_ind = &col_major;
    }
    else {
        get_ind = &row_major;
    }

    // Take QR of A
    std::vector<T> Q = QR(A,m,n,is_col_major);

    // Multiply right hand side, b:= Q^T*b. Have to make new vetor, rhs.
    std::vector<T> rhs(m,0);
    for (I i=0; i<m; i++) {
        for (I k=0; k<m; k++) {
            rhs[i] += b[k] * Q[get_ind(k,i,m)];
        }
    }

    // Solve upper triangular system, store solution in x.
    upper_tri_solve(A,&rhs[0],x,m,n,is_col_major);
}

#endif
