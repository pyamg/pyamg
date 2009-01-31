#ifndef LINALG_H
#define LINALG_H

#include <math.h>
#include <limits>

// sign function that assigns a sign of 1 to 0
inline int signof(int a) { return (a<0 ? -1 : 1); }
inline float signof(float a) { return (a<0.0 ? -1.0 : 1.0); }
inline double signof(double a) { return (a<0.0 ? -1.0 : 1.0); }

// Overloaded routines for complex arithmetic
inline float conjugate(const float& x)
    { return x; }
inline double conjugate(const double& x)
    { return x; }
inline npy_cfloat_wrapper conjugate(const npy_cfloat_wrapper& x)
    { return npy_cfloat_wrapper(x.real, -x.imag); }
inline npy_cdouble_wrapper conjugate(const npy_cdouble_wrapper& x)
    { return npy_cdouble_wrapper(x.real, -x.imag); }

inline float real(const float& x)
    { return x; }
inline double real(const double& x)
    { return x; }
inline float real(const npy_cfloat_wrapper& x)
    { return x.real; }
inline double real(const npy_cdouble_wrapper& x)
    { return x.real; }

inline float imag(const float& x)
    { return 0.0; }
inline double imag(const double& x)
    { return 0.0; }
inline float imag(const npy_cfloat_wrapper& x)
    { return x.imag; }
inline double imag(const npy_cdouble_wrapper& x)
    { return x.imag; }

inline float mynorm(const float& x)
    { return fabs(x); }
inline double mynorm(const double& x)
    { return fabs(x); }
inline float mynorm(const npy_cfloat_wrapper& x)
    { return sqrt(x.real*x.real + x.imag*x.imag); }
inline double mynorm(const npy_cdouble_wrapper& x)
    { return sqrt(x.real*x.real + x.imag*x.imag); }

inline float mynormsq(const float& x)
    { return (x*x); }
inline double mynormsq(const double& x)
    { return (x*x); }
inline float mynormsq(const npy_cfloat_wrapper& x)
    { return (x.real*x.real + x.imag*x.imag); }
inline double mynormsq(const npy_cdouble_wrapper& x)
    { return (x.real*x.real + x.imag*x.imag); }


//Dense Algebra Routines

/* dot(x, y, n)
 * x,y are n-vectors
 * calculate conjuate(x).T y
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
 * x is an n-vectors
 * calculate sqrt( <x, x> )
*/
template<class I, class T, class F>
inline void norm(const T x[], const I n, F &normx)
{
    normx = sqrt(real(dot_prod(x,x,n)));
}


/* axpy(x, y, alpha, n)
 * x, y are n-vectors
 * alpha is a constant scalar
 * calculate x = x + alpha*y
*/
template<class I, class T>
inline void axpy(T x[], const T y[], const T alpha, const I n)
{
    for( I i = 0; i < n; i++)
    {   x[i] += alpha*y[i]; }
}

/*
 * Compute A*B ==> S
 *
 * Parameters:
 * A      -  Left operand in row major
 * B      -  Right operand in column major
 * S      -  A*B, in row-major
 * Atrans -  Whether to transpose A before multiply
 * Btrans -  Whether to transpose B before multiply
 * Strans -  Whether to transpose S after multiply, Outputted in row-major         
 *
 * Returns:
 *  S = A*B
 *
 * Notes:
 *    Not fully implemented, 
 *    - Atrans and Btrans not implemented
 *    - No error checking on inputs
 *
 */

/*
 * transpose Ax by overwriting Bx
 * Ax is (m,n)
 * Bx is (n,m)
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

/*
 * A is row major
 * B is col major
 * Strans = 'T' gives S in col major
 * Strans = 'F' gives S in row major
 * Contents of S are overwritten
 */
template<class I, class T>
void gemm(const T Ax[], const I Arows, const I Acols, const char Atrans, 
          const T Bx[], const I Brows, const I Bcols, const char Btrans, 
          T Sx[], const I Srows, const I Scols, const char Strans)
{
    //Add checks for dimensions, but leaving them out speeds things up
    //Add functionality for transposes

    if(Strans == 'T')
    {
        I s_counter = 0; I a_counter =0; I b_counter =0; I a_start = 0;
        for(I i = 0; i < Arows; i++)
        {
            s_counter = i;
            b_counter = 0; 
            for(I j = 0; j < Bcols; j++)
            {
                Sx[s_counter] = 0.0;
                a_counter = a_start;
                for(I k = 0; k < Brows; k++)
                {
                    //S[i,j] += Ax[i,k]*B[k,j]
                    Sx[s_counter] += Ax[a_counter]*Bx[b_counter];
                    a_counter++; b_counter++;
                }
                s_counter+=Scols;
            }
            a_start += Acols;
        }
    }
    else if(Strans == 'F')
    {
        I s_counter = 0; I a_counter =0; I b_counter =0; I a_start = 0;
        for(I i = 0; i < Arows; i++)
        {
            b_counter = 0; 
            for(I j = 0; j < Bcols; j++)
            {
                Sx[s_counter] = 0.0;
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
}



/*
 * Compute the SVD of a matrix, A, using the Jacobi method.  See reference,
 * De Rijk, "A One-Sided Jacobi Algorithm for computing the singular value 
 * decomposition on a vector computer", SIAM J Sci and Statistical Comp,
 * Vol 10, No 2, p 359-371, March 1989. 
 *
 * Compute A = U S V.H, where S is diagonal and U and V are orthogonal
 * Input
 * -----
 * A        dense matrix, stored in col major form
 *          A is (m,n), m > n.
 * U        All 0.0 matrix of size (m, n)
 * V        All 0.0 matrix of size (n, n) 
 * S        All 0.0 vector of size max(m,n)
 * m, n     size of A, it must be that m > n
 *
 * Output
 * ------
 * V = V, i.e. not V.H, in col major
 * U = U, in col major
 * S holds the singular values
 *
 * returns int
 *    -1:  error
 *     0:  successful
 *     1:  did not converge
 */

template<class I, class T, class F>
I svd_jacobi (const T Ax[], T Tx[], T Bx[], F Sx[], const I m, const I n)
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
        
        return 0.0;
    }
  
    // Workspace
    I i, j, k;
    I nsq = n*n;
    F normx;

    // Initialize the rotation counter and the sweep counter.
    I count = 1;
    I sweep = 0;

    // Always do at least 12 sweeps
    I sweepmax = std::max(8*n, 12);

    F tolerance = 10.0*m*std::numeric_limits<F>::epsilon();

    // Set V to the identity matrix
    for(i = 0; i < nsq; i++)
    {   V[i] = 0.0;}
    for(i = 0; i < nsq; i+= (n+1) )
    {   V[i] = 1.0;}

    // Copy A to U, note that the stop address &(A[nsq]) 
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
                    continue;
                }
                
                // swap cols ||     Handle 0 matrix case
                if(!sorted   || ( (norm_d == 0.0) && (a==b)  ) )
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
                else
                {
                    // calculate rotation angles for 
                    // jacobi_rot = [cos          sin]
                    //              [-conj(sin)   cos]
                    F tau = (b*b - a*a)/(2.0*norm_d);
                    F t = signof(tau)/(fabs(tau) + sqrt(1.0 + tau*tau));
                    cos = 1.0/(sqrt(1.0 + t*t));
                    sin = d*(t*cos/norm_d);
                    neg_conj_sin = conjugate(sin)*-1.0;
                
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
    F prev_norm = -1.0;
    I Uoffset = 0;
    I iszero = n;
    for (j = 0; j < n; j++)
    {
        F curr_norm;
        norm(&(U[Uoffset]), m, curr_norm);              // || U[:,j] ||
        
        // Determine if singular value is zero, according to the
        // criteria used in the main loop above (i.e. comparison
        // with norm of previous column).
        if(curr_norm == 0.0 || prev_norm == 0.0 
            || (j > 0 && (curr_norm <= tolerance*prev_norm)) )
        {   
            iszero--;                               // detect all zero matrix
            S[j] = 0.0;                             // singular
            for(i = Uoffset; i < (Uoffset + m); i++)
            {   U[i] = 0.0; }                       // annihilate U[:,j]
            prev_norm = 0.0;
        }
        else
        {
            S[j] = curr_norm;                       // non-singular
            for(i = Uoffset; i < (Uoffset + m); i++)
            {   U[i] = U[i]/curr_norm; }            // normalize column U[:,j]
            prev_norm = curr_norm;
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
 * Solve a system with the SVD, i.e. use a robust pseudo-inverse
 * to multiply the RHS
 * Input:
 * A is (m,n), in column major, m>n
 * b is RHS and is m-vector
 * sing_vals holds the singular values upon return
 * work is a worksize array so that we avoid reallocating 
 *      memory on the heap for multiple calls to svd_solve
 * worksize must be > m*n + n
 *
 * Output:
 * A^{-1} b replaces b
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
    svd_jacobi(&(Ax[0]), &(U[0]), &(V[0]), &(sing_vals[0]), n, n);
        
    // Forming conjugate(U.T) in row major requires just
    // conjugating the current entries of U in col major
    for(I i = 0; i < m*n; i++) 
    {   U[i] = conjugate(U[i]); }

    // A^{-1} b = V*Sinv*U.H*b, in 3 steps
    // Step 1, U.H*b
    gemm(&(U[0]), n, n, trans, &(b[0]), n, 1, trans,  
         &(x[0]), n, 1, trans);

    // Setp 2, scale x by Sinv
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
         &(b[0]), n, 1, trans);
    
    return;
}

/*
 * Ax is (m, n, n), and is assumed to be "raveled" and in row major form
 * 
 * Replace each block of A with a moore-penrose pseudo inverse of that block
 * The pseudo inverse, however, will be in row major, so python will immediately
 * be able to use it
 *
 * TransA='T' forces a transpose of each block of A.  
 * This is needed if calling pinv_array from python.
 *
 * Routine is designed to be called once for a large m.  Calling this routine repeatably 
 * would not be efficient
 */
template<class I, class T, class F>
void pinv_array(T Ax[], const I m, const I n, const char TransA)
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
            transpose(&(Ax[Acounter]), &(Tran[0]), n, n); 
            
            // calculate SVD
            svd_jacobi(&(Tran[0]), &(U[0]), &(V[0]), &(S[0]), n, n);
        }
        else
        {
            // calculate SVD
            svd_jacobi(&(Ax[Acounter]), &(U[0]), &(V[0]), &(S[0]), n, n);
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
             &(Ax[Acounter]), n, n, t);

        Acounter += nsq;
    }
    
    delete[] Tran;
    delete[] U;
    delete[] V;
    delete[] S;
    delete[] SinvUh;
    
    return;
}


#endif
