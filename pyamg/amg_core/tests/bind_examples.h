#include <iostream>

//
// Series of tests for cpp binding templates
//
//

// ---------------------------------------------------------------- // Docstring Tests
// ----------------------------------------------------------------

//
// Testing docstring
//
template <class I>
int test1(const I n)
{
    return 1;
}

// Testing docstring
template <class I>
int test2(const I n)
{
    return 1;
}

/*
 * Testing a docstring
 *
 */
template <class I>
int test3(const I n)
{
    return 1;
}

/* Testing a docstring */
template <class I>
int test4(const I n)
{
    return 1;
}

template <class I>
int test5(const I n)
{
    return 1;
}

/* Testing a docstring */
int test6(const int n)
{
    return 1;
}

int test7(const int n)
{
    return 1;
}

// ----------------------------------------------------------------
// Untemplated tests
// ----------------------------------------------------------------

// untemplated
int test8(const int n,
                int m,
            double* x, const int x_size,
              int J[], const int J_size) {
    x[0] = 7.5;
    J[0] = 7;

    return 1;
}

//
// templated functions and types
//

// some class
template <class I, class T, class F>
int test9(const I J[], const int J_size,
                T x[], const int x_size,
                F y[], const int y_size)
{
    F myval (7.5, 8.25);
    x[0] = 7.5;
    y[0] = myval;
    return J[0];
}

// This will test different instantiation types
// I : int32, bool
// T : float32, float64, complex32, complex64
template <class I, class T>
int test10(I J[], const int J_size,
           T x[], const int x_size)
{
    J[0] = 0;
    x[0] = 7.5;
    return 1;
}
