#include <iostream>

//
// Series of tests for cpp binding templates
//
//

// ----------------------------------------------------------------
// Docstring Tests
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
              int J[], const int J_size,
          double &val) {
    x[0] = 7.7;
    J[0] = 7;
    val  = 7.7;

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
    F myval (7.7, 8.8);
    x[0] = 7.7;
    y[0] = myval;
    return J[0];
}
