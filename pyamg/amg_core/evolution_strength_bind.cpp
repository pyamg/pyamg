// DO NOT EDIT: this file is generated

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>

#include "evolution_strength.h"

namespace py = pybind11;

template<class I, class T>
void _apply_absolute_distance_filter(
            const I n_row,
          const T epsilon,
      py::array_t<I> & Sp,
      py::array_t<I> & Sj,
      py::array_t<T> & Sx
                                     )
{
    auto py_Sp = Sp.unchecked();
    auto py_Sj = Sj.unchecked();
    auto py_Sx = Sx.mutable_unchecked();
    const I *_Sp = py_Sp.data();
    const I *_Sj = py_Sj.data();
    T *_Sx = py_Sx.mutable_data();

    return apply_absolute_distance_filter<I, T>(
                    n_row,
                  epsilon,
                      _Sp, Sp.shape(0),
                      _Sj, Sj.shape(0),
                      _Sx, Sx.shape(0)
                                                );
}

template<class I, class T>
void _apply_distance_filter(
            const I n_row,
          const T epsilon,
      py::array_t<I> & Sp,
      py::array_t<I> & Sj,
      py::array_t<T> & Sx
                            )
{
    auto py_Sp = Sp.unchecked();
    auto py_Sj = Sj.unchecked();
    auto py_Sx = Sx.mutable_unchecked();
    const I *_Sp = py_Sp.data();
    const I *_Sj = py_Sj.data();
    T *_Sx = py_Sx.mutable_data();

    return apply_distance_filter<I, T>(
                    n_row,
                  epsilon,
                      _Sp, Sp.shape(0),
                      _Sj, Sj.shape(0),
                      _Sx, Sx.shape(0)
                                       );
}

template<class I, class T>
void _min_blocks(
         const I n_blocks,
        const I blocksize,
      py::array_t<T> & Sx,
      py::array_t<T> & Tx
                 )
{
    auto py_Sx = Sx.unchecked();
    auto py_Tx = Tx.mutable_unchecked();
    const T *_Sx = py_Sx.data();
    T *_Tx = py_Tx.mutable_data();

    return min_blocks<I, T>(
                 n_blocks,
                blocksize,
                      _Sx, Sx.shape(0),
                      _Tx, Tx.shape(0)
                            );
}

template<class I, class T, class F>
void _evolution_strength_helper(
      py::array_t<T> & Sx,
      py::array_t<I> & Sp,
      py::array_t<I> & Sj,
            const I nrows,
       py::array_t<T> & x,
       py::array_t<T> & y,
       py::array_t<T> & b,
          const I BDBCols,
          const I NullDim,
              const F tol
                                )
{
    auto py_Sx = Sx.mutable_unchecked();
    auto py_Sp = Sp.unchecked();
    auto py_Sj = Sj.unchecked();
    auto py_x = x.unchecked();
    auto py_y = y.unchecked();
    auto py_b = b.unchecked();
    T *_Sx = py_Sx.mutable_data();
    const I *_Sp = py_Sp.data();
    const I *_Sj = py_Sj.data();
    const T *_x = py_x.data();
    const T *_y = py_y.data();
    const T *_b = py_b.data();

    return evolution_strength_helper<I, T, F>(
                      _Sx, Sx.shape(0),
                      _Sp, Sp.shape(0),
                      _Sj, Sj.shape(0),
                    nrows,
                       _x, x.shape(0),
                       _y, y.shape(0),
                       _b, b.shape(0),
                  BDBCols,
                  NullDim,
                      tol
                                              );
}

template<class I, class T, class F>
void _incomplete_mat_mult_csr(
      py::array_t<I> & Ap,
      py::array_t<I> & Aj,
      py::array_t<T> & Ax,
      py::array_t<I> & Bp,
      py::array_t<I> & Bj,
      py::array_t<T> & Bx,
      py::array_t<I> & Sp,
      py::array_t<I> & Sj,
      py::array_t<T> & Sx,
         const I num_rows
                              )
{
    auto py_Ap = Ap.unchecked();
    auto py_Aj = Aj.unchecked();
    auto py_Ax = Ax.unchecked();
    auto py_Bp = Bp.unchecked();
    auto py_Bj = Bj.unchecked();
    auto py_Bx = Bx.unchecked();
    auto py_Sp = Sp.unchecked();
    auto py_Sj = Sj.unchecked();
    auto py_Sx = Sx.mutable_unchecked();
    const I *_Ap = py_Ap.data();
    const I *_Aj = py_Aj.data();
    const T *_Ax = py_Ax.data();
    const I *_Bp = py_Bp.data();
    const I *_Bj = py_Bj.data();
    const T *_Bx = py_Bx.data();
    const I *_Sp = py_Sp.data();
    const I *_Sj = py_Sj.data();
    T *_Sx = py_Sx.mutable_data();

    return incomplete_mat_mult_csr<I, T, F>(
                      _Ap, Ap.shape(0),
                      _Aj, Aj.shape(0),
                      _Ax, Ax.shape(0),
                      _Bp, Bp.shape(0),
                      _Bj, Bj.shape(0),
                      _Bx, Bx.shape(0),
                      _Sp, Sp.shape(0),
                      _Sj, Sj.shape(0),
                      _Sx, Sx.shape(0),
                 num_rows
                                            );
}

PYBIND11_MODULE(evolution_strength, m) {
    m.doc() = R"pbdoc(
    Pybind11 bindings for evolution_strength.h

    Methods
    -------
    apply_absolute_distance_filter
    apply_distance_filter
    min_blocks
    evolution_strength_helper
    incomplete_mat_mult_csr
    )pbdoc";

    py::options options;
    options.disable_function_signatures();

    m.def("apply_absolute_distance_filter", &_apply_absolute_distance_filter<int, float>,
        py::arg("n_row"), py::arg("epsilon"), py::arg("Sp").noconvert(), py::arg("Sj").noconvert(), py::arg("Sx").noconvert());
    m.def("apply_absolute_distance_filter", &_apply_absolute_distance_filter<int, double>,
        py::arg("n_row"), py::arg("epsilon"), py::arg("Sp").noconvert(), py::arg("Sj").noconvert(), py::arg("Sx").noconvert(),
R"pbdoc(
Return a filtered strength-of-connection matrix by applying a drop tolerance.

Strength values are assumed to be "distance"-like, i.e. the smaller the
value the stronger the connection.
Strength values are _Not_ evaluated relatively, i.e. an off-diagonal
entry A[i,j] is a strong connection iff::

    S[i,j] <= epsilon,   where k != i

Also, set the diagonal to 1.0, as each node is perfectly close to itself.

Parameters
----------
n_row : {int}
     Dimension of matrix, S
epsilon : {float}
     Drop tolerance
Sp : {int array}
     Row pointer array for CSR matrix S
Sj : {int array}
     Col index array for CSR matrix S
Sx : {float|complex array}
     Value array for CSR matrix S

Returns
-------
Sx : {float|complex array}
     Modified in place such that the above dropping strategy has been applied
     There will be explicit zero entries for each weak connection

Notes
-----
Principle calling routines are strength of connection routines, e.g.,
`distance_strength_of_connection`

Examples
--------
>>> from scipy.sparse import csr_matrix
>>> from pyamg.amg_core import apply_absolute_distance_filter
>>> from scipy import array
>>> # Graph in CSR where entries in row i represent distances from dof i
>>> indptr = array([0,3,6,9])
>>> indices = array([0,1,2,0,1,2,0,1,2])
>>> data = array([1.,2.,3.,4.,1.,2.,3.,9.,1.])
>>> S = csr_matrix( (data,indices,indptr), shape=(3,3) )
>>> print "Matrix Before Applying Filter\n" + str(S.todense())
>>> apply_absolute_distance_filter(3, 1.9, S.indptr, S.indices, S.data)
>>> print "Matrix After Applying Filter\n" + str(S.todense()))pbdoc");

    m.def("apply_distance_filter", &_apply_distance_filter<int, float>,
        py::arg("n_row"), py::arg("epsilon"), py::arg("Sp").noconvert(), py::arg("Sj").noconvert(), py::arg("Sx").noconvert());
    m.def("apply_distance_filter", &_apply_distance_filter<int, double>,
        py::arg("n_row"), py::arg("epsilon"), py::arg("Sp").noconvert(), py::arg("Sj").noconvert(), py::arg("Sx").noconvert(),
R"pbdoc(
Return a filtered strength-of-connection matrix by applying a drop tolerance
 Strength values are assumed to be "distance"-like, i.e. the smaller the
 value the stronger the connection

   An off-diagonal entry A[i,j] is a strong connection iff

           S[i,j] <= epsilon * min( S[i,k] )   where k != i

  Also, set the diagonal to 1.0, as each node is perfectly close to itself

Parameters
----------
n_row : {int}
     Dimension of matrix, S
epsilon : {float}
     Drop tolerance
Sp : {int array}
     Row pointer array for CSR matrix S
Sj : {int array}
     Col index array for CSR matrix S
Sx : {float|complex array}
     Value array for CSR matrix S

Returns
-------
Sx : {float|complex array}
     Modified in place such that the above dropping strategy has been applied
     There will be explicit zero entries for each weak connection

Notes
-----
Principle calling routines are strength of connection routines, e.g.
evolution_strength_of_connection(...) in strength.py.  It is used to apply
a drop tolerance.

Examples
--------
>>> from scipy.sparse import csr_matrix
>>> from pyamg.amg_core import apply_distance_filter
>>> from scipy import array
>>> # Graph in CSR where entries in row i represent distances from dof i
>>> indptr = array([0,3,6,9])
>>> indices = array([0,1,2,0,1,2,0,1,2])
>>> data = array([1.,2.,3.,4.,1.,2.,3.,9.,1.])
>>> S = csr_matrix( (data,indices,indptr), shape=(3,3) )
>>> print "Matrix BEfore Applying Filter\n" + str(S.todense())
>>> apply_distance_filter(3, 1.9, S.indptr, S.indices, S.data)
>>> print "Matrix AFter Applying Filter\n" + str(S.todense()))pbdoc");

    m.def("min_blocks", &_min_blocks<int, float>,
        py::arg("n_blocks"), py::arg("blocksize"), py::arg("Sx").noconvert(), py::arg("Tx").noconvert());
    m.def("min_blocks", &_min_blocks<int, double>,
        py::arg("n_blocks"), py::arg("blocksize"), py::arg("Sx").noconvert(), py::arg("Tx").noconvert(),
R"pbdoc(
Given a BSR with num_blocks stored, return a linear array of length
 num_blocks, which holds each block's smallest, nonzero, entry

Parameters
----------
n_blocks : {int}
     Number of blocks in matrix
blocksize : {int}
     Size of each block
Sx : {float|complex array}
     Block data structure of BSR matrix, S
     Sx is (n_blocks x blocksize) in length
Tx : {float|complex array}
     modified in place for output

Returns
-------
Tx : {float|complex array}
     Modified in place; Tx[i] holds the minimum nonzero value of block i of S

Notes
-----
Principle calling routine is evolution_strength_of_connection(...) in strength.py.
In that routine, it is used to assign a strength of connection value between
supernodes by setting the strength value to be the minimum nonzero in a block.

Examples
--------
>>> from scipy.sparse import bsr_matrix, csr_matrix
>>> from pyamg.amg_core import min_blocks
>>> from numpy import zeros, array, ravel, round
>>> from scipy import rand
>>> row  = array([0,2,4,6])
>>> col  = array([0,2,2,0,1,2])
>>> data = round(10*rand(6,2,2), decimals=1)
>>> S = bsr_matrix( (data,col,row), shape=(6,6) )
>>> T = zeros(data.shape[0])
>>> print "Matrix BEfore\n" + str(S.todense())
>>> min_blocks(6, 4, ravel(S.data), T)
>>> S2 = csr_matrix((T, S.indices, S.indptr), shape=(3,3))
>>> print "Matrix AFter\n" + str(S2.todense()))pbdoc");

    m.def("evolution_strength_helper", &_evolution_strength_helper<int, float, float>,
        py::arg("Sx").noconvert(), py::arg("Sp").noconvert(), py::arg("Sj").noconvert(), py::arg("nrows"), py::arg("x").noconvert(), py::arg("y").noconvert(), py::arg("b").noconvert(), py::arg("BDBCols"), py::arg("NullDim"), py::arg("tol"));
    m.def("evolution_strength_helper", &_evolution_strength_helper<int, double, double>,
        py::arg("Sx").noconvert(), py::arg("Sp").noconvert(), py::arg("Sj").noconvert(), py::arg("nrows"), py::arg("x").noconvert(), py::arg("y").noconvert(), py::arg("b").noconvert(), py::arg("BDBCols"), py::arg("NullDim"), py::arg("tol"));
    m.def("evolution_strength_helper", &_evolution_strength_helper<int, std::complex<float>, float>,
        py::arg("Sx").noconvert(), py::arg("Sp").noconvert(), py::arg("Sj").noconvert(), py::arg("nrows"), py::arg("x").noconvert(), py::arg("y").noconvert(), py::arg("b").noconvert(), py::arg("BDBCols"), py::arg("NullDim"), py::arg("tol"));
    m.def("evolution_strength_helper", &_evolution_strength_helper<int, std::complex<double>, double>,
        py::arg("Sx").noconvert(), py::arg("Sp").noconvert(), py::arg("Sj").noconvert(), py::arg("nrows"), py::arg("x").noconvert(), py::arg("y").noconvert(), py::arg("b").noconvert(), py::arg("BDBCols"), py::arg("NullDim"), py::arg("tol"),
R"pbdoc(
Create strength-of-connection matrix based on constrained min problem of
   min( z - B*x ), such that
      (B*x)|_i = z|_i, i.e. they are equal at point i
       z = (I - (t/k) Dinv A)^k delta_i

Strength is defined as the relative point-wise approximation error between
B*x and z.  B is the near-nullspace candidates.  The constrained min problem
is also restricted to consider B*x and z only at the nonzeros of column i of A

Can use either the D_A inner product, or l2 inner-prod in the minimization
problem. Using D_A gives scale invariance.  This choice is reflected in
whether the parameter DB = B or diag(A)*B

This is a quadratic minimization problem with a linear constraint, so
we can build a linear system and solve it to find the critical point, i.e. minimum.

Parameters
----------
Sp : {int array}
     Row pointer array for CSR matrix S
Sj : {int array}
     Col index array for CSR matrix S
Sx : {float|complex array}
     Value array for CSR matrix S.
     Upon entry to the routine, S = (I - (t/k) Dinv A)^k
nrows : {int}
     Dimension of S
B : {float|complex array}
     nrows x NullDim array of near nullspace vectors in col major form,
     if calling from within Python, take a transpose.
DB : {float|complex array}
     nrows x NullDim array of possibly scaled near nullspace
     vectors in col major form.  If calling from within Python, take a
     transpose.  For a scale invariant measure,
     DB = diag(A)*conjugate(B)), corresponding to the D_A inner-product
     Otherwise, DB = conjugate(B), corresponding to the l2-inner-product
b : {float|complex array}
     nrows x BDBCols array in row-major form.
     This  array is B-squared, i.e. it is each column of B
     multiplied against each other column of B.  For a Nx3 B,
     b[:,0] = conjugate(B[:,0])*B[:,0]
     b[:,1] = conjugate(B[:,0])*B[:,1]
     b[:,2] = conjugate(B[:,0])*B[:,2]
     b[:,3] = conjugate(B[:,1])*B[:,1]
     b[:,4] = conjugate(B[:,1])*B[:,2]
     b[:,5] = conjugate(B[:,2])*B[:,2]
BDBCols : {int}
     sum(range(NullDim+1)), i.e. number of columns in b
NullDim : {int}
     Number of nullspace vectors
tol : {float}
     Used to determine when values are numerically zero

Returns
-------
Sx : {float|complex array}
     Modified inplace and holds strength values for the above minimization problem

Notes
-----
Upon entry to the routine, S = (I - (t/k) Dinv A)^k.  However,
we only need the values of S at the sparsity pattern of A.  Hence,
there is no need to completely calculate all of S.

b is used to save on the computation of each local minimization problem

Principle calling routine is evolution_strength_of_connection(...) in strength.py.
In that routine, it is used to calculate strength-of-connection for the case
of multiple near-nullspace modes.

Examples
--------
See evolution_strength_of_connection(...) in strength.py)pbdoc");

    m.def("incomplete_mat_mult_csr", &_incomplete_mat_mult_csr<int, float, float>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("Bp").noconvert(), py::arg("Bj").noconvert(), py::arg("Bx").noconvert(), py::arg("Sp").noconvert(), py::arg("Sj").noconvert(), py::arg("Sx").noconvert(), py::arg("num_rows"));
    m.def("incomplete_mat_mult_csr", &_incomplete_mat_mult_csr<int, double, double>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("Bp").noconvert(), py::arg("Bj").noconvert(), py::arg("Bx").noconvert(), py::arg("Sp").noconvert(), py::arg("Sj").noconvert(), py::arg("Sx").noconvert(), py::arg("num_rows"));
    m.def("incomplete_mat_mult_csr", &_incomplete_mat_mult_csr<int, std::complex<float>, float>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("Bp").noconvert(), py::arg("Bj").noconvert(), py::arg("Bx").noconvert(), py::arg("Sp").noconvert(), py::arg("Sj").noconvert(), py::arg("Sx").noconvert(), py::arg("num_rows"));
    m.def("incomplete_mat_mult_csr", &_incomplete_mat_mult_csr<int, std::complex<double>, double>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("Bp").noconvert(), py::arg("Bj").noconvert(), py::arg("Bx").noconvert(), py::arg("Sp").noconvert(), py::arg("Sj").noconvert(), py::arg("Sx").noconvert(), py::arg("num_rows"),
R"pbdoc(
Calculate A*B = S, but only at the pre-existing sparsity
pattern of S, i.e. do an exact, but incomplete mat-mat multiply.

A must be in CSR, B must be in CSC and S must be in CSR
Indices for A, B and S must be sorted
A, B, and S must be square

Parameters
----------
Ap : {int array}
     Row pointer array for CSR matrix A
Aj : {int array}
     Col index array for CSR matrix A
Ax : {float|complex array}
     Value array for CSR matrix A
Bp : {int array}
     Row pointer array for CSC matrix B
Bj : {int array}
     Col index array for CSC matrix B
Bx : {float|complex array}
     Value array for CSC matrix B
Sp : {int array}
     Row pointer array for CSR matrix S
Sj : {int array}
     Col index array for CSR matrix S
Sx : {float|complex array}
     Value array for CSR matrix S
dimen: {int}
     dimensionality of A,B and S

Returns
-------
Sx : {float|complex array}
     Modified inplace to reflect S(i,j) = <A_{i,:}, B_{:,j}>

Notes
-----
A must be in CSR, B must be in CSC and S must be in CSR.
Indices for A, B and S must all be sorted.
A, B and S must be square.

Algorithm is naive, S(i,j) = <A_{i,:}, B_{:,j}>
But, the routine is written for the case when S's
sparsity pattern is a subset of A*B, so this algorithm
should work well.

Principle calling routine is evolution_strength_of_connection in
strength.py.  Here it is used to calculate S*S only at the
sparsity pattern of the original operator.  This allows for
BIG cost savings.

Examples
--------
>>> from pyamg.amg_core import incomplete_mat_mult_csr
>>> from scipy import arange, eye, ones
>>> from scipy.sparse import csr_matrix, csc_matrix
>>>
>>> A = csr_matrix(arange(1,10,dtype=float).reshape(3,3))
>>> B = csc_matrix(ones((3,3),dtype=float))
>>> AB = csr_matrix(eye(3,3,dtype=float))
>>> A.sort_indices()
>>> B.sort_indices()
>>> AB.sort_indices()
>>> incomplete_mat_mult_csr(A.indptr, A.indices, A.data, B.indptr, B.indices,
                      B.data, AB.indptr, AB.indices, AB.data, 3)
>>> print "Incomplete Matrix-Matrix Multiplication\n" + str(AB.todense())
>>> print "Complete Matrix-Matrix Multiplication\n" + str((A*B).todense()))pbdoc");

}

