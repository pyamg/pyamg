// DO NOT EDIT: this file is generated

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>

#include "smoothed_aggregation.h"

namespace py = pybind11;

template<class I, class T, class F>
void _symmetric_strength_of_connection(
            const I n_row,
            const F theta,
      py::array_t<I> & Ap,
      py::array_t<I> & Aj,
      py::array_t<T> & Ax,
      py::array_t<I> & Sp,
      py::array_t<I> & Sj,
      py::array_t<T> & Sx
                                       )
{
    auto py_Ap = Ap.unchecked();
    auto py_Aj = Aj.unchecked();
    auto py_Ax = Ax.unchecked();
    auto py_Sp = Sp.mutable_unchecked();
    auto py_Sj = Sj.mutable_unchecked();
    auto py_Sx = Sx.mutable_unchecked();
    const I *_Ap = py_Ap.data();
    const I *_Aj = py_Aj.data();
    const T *_Ax = py_Ax.data();
    I *_Sp = py_Sp.mutable_data();
    I *_Sj = py_Sj.mutable_data();
    T *_Sx = py_Sx.mutable_data();

    return symmetric_strength_of_connection<I, T, F>(
                    n_row,
                    theta,
                      _Ap, Ap.shape(0),
                      _Aj, Aj.shape(0),
                      _Ax, Ax.shape(0),
                      _Sp, Sp.shape(0),
                      _Sj, Sj.shape(0),
                      _Sx, Sx.shape(0)
                                                     );
}

template <class I>
I _standard_aggregation(
            const I n_row,
      py::array_t<I> & Ap,
      py::array_t<I> & Aj,
       py::array_t<I> & x,
       py::array_t<I> & y
                        )
{
    auto py_Ap = Ap.unchecked();
    auto py_Aj = Aj.unchecked();
    auto py_x = x.mutable_unchecked();
    auto py_y = y.mutable_unchecked();
    const I *_Ap = py_Ap.data();
    const I *_Aj = py_Aj.data();
    I *_x = py_x.mutable_data();
    I *_y = py_y.mutable_data();

    return standard_aggregation <I>(
                    n_row,
                      _Ap, Ap.shape(0),
                      _Aj, Aj.shape(0),
                       _x, x.shape(0),
                       _y, y.shape(0)
                                    );
}

template <class I>
I _naive_aggregation(
            const I n_row,
      py::array_t<I> & Ap,
      py::array_t<I> & Aj,
       py::array_t<I> & x,
       py::array_t<I> & y
                     )
{
    auto py_Ap = Ap.unchecked();
    auto py_Aj = Aj.unchecked();
    auto py_x = x.mutable_unchecked();
    auto py_y = y.mutable_unchecked();
    const I *_Ap = py_Ap.data();
    const I *_Aj = py_Aj.data();
    I *_x = py_x.mutable_data();
    I *_y = py_y.mutable_data();

    return naive_aggregation <I>(
                    n_row,
                      _Ap, Ap.shape(0),
                      _Aj, Aj.shape(0),
                       _x, x.shape(0),
                       _y, y.shape(0)
                                 );
}

template <class I, class T>
void _fit_candidates_real(
            const I n_row,
            const I n_col,
               const I K1,
               const I K2,
      py::array_t<I> & Ap,
      py::array_t<I> & Ai,
      py::array_t<T> & Ax,
       py::array_t<T> & B,
       py::array_t<T> & R,
              const T tol
                          )
{
    auto py_Ap = Ap.unchecked();
    auto py_Ai = Ai.unchecked();
    auto py_Ax = Ax.mutable_unchecked();
    auto py_B = B.unchecked();
    auto py_R = R.mutable_unchecked();
    const I *_Ap = py_Ap.data();
    const I *_Ai = py_Ai.data();
    T *_Ax = py_Ax.mutable_data();
    const T *_B = py_B.data();
    T *_R = py_R.mutable_data();

    return fit_candidates_real <I, T>(
                    n_row,
                    n_col,
                       K1,
                       K2,
                      _Ap, Ap.shape(0),
                      _Ai, Ai.shape(0),
                      _Ax, Ax.shape(0),
                       _B, B.shape(0),
                       _R, R.shape(0),
                      tol
                                      );
}

template <class I, class S, class T>
void _fit_candidates_complex(
            const I n_row,
            const I n_col,
               const I K1,
               const I K2,
      py::array_t<I> & Ap,
      py::array_t<I> & Ai,
      py::array_t<T> & Ax,
       py::array_t<T> & B,
       py::array_t<T> & R,
              const S tol
                             )
{
    auto py_Ap = Ap.unchecked();
    auto py_Ai = Ai.unchecked();
    auto py_Ax = Ax.mutable_unchecked();
    auto py_B = B.unchecked();
    auto py_R = R.mutable_unchecked();
    const I *_Ap = py_Ap.data();
    const I *_Ai = py_Ai.data();
    T *_Ax = py_Ax.mutable_data();
    const T *_B = py_B.data();
    T *_R = py_R.mutable_data();

    return fit_candidates_complex <I, S, T>(
                    n_row,
                    n_col,
                       K1,
                       K2,
                      _Ap, Ap.shape(0),
                      _Ai, Ai.shape(0),
                      _Ax, Ax.shape(0),
                       _B, B.shape(0),
                       _R, R.shape(0),
                      tol
                                            );
}

template<class I, class T, class F>
void _satisfy_constraints_helper(
     const I RowsPerBlock,
     const I ColsPerBlock,
   const I num_block_rows,
          const I NullDim,
       py::array_t<T> & x,
       py::array_t<T> & y,
       py::array_t<T> & z,
      py::array_t<I> & Sp,
      py::array_t<I> & Sj,
      py::array_t<T> & Sx
                                 )
{
    auto py_x = x.unchecked();
    auto py_y = y.unchecked();
    auto py_z = z.unchecked();
    auto py_Sp = Sp.unchecked();
    auto py_Sj = Sj.unchecked();
    auto py_Sx = Sx.mutable_unchecked();
    const T *_x = py_x.data();
    const T *_y = py_y.data();
    const T *_z = py_z.data();
    const I *_Sp = py_Sp.data();
    const I *_Sj = py_Sj.data();
    T *_Sx = py_Sx.mutable_data();

    return satisfy_constraints_helper<I, T, F>(
             RowsPerBlock,
             ColsPerBlock,
           num_block_rows,
                  NullDim,
                       _x, x.shape(0),
                       _y, y.shape(0),
                       _z, z.shape(0),
                      _Sp, Sp.shape(0),
                      _Sj, Sj.shape(0),
                      _Sx, Sx.shape(0)
                                               );
}

template<class I, class T, class F>
void _calc_BtB(
          const I NullDim,
           const I Nnodes,
     const I ColsPerBlock,
       py::array_t<T> & b,
          const I BsqCols,
       py::array_t<T> & x,
      py::array_t<I> & Sp,
      py::array_t<I> & Sj
               )
{
    auto py_b = b.unchecked();
    auto py_x = x.mutable_unchecked();
    auto py_Sp = Sp.unchecked();
    auto py_Sj = Sj.unchecked();
    const T *_b = py_b.data();
    T *_x = py_x.mutable_data();
    const I *_Sp = py_Sp.data();
    const I *_Sj = py_Sj.data();

    return calc_BtB<I, T, F>(
                  NullDim,
                   Nnodes,
             ColsPerBlock,
                       _b, b.shape(0),
                  BsqCols,
                       _x, x.shape(0),
                      _Sp, Sp.shape(0),
                      _Sj, Sj.shape(0)
                             );
}

template<class I, class T, class F>
void _incomplete_mat_mult_bsr(
      py::array_t<I> & Ap,
      py::array_t<I> & Aj,
      py::array_t<T> & Ax,
      py::array_t<I> & Bp,
      py::array_t<I> & Bj,
      py::array_t<T> & Bx,
      py::array_t<I> & Sp,
      py::array_t<I> & Sj,
      py::array_t<T> & Sx,
           const I n_brow,
           const I n_bcol,
           const I brow_A,
           const I bcol_A,
           const I bcol_B
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

    return incomplete_mat_mult_bsr<I, T, F>(
                      _Ap, Ap.shape(0),
                      _Aj, Aj.shape(0),
                      _Ax, Ax.shape(0),
                      _Bp, Bp.shape(0),
                      _Bj, Bj.shape(0),
                      _Bx, Bx.shape(0),
                      _Sp, Sp.shape(0),
                      _Sj, Sj.shape(0),
                      _Sx, Sx.shape(0),
                   n_brow,
                   n_bcol,
                   brow_A,
                   bcol_A,
                   bcol_B
                                            );
}

template<class I, class T, class F>
void _truncate_rows_csr(
            const I n_row,
                const I k,
      py::array_t<I> & Sp,
      py::array_t<I> & Sj,
      py::array_t<T> & Sx
                        )
{
    auto py_Sp = Sp.unchecked();
    auto py_Sj = Sj.mutable_unchecked();
    auto py_Sx = Sx.mutable_unchecked();
    const I *_Sp = py_Sp.data();
    I *_Sj = py_Sj.mutable_data();
    T *_Sx = py_Sx.mutable_data();

    return truncate_rows_csr<I, T, F>(
                    n_row,
                        k,
                      _Sp, Sp.shape(0),
                      _Sj, Sj.shape(0),
                      _Sx, Sx.shape(0)
                                      );
}

PYBIND11_MODULE(smoothed_aggregation, m) {
    m.doc() = R"pbdoc(
    Pybind11 bindings for smoothed_aggregation.h

    Methods
    -------
    symmetric_strength_of_connection
    standard_aggregation
    naive_aggregation
    fit_candidates_real
    fit_candidates_complex
    satisfy_constraints_helper
    calc_BtB
    incomplete_mat_mult_bsr
    truncate_rows_csr
    )pbdoc";

    py::options options;
    options.disable_function_signatures();

    m.def("symmetric_strength_of_connection", &_symmetric_strength_of_connection<int, float, float>,
        py::arg("n_row"), py::arg("theta"), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("Sp").noconvert(), py::arg("Sj").noconvert(), py::arg("Sx").noconvert());
    m.def("symmetric_strength_of_connection", &_symmetric_strength_of_connection<int, double, double>,
        py::arg("n_row"), py::arg("theta"), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("Sp").noconvert(), py::arg("Sj").noconvert(), py::arg("Sx").noconvert());
    m.def("symmetric_strength_of_connection", &_symmetric_strength_of_connection<int, std::complex<float>, float>,
        py::arg("n_row"), py::arg("theta"), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("Sp").noconvert(), py::arg("Sj").noconvert(), py::arg("Sx").noconvert());
    m.def("symmetric_strength_of_connection", &_symmetric_strength_of_connection<int, std::complex<double>, double>,
        py::arg("n_row"), py::arg("theta"), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("Sp").noconvert(), py::arg("Sj").noconvert(), py::arg("Sx").noconvert(),
R"pbdoc(
Compute a strength of connection matrix using the standard symmetric
 Smoothed Aggregation heuristic.  Both the input and output matrices
 are stored in CSR format.  A nonzero connection A[i,j] is considered
 strong if:

     abs(A[i,j]) >= theta * sqrt( abs(A[i,i]) * abs(A[j,j]) )

 The strength of connection matrix S is simply the set of nonzero entries
 of A that qualify as strong connections.

 Parameters
     num_rows   - number of rows in A
     theta      - stength of connection tolerance
     Ap[]       - CSR row pointer
     Aj[]       - CSR index array
     Ax[]       - CSR data array
     Sp[]       - (output) CSR row pointer
     Sj[]       - (output) CSR index array
     Sx[]       - (output) CSR data array


 Returns:
     Nothing, S will be stored in Sp, Sj, Sx

 Notes:
     Storage for S must be preallocated.  Since S will consist of a subset
     of A's nonzero values, a conservative bound is to allocate the same
     storage for S as is used by A.)pbdoc");

    m.def("standard_aggregation", &_standard_aggregation<int>,
        py::arg("n_row"), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("x").noconvert(), py::arg("y").noconvert(),
R"pbdoc(
Compute aggregates for a matrix A stored in CSR format

Parameters:
  n_row         - number of rows in A
  Ap[n_row + 1] - CSR row pointer
  Aj[nnz]       - CSR column indices
   x[n_row]     - aggregate numbers for each node
   y[n_row]     - will hold Cpts upon return

Returns:
 The number of aggregates (== max(x[:]) + 1 )

Notes:
   It is assumed that A is symmetric.
   A may contain diagonal entries (self loops)
   Unaggregated nodes are marked with a -1)pbdoc");

    m.def("naive_aggregation", &_naive_aggregation<int>,
        py::arg("n_row"), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("x").noconvert(), py::arg("y").noconvert(),
R"pbdoc(
Compute aggregates for a matrix A stored in CSR format

Parameters:
  n_row         - number of rows in A
  Ap[n_row + 1] - CSR row pointer
  Aj[nnz]       - CSR column indices
   x[n_row]     - aggregate numbers for each node
   y[n_row]     - will hold Cpts upon return

Returns:
 The number of aggregates (== max(x[:]) + 1 )

Notes:
Differs from standard aggregation.  Each dof is considered.
If it has been aggregated, skip over.  Otherwise, put dof
and any unaggregated neighbors in an aggregate.  Results
in possibly much higher complexities.)pbdoc");

    m.def("fit_candidates", &_fit_candidates_real<int, float>,
        py::arg("n_row"), py::arg("n_col"), py::arg("K1"), py::arg("K2"), py::arg("Ap").noconvert(), py::arg("Ai").noconvert(), py::arg("Ax").noconvert(), py::arg("B").noconvert(), py::arg("R").noconvert(), py::arg("tol"));
    m.def("fit_candidates", &_fit_candidates_real<int, double>,
        py::arg("n_row"), py::arg("n_col"), py::arg("K1"), py::arg("K2"), py::arg("Ap").noconvert(), py::arg("Ai").noconvert(), py::arg("Ax").noconvert(), py::arg("B").noconvert(), py::arg("R").noconvert(), py::arg("tol"),
R"pbdoc(
)pbdoc");

    m.def("fit_candidates", &_fit_candidates_complex<int, float, std::complex<float>>,
        py::arg("n_row"), py::arg("n_col"), py::arg("K1"), py::arg("K2"), py::arg("Ap").noconvert(), py::arg("Ai").noconvert(), py::arg("Ax").noconvert(), py::arg("B").noconvert(), py::arg("R").noconvert(), py::arg("tol"));
    m.def("fit_candidates", &_fit_candidates_complex<int, double, std::complex<double>>,
        py::arg("n_row"), py::arg("n_col"), py::arg("K1"), py::arg("K2"), py::arg("Ap").noconvert(), py::arg("Ai").noconvert(), py::arg("Ax").noconvert(), py::arg("B").noconvert(), py::arg("R").noconvert(), py::arg("tol"),
R"pbdoc(
)pbdoc");

    m.def("satisfy_constraints_helper", &_satisfy_constraints_helper<int, float, float>,
        py::arg("RowsPerBlock"), py::arg("ColsPerBlock"), py::arg("num_block_rows"), py::arg("NullDim"), py::arg("x").noconvert(), py::arg("y").noconvert(), py::arg("z").noconvert(), py::arg("Sp").noconvert(), py::arg("Sj").noconvert(), py::arg("Sx").noconvert());
    m.def("satisfy_constraints_helper", &_satisfy_constraints_helper<int, double, double>,
        py::arg("RowsPerBlock"), py::arg("ColsPerBlock"), py::arg("num_block_rows"), py::arg("NullDim"), py::arg("x").noconvert(), py::arg("y").noconvert(), py::arg("z").noconvert(), py::arg("Sp").noconvert(), py::arg("Sj").noconvert(), py::arg("Sx").noconvert());
    m.def("satisfy_constraints_helper", &_satisfy_constraints_helper<int, std::complex<float>, float>,
        py::arg("RowsPerBlock"), py::arg("ColsPerBlock"), py::arg("num_block_rows"), py::arg("NullDim"), py::arg("x").noconvert(), py::arg("y").noconvert(), py::arg("z").noconvert(), py::arg("Sp").noconvert(), py::arg("Sj").noconvert(), py::arg("Sx").noconvert());
    m.def("satisfy_constraints_helper", &_satisfy_constraints_helper<int, std::complex<double>, double>,
        py::arg("RowsPerBlock"), py::arg("ColsPerBlock"), py::arg("num_block_rows"), py::arg("NullDim"), py::arg("x").noconvert(), py::arg("y").noconvert(), py::arg("z").noconvert(), py::arg("Sp").noconvert(), py::arg("Sj").noconvert(), py::arg("Sx").noconvert(),
R"pbdoc(
)pbdoc");

    m.def("calc_BtB", &_calc_BtB<int, float, float>,
        py::arg("NullDim"), py::arg("Nnodes"), py::arg("ColsPerBlock"), py::arg("b").noconvert(), py::arg("BsqCols"), py::arg("x").noconvert(), py::arg("Sp").noconvert(), py::arg("Sj").noconvert());
    m.def("calc_BtB", &_calc_BtB<int, double, double>,
        py::arg("NullDim"), py::arg("Nnodes"), py::arg("ColsPerBlock"), py::arg("b").noconvert(), py::arg("BsqCols"), py::arg("x").noconvert(), py::arg("Sp").noconvert(), py::arg("Sj").noconvert());
    m.def("calc_BtB", &_calc_BtB<int, std::complex<float>, float>,
        py::arg("NullDim"), py::arg("Nnodes"), py::arg("ColsPerBlock"), py::arg("b").noconvert(), py::arg("BsqCols"), py::arg("x").noconvert(), py::arg("Sp").noconvert(), py::arg("Sj").noconvert());
    m.def("calc_BtB", &_calc_BtB<int, std::complex<double>, double>,
        py::arg("NullDim"), py::arg("Nnodes"), py::arg("ColsPerBlock"), py::arg("b").noconvert(), py::arg("BsqCols"), py::arg("x").noconvert(), py::arg("Sp").noconvert(), py::arg("Sj").noconvert(),
R"pbdoc(
Helper routine for energy_prolongation_smoother
Calculates the following python code:

  RowsPerBlock = Sparsity_Pattern.blocksize[0]
  BtB = zeros((Nnodes,NullDim,NullDim), dtype=B.dtype)
  S2 = Sparsity_Pattern.tocsr()
  for i in range(Nnodes):
      Bi = mat( B[S2.indices[S2.indptr[i*RowsPerBlock]:S2.indptr[i*RowsPerBlock + 1]],:] )
      BtB[i,:,:] = Bi.H*Bi

Parameters
----------
NullDim : {int}
     Number of near nullspace vectors
Nnodes : {int}
     Number of nodes, i.e. number of block rows in BSR matrix, S
ColsPerBlock : {int}
     Columns per block in S
b : {float|complex array}
     Nnodes x BsqCols array, in row-major form.
     This is B-squared, i.e. it is each column of B
     multiplied against each other column of B.  For a Nx3 B,
     b[:,0] = conjugate(B[:,0])*B[:,0]
     b[:,1] = conjugate(B[:,0])*B[:,1]
     b[:,2] = conjugate(B[:,0])*B[:,2]
     b[:,3] = conjugate(B[:,1])*B[:,1]
     b[:,4] = conjugate(B[:,1])*B[:,2]
     b[:,5] = conjugate(B[:,2])*B[:,2]
BsqCols : {int}
     sum(range(NullDim+1)), i.e. number of columns in b
x  : {float|complex array}
     Modified inplace for output.  Should be zeros upon entry
Sp,Sj : {int array}
     BSR indptr and indices members for matrix, S

Return
------
BtB[i] = B_i.H*B_i in __column__ major format
where B_i is B[colindices,:], colindices = all the nonzero
column indices for block row i in S

Notes
-----
Principle calling routine is energy_prolongation_smoother(...) in smooth.py.)pbdoc");

    m.def("incomplete_mat_mult_bsr", &_incomplete_mat_mult_bsr<int, float, float>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("Bp").noconvert(), py::arg("Bj").noconvert(), py::arg("Bx").noconvert(), py::arg("Sp").noconvert(), py::arg("Sj").noconvert(), py::arg("Sx").noconvert(), py::arg("n_brow"), py::arg("n_bcol"), py::arg("brow_A"), py::arg("bcol_A"), py::arg("bcol_B"));
    m.def("incomplete_mat_mult_bsr", &_incomplete_mat_mult_bsr<int, double, double>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("Bp").noconvert(), py::arg("Bj").noconvert(), py::arg("Bx").noconvert(), py::arg("Sp").noconvert(), py::arg("Sj").noconvert(), py::arg("Sx").noconvert(), py::arg("n_brow"), py::arg("n_bcol"), py::arg("brow_A"), py::arg("bcol_A"), py::arg("bcol_B"));
    m.def("incomplete_mat_mult_bsr", &_incomplete_mat_mult_bsr<int, std::complex<float>, float>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("Bp").noconvert(), py::arg("Bj").noconvert(), py::arg("Bx").noconvert(), py::arg("Sp").noconvert(), py::arg("Sj").noconvert(), py::arg("Sx").noconvert(), py::arg("n_brow"), py::arg("n_bcol"), py::arg("brow_A"), py::arg("bcol_A"), py::arg("bcol_B"));
    m.def("incomplete_mat_mult_bsr", &_incomplete_mat_mult_bsr<int, std::complex<double>, double>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("Bp").noconvert(), py::arg("Bj").noconvert(), py::arg("Bx").noconvert(), py::arg("Sp").noconvert(), py::arg("Sj").noconvert(), py::arg("Sx").noconvert(), py::arg("n_brow"), py::arg("n_bcol"), py::arg("brow_A"), py::arg("bcol_A"), py::arg("bcol_B"),
R"pbdoc(
Calculate A*B = S, but only at the pre-existing sparsity
pattern of S, i.e. do an exact, but incomplete mat-mat mult.

A, B and S must all be in BSR, may be rectangular, but the
indices need not be sorted.
Also, A.blocksize[0] must equal S.blocksize[0]
      A.blocksize[1] must equal B.blocksize[0]
      B.blocksize[1] must equal S.blocksize[1]

Parameters
----------
Ap : {int array}
     BSR row pointer array
Aj : {int array}
     BSR col index array
Ax : {float|complex array}
     BSR value array
Bp : {int array}
     BSR row pointer array
Bj : {int array}
     BSR col index array
Bx : {float|complex array}
     BSR value array
Sp : {int array}
     BSR row pointer array
Sj : {int array}
     BSR col index array
Sx : {float|complex array}
     BSR value array
n_brow : {int}
     Number of block-rows in A
n_bcol : {int}
     Number of block-cols in S
brow_A : {int}
     row blocksize for A
bcol_A : {int}
     column blocksize for A
bcol_B : {int}
     column blocksize for B

Returns
-------
Sx is modified in-place to reflect S(i,j) = <A_{i,:}, B_{:,j}>
but only for those entries already present in the sparsity pattern
of S.

Notes
-----

Algorithm is SMMP

Principle calling routine is energy_prolongation_smoother(...) in
smooth.py.  Here it is used to calculate the descent direction
A*P_tent, but only within an accepted sparsity pattern.

Is generally faster than the commented out incomplete_BSRmatmat(...)
routine below, except when S has far few nonzeros than A or B.)pbdoc");

    m.def("truncate_rows_csr", &_truncate_rows_csr<int, float, float>,
        py::arg("n_row"), py::arg("k"), py::arg("Sp").noconvert(), py::arg("Sj").noconvert(), py::arg("Sx").noconvert());
    m.def("truncate_rows_csr", &_truncate_rows_csr<int, double, double>,
        py::arg("n_row"), py::arg("k"), py::arg("Sp").noconvert(), py::arg("Sj").noconvert(), py::arg("Sx").noconvert());
    m.def("truncate_rows_csr", &_truncate_rows_csr<int, std::complex<float>, float>,
        py::arg("n_row"), py::arg("k"), py::arg("Sp").noconvert(), py::arg("Sj").noconvert(), py::arg("Sx").noconvert());
    m.def("truncate_rows_csr", &_truncate_rows_csr<int, std::complex<double>, double>,
        py::arg("n_row"), py::arg("k"), py::arg("Sp").noconvert(), py::arg("Sj").noconvert(), py::arg("Sx").noconvert(),
R"pbdoc(
Truncate the entries in A, such that only the largest (in magnitude)
 k entries per row are left.   Smaller entries are zeroed out.

 Parameters
     n_row      - number of rows in A
     k          - number of entries per row to keep
     Sp[]       - CSR row pointer
     Sj[]       - CSR index array
     Sx[]       - CSR data array


 Returns:
     Nothing, A will be stored in Sp, Sj, Sx with some entries zeroed out)pbdoc");

}

