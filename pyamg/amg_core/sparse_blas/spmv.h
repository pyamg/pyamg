// Threaded sparse matrix-vector products for CSR and BSR formats.
//
// Algorithm: row-wise SpMV with OpenMP parallelism over output rows.
//
// References
// ----------
// Gustavson, F.G. "Two Fast Algorithms for Sparse Matrices: Multiplication
//   and Permuted Transposition." ACM TOMS 4(3):250-269, 1978.
//   doi:10.1145/355791.355796
//
// Bell, N., Dalton, S., Olson, L.N. "Exposing Fine-Grained Parallelism in
//   Algebraic Multigrid Methods." SIAM J. Sci. Comput. 34(4):C123-C152, 2012.
//   doi:10.1137/110838844
//
#ifndef PYAMG_SPARSE_BLAS_SPMV_H
#define PYAMG_SPARSE_BLAS_SPMV_H

#ifdef _OPENMP
#include <omp.h>
#endif

namespace pyamg {
namespace sparse_blas {

// y = A @ x for CSR-stored A.
//
// A is n_row x n_col, stored in CSR with index arrays Ap (length n_row+1)
// and Aj, and value array Ax. x is length n_col, y is length n_row.
// Overwrites y.
//
// Threading: OpenMP parallel-for over output rows. Each thread owns a
// contiguous range of rows; no atomics, no false sharing on y (each thread
// writes distinct y[i]).
template <class I, class T>
void csr_matvec(const I n_row,
                const I Ap[],
                const I Aj[],
                const T Ax[],
                const T x[],
                T y[])
{
    // `if(omp_get_max_threads() > 1)` skips the parallel-region machinery
    // entirely when only one thread is available, avoiding libgomp's
    // team-creation cost. Compiles to a plain serial loop in that case.
    #pragma omp parallel for if(omp_get_max_threads() > 1) schedule(dynamic, 64)
    for (I i = 0; i < n_row; ++i) {
        T sum = T(0);
        const I row_start = Ap[i];
        const I row_end   = Ap[i + 1];
        for (I jj = row_start; jj < row_end; ++jj) {
            sum += Ax[jj] * x[Aj[jj]];
        }
        y[i] = sum;
    }
}

// y = A @ x for BSR-stored A with block size R_blk x C_blk.
//
// Ap (length n_brow + 1) and Aj are at block granularity. Ax is the dense
// block storage, length n_blocks * R_blk * C_blk, blocks laid out row-major.
// x is length n_brow*C_blk (well, n_bcol*C_blk; we take n_bcol implicitly via
// Aj indexing). y is length n_brow*R_blk. Overwrites y.
//
// Threading: OpenMP parallel-for over block-rows. Each thread writes a
// disjoint R_blk-sized strip of y per block-row, so no false sharing.
template <class I, class T>
void bsr_matvec(const I n_brow,
                const I R_blk,
                const I C_blk,
                const I Ap[],
                const I Aj[],
                const T Ax[],
                const T x[],
                T y[])
{
    const I block_size = R_blk * C_blk;
    #pragma omp parallel for if(omp_get_max_threads() > 1) schedule(dynamic, 16)
    for (I bi = 0; bi < n_brow; ++bi) {
        T* y_row = y + bi * R_blk;
        for (I r = 0; r < R_blk; ++r) y_row[r] = T(0);

        const I row_start = Ap[bi];
        const I row_end   = Ap[bi + 1];
        for (I jj = row_start; jj < row_end; ++jj) {
            const I bj = Aj[jj];
            const T* block = Ax + jj * block_size;
            const T* x_blk = x + bj * C_blk;
            for (I r = 0; r < R_blk; ++r) {
                T s = T(0);
                const T* brow = block + r * C_blk;
                for (I c = 0; c < C_blk; ++c) {
                    s += brow[c] * x_blk[c];
                }
                y_row[r] += s;
            }
        }
    }
}

} // namespace sparse_blas
} // namespace pyamg

#endif // PYAMG_SPARSE_BLAS_SPMV_H
