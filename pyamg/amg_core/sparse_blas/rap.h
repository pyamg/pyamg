// Threaded fused triple-product RAP (CSR or BSR @ same @ same).
//
// Computes C = R @ A @ P in a single call without materializing the
// intermediate RA. Two-pass (symbolic then numeric), OpenMP-parallel over
// output (block-)rows.
//
// Block dimension. The kernel treats CSR as the BS=1 case of BSR with
// square BS x BS blocks. The symbolic pass (rap_pass1) is block-agnostic
// — it walks indptr/indices only. The numeric pass (rap_pass2) uses a
// dense q-SPA over the inner block-column space and a dense k-SPA over
// the output block-column space; each SPA slot holds one BS x BS dense
// block, accumulated by fused triple block-matmuls
//     acc_block[K] += R_block[I,P] @ A_block[P,Q] @ P_block[Q,K]
// Specialized fully-unrolled templates are emitted for BS in
// {1, 2, 3, 4, 6, 8} (covers Poisson scalar, elasticity 2D/3D, common
// nullspace dims). A runtime-bs fallback handles arbitrary square blocks.
//
// References
// ----------
// Bell, N., Dalton, S., Olson, L.N. "Exposing Fine-Grained Parallelism in
//   Algebraic Multigrid Methods." SIAM J. Sci. Comput. 34(4):C123-C152,
//   2012. doi:10.1137/110838844 — row-wise outer-product RAP structure.
//
// Gustavson, F.G. "Two Fast Algorithms for Sparse Matrices: Multiplication
//   and Permuted Transposition." ACM Trans. Math. Software 4(3):250-269,
//   1978. doi:10.1145/355791.355796
//
#ifndef PYAMG_SPARSE_BLAS_RAP_H
#define PYAMG_SPARSE_BLAS_RAP_H

#ifdef _OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <cstddef>
#include <vector>

namespace pyamg {
namespace sparse_blas {

// ------------------------------------------------------------------
// Block matmul: C = A @ B (square BS x BS, row-major). BS==1 collapses
// to a scalar multiply.
// ------------------------------------------------------------------
template <class T, int BS>
inline void block_matmul(const T* A, const T* B, T* C) {
    for (int i = 0; i < BS; ++i) {
        for (int j = 0; j < BS; ++j) {
            T s = T(0);
            for (int k = 0; k < BS; ++k) {
                s += A[i * BS + k] * B[k * BS + j];
            }
            C[i * BS + j] = s;
        }
    }
}

template <class T>
inline void block_matmul_dyn(const T* A, const T* B, T* C, int bs) {
    for (int i = 0; i < bs; ++i) {
        for (int j = 0; j < bs; ++j) {
            T s = T(0);
            for (int k = 0; k < bs; ++k) {
                s += A[i * bs + k] * B[k * bs + j];
            }
            C[i * bs + j] = s;
        }
    }
}

// ------------------------------------------------------------------
// Per-thread persistent scratch. Block-aware: q_sums and k_sums are sized
// in scalar entries (n_inner_blocks * bs^2 for q, n_col_blocks * bs^2 for
// k), so that the SPA logically holds one BS x BS dense block per slot.
// ------------------------------------------------------------------
template <class I>
struct RapPass1Scratch {
    std::vector<long long> q_mask;
    std::vector<long long> k_mask;
    std::vector<I> q_chain;
    long long q_epoch = 0;
    long long k_epoch = 0;
};

template <class I, class T>
struct RapPass2Scratch {
    std::vector<long long> q_mask;
    std::vector<T>         q_sums;
    std::vector<I>         q_chain;
    std::vector<long long> k_mask;
    std::vector<T>         k_sums;
    std::vector<I>         k_chain;
    long long q_epoch = 0;
    long long k_epoch = 0;
};

// ============================================================
// Pass 1 (symbolic): output (block-)row pointers. Block-agnostic.
// n_row, n_inner, n_col are *block* counts when called for BSR.
// ============================================================
template <class I>
void rap_pass1(const I n_row,
               const I n_inner,
               const I n_col,
               const I Rp[], const I Rj[],
               const I Ap[], const I Aj[],
               const I Pp[], const I Pj[],
               I Cp[])
{
    #pragma omp parallel if(omp_get_max_threads() > 1)
    {
        static thread_local RapPass1Scratch<I> spa;
        if (spa.q_mask.size() < static_cast<std::size_t>(n_inner)) {
            spa.q_mask.assign(n_inner, -1);
        }
        if (spa.k_mask.size() < static_cast<std::size_t>(n_col)) {
            spa.k_mask.assign(n_col, -1);
        }
        const long long q_base = spa.q_epoch; spa.q_epoch += n_row;
        const long long k_base = spa.k_epoch; spa.k_epoch += n_row;

        #pragma omp for schedule(dynamic, 64)
        for (I i = 0; i < n_row; ++i) {
            const long long qtag = q_base + i;
            const long long ktag = k_base + i;
            spa.q_chain.clear();

            for (I jj = Rp[i]; jj < Rp[i + 1]; ++jj) {
                const I p = Rj[jj];
                for (I kk = Ap[p]; kk < Ap[p + 1]; ++kk) {
                    const I q = Aj[kk];
                    if (spa.q_mask[q] != qtag) {
                        spa.q_mask[q] = qtag;
                        spa.q_chain.push_back(q);
                    }
                }
            }

            I row_nnz = 0;
            for (std::size_t idx = 0; idx < spa.q_chain.size(); ++idx) {
                const I q = spa.q_chain[idx];
                for (I kk = Pp[q]; kk < Pp[q + 1]; ++kk) {
                    const I k = Pj[kk];
                    if (spa.k_mask[k] != ktag) {
                        spa.k_mask[k] = ktag;
                        ++row_nnz;
                    }
                }
            }
            Cp[i + 1] = row_nnz;
        }
    }

    Cp[0] = 0;
    for (I i = 0; i < n_row; ++i) Cp[i + 1] += Cp[i];
}

// ============================================================
// Pass 2 (numeric): templated specialization on compile-time BS.
// Rx/Ax/Px hold nnz_blocks * BS*BS scalar values; same for Cx.
// ============================================================
template <class I, class T, int BS>
void rap_pass2_bs(const I n_row,
                  const I n_inner,
                  const I n_col,
                  const I Rp[], const I Rj[], const T Rx[],
                  const I Ap[], const I Aj[], const T Ax[],
                  const I Pp[], const I Pj[], const T Px[],
                  const I Cp[],
                  I Cj[], T Cx[],
                  const bool sort_indices)
{
    constexpr std::size_t B2 = static_cast<std::size_t>(BS) * BS;

    #pragma omp parallel if(omp_get_max_threads() > 1)
    {
        static thread_local RapPass2Scratch<I, T> spa;
        const std::size_t need_q = static_cast<std::size_t>(n_inner) * B2;
        const std::size_t need_k = static_cast<std::size_t>(n_col)   * B2;
        if (spa.q_mask.size() < static_cast<std::size_t>(n_inner)) {
            spa.q_mask.assign(n_inner, -1);
            spa.q_sums.resize(need_q);
        }
        if (spa.k_mask.size() < static_cast<std::size_t>(n_col)) {
            spa.k_mask.assign(n_col, -1);
            spa.k_sums.resize(need_k);
        }
        const long long q_base = spa.q_epoch; spa.q_epoch += n_row;
        const long long k_base = spa.k_epoch; spa.k_epoch += n_row;

        T contrib[B2];

        #pragma omp for schedule(dynamic, 64)
        for (I i = 0; i < n_row; ++i) {
            const long long qtag = q_base + i;
            const long long ktag = k_base + i;

            // 2a: build RA[i, :] in q-SPA.
            spa.q_chain.clear();
            for (I jj = Rp[i]; jj < Rp[i + 1]; ++jj) {
                const I p = Rj[jj];
                const T* Rblk = Rx + static_cast<std::size_t>(jj) * B2;
                for (I kk = Ap[p]; kk < Ap[p + 1]; ++kk) {
                    const I q = Aj[kk];
                    const T* Ablk = Ax + static_cast<std::size_t>(kk) * B2;
                    block_matmul<T, BS>(Rblk, Ablk, contrib);
                    T* dst = &spa.q_sums[static_cast<std::size_t>(q) * B2];
                    if (spa.q_mask[q] != qtag) {
                        spa.q_mask[q] = qtag;
                        for (std::size_t b = 0; b < B2; ++b) dst[b] = contrib[b];
                        spa.q_chain.push_back(q);
                    } else {
                        for (std::size_t b = 0; b < B2; ++b) dst[b] += contrib[b];
                    }
                }
            }

            // 2b: project RA[i, :] through P into RAP[i, :] via k-SPA.
            spa.k_chain.clear();
            for (std::size_t idx = 0; idx < spa.q_chain.size(); ++idx) {
                const I q = spa.q_chain[idx];
                const T* RAblk = &spa.q_sums[static_cast<std::size_t>(q) * B2];
                for (I kk = Pp[q]; kk < Pp[q + 1]; ++kk) {
                    const I k = Pj[kk];
                    const T* Pblk = Px + static_cast<std::size_t>(kk) * B2;
                    block_matmul<T, BS>(RAblk, Pblk, contrib);
                    T* dst = &spa.k_sums[static_cast<std::size_t>(k) * B2];
                    if (spa.k_mask[k] != ktag) {
                        spa.k_mask[k] = ktag;
                        for (std::size_t b = 0; b < B2; ++b) dst[b] = contrib[b];
                        spa.k_chain.push_back(k);
                    } else {
                        for (std::size_t b = 0; b < B2; ++b) dst[b] += contrib[b];
                    }
                }
            }

            if (sort_indices) {
                std::sort(spa.k_chain.begin(), spa.k_chain.end());
            }
            I dest = Cp[i];
            for (std::size_t idx = 0; idx < spa.k_chain.size(); ++idx) {
                const I k = spa.k_chain[idx];
                Cj[dest] = k;
                T* outdst = Cx + static_cast<std::size_t>(dest) * B2;
                T* src    = &spa.k_sums[static_cast<std::size_t>(k) * B2];
                for (std::size_t b = 0; b < B2; ++b) outdst[b] = src[b];
                ++dest;
            }
        }
    }
}

// ============================================================
// Pass 2 (numeric): runtime block size. Same algorithm; inner block-
// matmul uses regular loops over `bs` (no compile-time unroll). Used as
// the fallthrough for block sizes outside the specialized set.
// ============================================================
template <class I, class T>
void rap_pass2_dyn(const int bs,
                   const I n_row,
                   const I n_inner,
                   const I n_col,
                   const I Rp[], const I Rj[], const T Rx[],
                   const I Ap[], const I Aj[], const T Ax[],
                   const I Pp[], const I Pj[], const T Px[],
                   const I Cp[],
                   I Cj[], T Cx[],
                   const bool sort_indices)
{
    const std::size_t B2 = static_cast<std::size_t>(bs) * bs;

    #pragma omp parallel if(omp_get_max_threads() > 1)
    {
        static thread_local RapPass2Scratch<I, T> spa;
        const std::size_t need_q = static_cast<std::size_t>(n_inner) * B2;
        const std::size_t need_k = static_cast<std::size_t>(n_col)   * B2;
        if (spa.q_mask.size() < static_cast<std::size_t>(n_inner)) {
            spa.q_mask.assign(n_inner, -1);
            spa.q_sums.resize(need_q);
        }
        if (spa.k_mask.size() < static_cast<std::size_t>(n_col)) {
            spa.k_mask.assign(n_col, -1);
            spa.k_sums.resize(need_k);
        }
        const long long q_base = spa.q_epoch; spa.q_epoch += n_row;
        const long long k_base = spa.k_epoch; spa.k_epoch += n_row;

        std::vector<T> contrib(B2);

        #pragma omp for schedule(dynamic, 64)
        for (I i = 0; i < n_row; ++i) {
            const long long qtag = q_base + i;
            const long long ktag = k_base + i;

            spa.q_chain.clear();
            for (I jj = Rp[i]; jj < Rp[i + 1]; ++jj) {
                const I p = Rj[jj];
                const T* Rblk = Rx + static_cast<std::size_t>(jj) * B2;
                for (I kk = Ap[p]; kk < Ap[p + 1]; ++kk) {
                    const I q = Aj[kk];
                    const T* Ablk = Ax + static_cast<std::size_t>(kk) * B2;
                    block_matmul_dyn<T>(Rblk, Ablk, contrib.data(), bs);
                    T* dst = &spa.q_sums[static_cast<std::size_t>(q) * B2];
                    if (spa.q_mask[q] != qtag) {
                        spa.q_mask[q] = qtag;
                        for (std::size_t b = 0; b < B2; ++b) dst[b] = contrib[b];
                        spa.q_chain.push_back(q);
                    } else {
                        for (std::size_t b = 0; b < B2; ++b) dst[b] += contrib[b];
                    }
                }
            }

            spa.k_chain.clear();
            for (std::size_t idx = 0; idx < spa.q_chain.size(); ++idx) {
                const I q = spa.q_chain[idx];
                const T* RAblk = &spa.q_sums[static_cast<std::size_t>(q) * B2];
                for (I kk = Pp[q]; kk < Pp[q + 1]; ++kk) {
                    const I k = Pj[kk];
                    const T* Pblk = Px + static_cast<std::size_t>(kk) * B2;
                    block_matmul_dyn<T>(RAblk, Pblk, contrib.data(), bs);
                    T* dst = &spa.k_sums[static_cast<std::size_t>(k) * B2];
                    if (spa.k_mask[k] != ktag) {
                        spa.k_mask[k] = ktag;
                        for (std::size_t b = 0; b < B2; ++b) dst[b] = contrib[b];
                        spa.k_chain.push_back(k);
                    } else {
                        for (std::size_t b = 0; b < B2; ++b) dst[b] += contrib[b];
                    }
                }
            }

            if (sort_indices) {
                std::sort(spa.k_chain.begin(), spa.k_chain.end());
            }
            I dest = Cp[i];
            for (std::size_t idx = 0; idx < spa.k_chain.size(); ++idx) {
                const I k = spa.k_chain[idx];
                Cj[dest] = k;
                T* outdst = Cx + static_cast<std::size_t>(dest) * B2;
                T* src    = &spa.k_sums[static_cast<std::size_t>(k) * B2];
                for (std::size_t b = 0; b < B2; ++b) outdst[b] = src[b];
                ++dest;
            }
        }
    }
}

// ============================================================
// Pass 2 (numeric): dispatch on `block_size`. Specializations exist for
// {1, 2, 3, 4, 6, 8}; other sizes fall through to the runtime-bs kernel.
// ============================================================
template <class I, class T>
void rap_pass2(const int block_size,
               const I n_row,
               const I n_inner,
               const I n_col,
               const I Rp[], const I Rj[], const T Rx[],
               const I Ap[], const I Aj[], const T Ax[],
               const I Pp[], const I Pj[], const T Px[],
               const I Cp[],
               I Cj[], T Cx[],
               const bool sort_indices)
{
    switch (block_size) {
        case 1:
            rap_pass2_bs<I, T, 1>(n_row, n_inner, n_col,
                                  Rp, Rj, Rx, Ap, Aj, Ax, Pp, Pj, Px,
                                  Cp, Cj, Cx, sort_indices);
            break;
        case 2:
            rap_pass2_bs<I, T, 2>(n_row, n_inner, n_col,
                                  Rp, Rj, Rx, Ap, Aj, Ax, Pp, Pj, Px,
                                  Cp, Cj, Cx, sort_indices);
            break;
        case 3:
            rap_pass2_bs<I, T, 3>(n_row, n_inner, n_col,
                                  Rp, Rj, Rx, Ap, Aj, Ax, Pp, Pj, Px,
                                  Cp, Cj, Cx, sort_indices);
            break;
        case 4:
            rap_pass2_bs<I, T, 4>(n_row, n_inner, n_col,
                                  Rp, Rj, Rx, Ap, Aj, Ax, Pp, Pj, Px,
                                  Cp, Cj, Cx, sort_indices);
            break;
        case 6:
            rap_pass2_bs<I, T, 6>(n_row, n_inner, n_col,
                                  Rp, Rj, Rx, Ap, Aj, Ax, Pp, Pj, Px,
                                  Cp, Cj, Cx, sort_indices);
            break;
        case 8:
            rap_pass2_bs<I, T, 8>(n_row, n_inner, n_col,
                                  Rp, Rj, Rx, Ap, Aj, Ax, Pp, Pj, Px,
                                  Cp, Cj, Cx, sort_indices);
            break;
        default:
            rap_pass2_dyn<I, T>(block_size, n_row, n_inner, n_col,
                                Rp, Rj, Rx, Ap, Aj, Ax, Pp, Pj, Px,
                                Cp, Cj, Cx, sort_indices);
            break;
    }
}

} // namespace sparse_blas
} // namespace pyamg

#endif // PYAMG_SPARSE_BLAS_RAP_H
