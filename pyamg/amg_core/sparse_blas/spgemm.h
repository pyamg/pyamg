// Threaded sparse matrix-matrix multiply (CSR @ CSR).
//
// Algorithm: row-wise Gustavson with adaptive sparse accumulator (SPA)
// strategy chosen per row. Two-pass (symbolic then numeric), OpenMP-parallel
// over output rows.
//
// Per-row SPA strategy:
//  * Dense SPA (length n_col) when expected row fanin is large relative to
//    n_col. Bookkeeping is O(1) lookup. Persistent thread-local storage
//    avoids per-call allocation churn; an int64 "tag" lets us skip resetting
//    the marker between rows or between kernel invocations (Gustavson 1978).
//  * Hash SPA (open-addressed, sized per row to ~2x fanin) when expected
//    fanin is small relative to n_col. Memory cost is O(fanin), avoiding
//    O(n_col) overhead on wide-skinny products like restriction.
// Switch threshold: hash when 4 * fanin < n_col.
//
// References
// ----------
// Gustavson, F.G. "Two Fast Algorithms for Sparse Matrices: Multiplication
//   and Permuted Transposition." ACM Trans. Math. Software 4(3):250-269,
//   1978. doi:10.1145/355791.355796
//
// Bell, N., Dalton, S., Olson, L.N. "Exposing Fine-Grained Parallelism in
//   Algebraic Multigrid Methods." SIAM J. Sci. Comput. 34(4):C123-C152,
//   2012. doi:10.1137/110838844
//
// Nagasaka, Y., Nukada, A., Matsuoka, S. "High-Performance and Memory-
//   Saving Sparse General Matrix-Matrix Multiplication for NVIDIA Pascal
//   GPU." Proc. ICPP 2017. doi:10.1109/ICPP.2017.19
//
// Nagasaka, Y., Matsuoka, S., Azad, A., Buluç, A. "Performance optimization,
//   modeling and analysis of sparse matrix-matrix products on multi-core
//   and many-core processors." Parallel Computing 90:102545, 2019.
//   doi:10.1016/j.parco.2019.102545
//
#ifndef PYAMG_SPARSE_BLAS_SPGEMM_H
#define PYAMG_SPARSE_BLAS_SPGEMM_H

#ifdef _OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <cstdint>
#include <vector>

namespace pyamg {
namespace sparse_blas {

// Hash threshold: we use hash SPA when the expected row fanin is at least
// this factor smaller than n_col. Empirical default; tunable per workload
// but "4" is a reasonable starting point per Nagasaka et al. 2019.
constexpr int HASH_VS_DENSE_THRESHOLD = 4;

// Round capacity up to the next power of two (min 16). Power-of-two lets
// us replace `%` with `& (cap - 1)` in the probe sequence.
inline std::size_t next_pow2_min16(std::size_t x) {
    std::size_t p = 16;
    while (p < x) p <<= 1;
    return p;
}

// Knuth/Fibonacci hash: multiply by golden-ratio prime, mask to capacity.
// Fast integer hash suited for column indices clustered in small ranges.
template <class I>
inline std::size_t hash_idx(I key, std::size_t cap_mask) {
    return (static_cast<std::uint64_t>(key) * 0x9E3779B97F4A7C15ULL) & cap_mask;
}

// ---- per-thread persistent scratch ----
//
// `static thread_local` storage gives each OpenMP worker thread its own
// scratch that survives across kernel calls. mask[] uses int64 tags so we
// don't have to wipe it between rows or between calls — first-touch init
// only. hash[] is reset lazily as we go (entries cleared on iterate).

template <class I>
struct Pass1Scratch {
    // Dense SPA: tag per column, advanced by n_row each call.
    std::vector<long long> mask;
    long long epoch = 0;
    // Hash SPA: open-addressed table (capacity is power of two).
    std::vector<I> hkeys;          // -1 = empty
    std::vector<I> hchain;         // slots touched for current row
    std::size_t hcap = 0;
};

template <class I, class T>
struct Pass2Scratch {
    // Dense SPA
    std::vector<long long> mask;
    std::vector<T> sums;
    std::vector<I> chain;          // columns touched for current row (dense)
    long long epoch = 0;
    // Hash SPA
    std::vector<I> hkeys;          // -1 = empty
    std::vector<T> hvals;
    std::vector<I> hchain;         // slot indices touched for current row
    std::size_t hcap = 0;
};

// Ensure hash capacity at least `target`, re-init keys to -1 if grown.
template <class I>
inline void grow_hash_pass1(Pass1Scratch<I>& spa, std::size_t target) {
    const std::size_t cap = next_pow2_min16(target);
    if (cap > spa.hcap) {
        spa.hkeys.assign(cap, -1);
        spa.hcap = cap;
    }
}

template <class I, class T>
inline void grow_hash_pass2(Pass2Scratch<I, T>& spa, std::size_t target) {
    const std::size_t cap = next_pow2_min16(target);
    if (cap > spa.hcap) {
        spa.hkeys.assign(cap, -1);
        spa.hvals.resize(cap);
        spa.hcap = cap;
    }
}

// Estimate pre-dedup output fanin for row i: sum of B's row sizes over
// A's nonzeros in row i. Bounds the actual output row nnz from above.
template <class I>
inline I estimate_fanin(I i, const I Ap[], const I Aj[], const I Bp[]) {
    I s = 0;
    for (I jj = Ap[i]; jj < Ap[i + 1]; ++jj) {
        const I j = Aj[jj];
        s += Bp[j + 1] - Bp[j];
    }
    return s;
}

// ============================================================
// Pass 1 (symbolic): fill Cp such that Cp[i+1] = nnz of row i
// ============================================================
template <class I>
void csr_matmat_pass1(const I n_row,
                      const I n_col,
                      const I Ap[],
                      const I Aj[],
                      const I Bp[],
                      const I Bj[],
                      I Cp[])
{
    #pragma omp parallel if(omp_get_max_threads() > 1)
    {
        static thread_local Pass1Scratch<I> spa;
        if (spa.mask.size() < static_cast<std::size_t>(n_col)) {
            spa.mask.assign(n_col, -1);
        }
        const long long base = spa.epoch;
        spa.epoch += n_row;

        #pragma omp for schedule(dynamic, 64)
        for (I i = 0; i < n_row; ++i) {
            const I fanin = estimate_fanin<I>(i, Ap, Aj, Bp);

            // Pick SPA strategy. If fanin tiny relative to n_col, hash wins.
            const bool use_hash = (HASH_VS_DENSE_THRESHOLD * static_cast<long long>(fanin)
                                   < static_cast<long long>(n_col));
            I row_nnz = 0;

            if (use_hash) {
                // Size hash to ~2x fanin (load factor target 0.5).
                grow_hash_pass1<I>(spa, static_cast<std::size_t>(fanin) * 2 + 1);
                const std::size_t cap_mask = spa.hcap - 1;
                spa.hchain.clear();
                for (I jj = Ap[i]; jj < Ap[i + 1]; ++jj) {
                    const I j = Aj[jj];
                    for (I kk = Bp[j]; kk < Bp[j + 1]; ++kk) {
                        const I k = Bj[kk];
                        std::size_t slot = hash_idx<I>(k, cap_mask);
                        while (true) {
                            const I cur = spa.hkeys[slot];
                            if (cur == -1) {
                                spa.hkeys[slot] = k;
                                spa.hchain.push_back(static_cast<I>(slot));
                                ++row_nnz;
                                break;
                            }
                            if (cur == k) break;  // already counted
                            slot = (slot + 1) & cap_mask;
                        }
                    }
                }
                // Reset touched slots to -1 for the next row.
                for (I slot : spa.hchain) spa.hkeys[slot] = -1;
            } else {
                // Dense path with int64 tag.
                const long long tag = base + i;
                for (I jj = Ap[i]; jj < Ap[i + 1]; ++jj) {
                    const I j = Aj[jj];
                    for (I kk = Bp[j]; kk < Bp[j + 1]; ++kk) {
                        const I k = Bj[kk];
                        if (spa.mask[k] != tag) {
                            spa.mask[k] = tag;
                            ++row_nnz;
                        }
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
// Pass 2 (numeric): fill Cj and Cx given Cp from pass 1
// ============================================================
template <class I, class T>
void csr_matmat_pass2(const I n_row,
                      const I n_col,
                      const I Ap[],
                      const I Aj[],
                      const T Ax[],
                      const I Bp[],
                      const I Bj[],
                      const T Bx[],
                      const I Cp[],
                      I Cj[],
                      T Cx[],
                      const bool sort_indices = true)
{
    #pragma omp parallel if(omp_get_max_threads() > 1)
    {
        static thread_local Pass2Scratch<I, T> spa;
        if (spa.mask.size() < static_cast<std::size_t>(n_col)) {
            spa.mask.assign(n_col, -1);
            spa.sums.resize(n_col);
        }
        spa.chain.reserve(64);
        const long long base = spa.epoch;
        spa.epoch += n_row;

        #pragma omp for schedule(dynamic, 64)
        for (I i = 0; i < n_row; ++i) {
            // Use the same dispatch policy as pass 1 so the strategy is
            // consistent across symbolic/numeric for each row. Pass 1's
            // exact output size is in Cp[i+1] - Cp[i]; we use the input
            // fanin estimate here too so the decision matches pass 1.
            const I fanin = estimate_fanin<I>(i, Ap, Aj, Bp);
            const bool use_hash = (HASH_VS_DENSE_THRESHOLD * static_cast<long long>(fanin)
                                   < static_cast<long long>(n_col));

            if (use_hash) {
                // Hash path. Size to ~2x fanin (post-dedup output is smaller,
                // but we don't know it row-by-row without pass 1's data here).
                grow_hash_pass2<I, T>(spa, static_cast<std::size_t>(fanin) * 2 + 1);
                const std::size_t cap_mask = spa.hcap - 1;
                spa.hchain.clear();
                for (I jj = Ap[i]; jj < Ap[i + 1]; ++jj) {
                    const I j = Aj[jj];
                    const T a = Ax[jj];
                    for (I kk = Bp[j]; kk < Bp[j + 1]; ++kk) {
                        const I k = Bj[kk];
                        const T contrib = a * Bx[kk];
                        std::size_t slot = hash_idx<I>(k, cap_mask);
                        while (true) {
                            const I cur = spa.hkeys[slot];
                            if (cur == -1) {
                                spa.hkeys[slot] = k;
                                spa.hvals[slot] = contrib;
                                spa.hchain.push_back(static_cast<I>(slot));
                                break;
                            }
                            if (cur == k) {
                                spa.hvals[slot] += contrib;
                                break;
                            }
                            slot = (slot + 1) & cap_mask;
                        }
                    }
                }
                // Emit and reset.
                I dest = Cp[i];
                if (sort_indices) {
                    // Pull (key, val) pairs out, sort by key, write to C.
                    // Uses a small per-row scratch built from the chain.
                    const std::size_t n = spa.hchain.size();
                    // Reuse spa.chain as a (slot) scratch we'll sort by key.
                    spa.chain.assign(spa.hchain.begin(), spa.hchain.end());
                    std::sort(spa.chain.begin(), spa.chain.end(),
                        [&](I a_slot, I b_slot) {
                            return spa.hkeys[a_slot] < spa.hkeys[b_slot];
                        });
                    for (std::size_t idx = 0; idx < n; ++idx) {
                        const I slot = spa.chain[idx];
                        Cj[dest] = spa.hkeys[slot];
                        Cx[dest] = spa.hvals[slot];
                        ++dest;
                    }
                } else {
                    for (I slot : spa.hchain) {
                        Cj[dest] = spa.hkeys[slot];
                        Cx[dest] = spa.hvals[slot];
                        ++dest;
                    }
                }
                // Clear touched slots for the next row.
                for (I slot : spa.hchain) spa.hkeys[slot] = -1;
            } else {
                // Dense path.
                const long long tag = base + i;
                spa.chain.clear();
                for (I jj = Ap[i]; jj < Ap[i + 1]; ++jj) {
                    const I j = Aj[jj];
                    const T a = Ax[jj];
                    for (I kk = Bp[j]; kk < Bp[j + 1]; ++kk) {
                        const I k = Bj[kk];
                        if (spa.mask[k] != tag) {
                            spa.mask[k] = tag;
                            spa.sums[k] = a * Bx[kk];
                            spa.chain.push_back(k);
                        } else {
                            spa.sums[k] += a * Bx[kk];
                        }
                    }
                }
                if (sort_indices) std::sort(spa.chain.begin(), spa.chain.end());
                I dest = Cp[i];
                for (std::size_t idx = 0; idx < spa.chain.size(); ++idx) {
                    const I k = spa.chain[idx];
                    Cj[dest] = k;
                    Cx[dest] = spa.sums[k];
                    ++dest;
                }
            }
        }
    }
}

} // namespace sparse_blas
} // namespace pyamg

#endif // PYAMG_SPARSE_BLAS_SPGEMM_H
