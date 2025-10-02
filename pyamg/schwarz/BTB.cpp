// g++ -O3 -std=c++17 -fvisibility=hidden -shared -fPIC `python3 -m pybind11 --includes` \
/*   */ -o pyamg_blocks$(python3-config --extension-suffix) blocks.cpp

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <complex>
#include <type_traits>
#include <stdexcept>

namespace py = pybind11;

template <class I, class T>
struct CSRView {
    I n_rows = 0, n_cols = 0;
    const I* indptr = nullptr; // size n_rows+1
    const I* indices = nullptr;// size nnz
    const T* data = nullptr;   // size nnz

    inline void row_slice(I r, const I*& cols, const T*& vals, I& len) const {
        I s = indptr[r];
        I e = indptr[r+1];
        cols = indices + s;
        vals = data + s;
        len = e - s;
    }
};

template <class T>
inline T my_conj(const T& x) {
    if constexpr (std::is_same_v<T, std::complex<double>> ||
                  std::is_same_v<T, std::complex<float>>  ||
                  std::is_same_v<T, std::complex<long double>>) {
        return std::conj(x);
    } else {
        return x;
    }
}

/*
Compute small dense diagonal blocks of (BTT_rows^H * B_rows)
without forming a full matrix product. Use identity
A = BTT[rows,:]^* B[rows,:]
    --> A_{p,q} = \sum_{r\inrows} \bar{BTT}(r,p) B(r,q)
where rows is the desired set of overlapping rows over which to
take the outer product. I.e., we view BTT^TB via outer products
between rows of BTT and B. For each selected row r, we only
gather entries whose column is in the overlapping subdomain,
and accumulate the outer product of those two tiny row-slices.

*/
template <class I, class T>
void local_outer_product(
    // B
    I n_rows,
    I n_cols,
    const I Bp[], const int Bp_size, 
    const I Bj[], const int Bj_size,
    const T Bx[], const int Bx_size,
    // BTT
    const I BTTp[], const int BTTp_size,
    const I BTTj[], const int BTTj_size,
    const T BTTx[], const int BTTx_size,
    // Groups: overlapping_rows (flattened CSR-like)
    const I rows_flat[], const int rows_flat_size,
    const I rows_indptr[], const int rows_indptr_size,
    // Groups: overlapping_subdomain (flattened CSR-like)
    const I cols_flat[], const int cols_flat_size,
    const I cols_indptr[], const int cols_indptr_size,
    // Output (preallocated): aux_flat and its indptr
    T aux_flat[], const int aux_flat_size,
    I aux_indptr[], const int aux_indptr_size)
{
    // Views
    CSRView<T> Bv{ n_rows, n_cols, Bp, Bj, Bx };
    CSRView<T> BTTv{ n_rows, n_cols, BTTp, BTTj, BTTx };

    const I* R = &rows_flat[0];
    const I* Rptr = &rows_indptr[0];
    const I* C = &cols_flat[0];
    const I* Cptr = &cols_indptr[0];

    T* out = &aux_flat[0];
    const I* Aptr = &aux_indptr[0];

    const I K = static_cast<I>(rows_indptr.size()) - 1;
    if (K != static_cast<I>(cols_indptr.size()) - 1 ||
        K != static_cast<I>(aux_indptr.size())  - 1) {
        throw std::invalid_argument("rows_indptr, cols_indptr, aux_indptr must have same K");
    }

    // scratch buffers per row
    std::vector<I> left_pos, right_pos; // positions in current group's cols
    std::vector<T> left_val, right_val;
    left_pos.reserve(64); right_pos.reserve(64);
    left_val.reserve(64); right_val.reserve(64);

    // Loop over all groups of overlapping rows and subdomains
    for (I gi = 0; gi < K; ++gi) {
        const I r0 = Rptr[gi], r1 = Rptr[gi+1];
        const I c0 = Cptr[gi], c1 = Cptr[gi+1];
        const I k  = c1 - c0; // Size of this subdomain block
        const I out0 = Aptr[gi], out1 = Aptr[gi+1];

        // basic sanity
        if (k < 0 || out1 - out0 != static_cast<I>(k) * static_cast<I>(k))
            throw std::invalid_argument("aux_indptr must match k_i^2 for each group");

        // Build col->pos map for this group/subdomain
        std::unordered_map<I,I> pos;
        pos.reserve(static_cast<size_t>(k * 2));
        for (I j = 0; j < k; ++j) {
            pos[ C[c0 + j] ] = j;
        }

        // zero the kxk block for this row in aux_flat
        std::fill(out + out0, out + out1, T(0));

        // Accumulate contributions to matrix outer product from each row r
        for (I rr = r0; rr < r1; ++rr) {
            I r = R[rr];

            // gather BTT row r restricted to current group's columns/subdomain
            // (left vector for outer product)
            const I *cbtt;
            const T *vbtt;
            I lbtt;
            BTTv.row_slice(r, cbtt, vbtt, lbtt);

            // Check if nonzero col indices in this row (cbtt)
            // are in subdomain indices (pos)
            left_pos.clear();
            left_val.clear();
            for (I t = 0; t < lbtt; ++t) {
                auto it = pos.find(cbtt[t]);
                if (it != pos.end()) {
                    // where in pos cbtt[t] is located; this returns the
                    // local index of the col->pos map
                    left_pos.push_back(it->second); 
                    left_val.push_back(vbtt[t]); 
                }
            }
            // Zero overlap, skip outer product for this row
            if (left_pos.empty()) continue;

            // gather B row r restricted to current group's columns/subdomain
            // (right vector for outer product)
            const I *cb;
            const T *vb;
            I lb;
            Bv.row_slice(r, cb, vb, lb);

            // Check if nonzero col indices in this row (cbtt)
            // are in subdomain indices (pos)
            right_pos.clear();
            right_val.clear();
            for (I t = 0; t < lb; ++t) {
                auto it = pos.find(cb[t]);
                if (it != pos.end()) {
                    // where in pos cb[t] is located; this returns the
                    // local index of the col->pos map
                    right_pos.push_back(it->second);
                    right_val.push_back(vb[t]);
                }
            }
            // Zero overlap, skip outer product for this row
            if (right_pos.empty()) continue;

            // Accumulate kxk subblock
            //   A_i = (BTT[rows,:]^H * B[rows,:])[cols, cols]
            // via outer prouct A += conj(left) * right^T
            //  --> block[p,q] += conj(left_val[p]) * right_val[q]
            for (size_t a = 0; a < left_pos.size(); ++a) {
                const I rp = left_pos[a];
                const T lc = my_conj(left_val[a]);

                T* rowp = out + out0 + static_cast<I>(rp) * k;
                for (size_t b = 0; b < right_pos.size(); ++b) {
                    const I cq = right_pos[b];
                    rowp[cq] += lc * right_val[b];
                }
            }
        }
    }
}

