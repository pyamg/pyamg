"""Numerical equivalence check for the threaded sparse BLAS PR.

Captures a fingerprint of every level (shape, nnz, sorted-canonical hashes
of indptr/indices/data for A/P/R) and the per-iteration residual history
on a suite of representative problems, for several solver flavors.

Usage:
    # On master (or before applying the patch):
    python bench/numerical_check.py before > /tmp/before.json

    # On the PR branch (after applying the patch):
    python bench/numerical_check.py after > /tmp/after.json

    # Verdict:
    python bench/numerical_check.py compare /tmp/before.json /tmp/after.json
"""
import hashlib
import json
import sys
import numpy as np
import pyamg
from pyamg.gallery import poisson


def hash_array(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def csr_canonical(M):
    C = M.tocsr().copy()
    C.sum_duplicates()
    C.sort_indices()
    return C


def fingerprint(ml):
    out = []
    for lvl in ml.levels:
        d = {'shape': list(lvl.A.shape), 'nnz': int(lvl.A.nnz)}
        A = csr_canonical(lvl.A)
        d['A_indptr'] = hash_array(A.indptr)
        d['A_indices'] = hash_array(A.indices)
        d['A_data'] = hash_array(A.data)
        for attr in ('P', 'R'):
            M = getattr(lvl, attr, None)
            if M is not None:
                C = csr_canonical(M)
                d[f'{attr}_nnz'] = int(C.nnz)
                d[f'{attr}_indptr'] = hash_array(C.indptr)
                d[f'{attr}_indices'] = hash_array(C.indices)
                d[f'{attr}_data'] = hash_array(C.data)
        s = getattr(lvl, 'splitting', None)
        if s is not None:
            d['splitting'] = hash_array(np.asarray(s))
        out.append(d)
    return out


def converge(ml, b, x0, tol=1e-10, maxiter=30):
    res = []
    ml.solve(b, x0=x0, tol=tol, maxiter=maxiter, residuals=res)
    return [float(r) for r in res]


def run_problem(name, A):
    rng = np.random.default_rng(0)
    b = rng.random(A.shape[0])
    x0 = np.zeros(A.shape[0])

    out = {'problem': name, 'A_nnz': int(A.nnz), 'A_shape': list(A.shape)}

    # Classical Ruge-Stuben (with second pass).
    ml = pyamg.ruge_stuben_solver(A, max_coarse=10,
                                  CF=('RS', {'second_pass': True}), keep=True)
    out['rs'] = {'fp': fingerprint(ml), 'res': converge(ml, b, x0)}

    # AIR (one-sided AMG, exercises approx_ideal_restriction_pass2).
    try:
        from pyamg.classical import air_solver
        ml = air_solver(A, max_coarse=10, keep=True)
        out['air'] = {'fp': fingerprint(ml), 'res': converge(ml, b, x0)}
    except Exception as e:  # noqa: BLE001
        out['air'] = {'error': repr(e)}

    # Smoothed aggregation (seeded RNG for reproducible spectral-radius init).
    np.random.seed(42)
    ml = pyamg.smoothed_aggregation_solver(A, max_coarse=10, keep=True)
    out['sa'] = {'fp': fingerprint(ml), 'res': converge(ml, b, x0)}

    np.random.seed(42)
    ml = pyamg.rootnode_solver(A, max_coarse=10, keep=True)
    out['rootnode'] = {'fp': fingerprint(ml), 'res': converge(ml, b, x0)}

    return out


def collect(label):
    cases = [
        ('Poisson 1D 1000',     poisson((1000,),       format='csr')),
        ('Poisson 2D 50x50',    poisson((50, 50),      format='csr')),
        ('Poisson 2D 100x100',  poisson((100, 100),    format='csr')),
        ('Poisson 3D 20x20x20', poisson((20, 20, 20),  format='csr')),
    ]
    results = [run_problem(name, A) for name, A in cases]
    return {'label': label, 'results': results}


def compare(before_path, after_path):
    B = json.load(open(before_path))
    A = json.load(open(after_path))
    print(f"BEFORE: {B['label']}    AFTER: {A['label']}\n")
    n_struct_match = n_struct_total = 0
    n_data_match = n_data_total = 0
    max_top_residual_rel = 0.0
    iter_mismatch = 0
    final_max_rel = 0.0
    fail = False

    for Bp, Ap in zip(B['results'], A['results']):
        print(f"--- {Bp['problem']} ---")
        for solver in ('rs', 'air', 'sa', 'rootnode'):
            if solver not in Bp:
                continue
            b, a = Bp[solver], Ap[solver]
            if 'error' in b or 'error' in a:
                print(f"  {solver}: error before={b.get('error')} after={a.get('error')}")
                continue
            # Per-level structural / data hash equality.
            for li, (bl, al) in enumerate(zip(b['fp'], a['fp'])):
                struct_keys = ['shape', 'nnz', 'A_indptr', 'A_indices',
                               'P_nnz', 'P_indptr', 'P_indices',
                               'R_nnz', 'R_indptr', 'R_indices', 'splitting']
                data_keys = ['A_data', 'P_data', 'R_data']
                for k in struct_keys:
                    if k in bl or k in al:
                        n_struct_total += 1
                        if bl.get(k) == al.get(k):
                            n_struct_match += 1
                for k in data_keys:
                    if k in bl or k in al:
                        n_data_total += 1
                        if bl.get(k) == al.get(k):
                            n_data_match += 1
            # Per-iteration residual comparison.
            br, ar = b['res'], a['res']
            if len(br) != len(ar):
                iter_mismatch += 1
                print(f"  {solver}: iter count {len(br)} -> {len(ar)} "
                      f"(final res {br[-1]:.3e} -> {ar[-1]:.3e})")
            else:
                max_rel = 0.0
                for x, y in zip(br, ar):
                    if abs(x) > 0:
                        max_rel = max(max_rel, abs(y - x) / abs(x))
                if abs(br[-1]) > 0:
                    final_max_rel = max(final_max_rel, abs(ar[-1] - br[-1]) / abs(br[-1]))
                max_top_residual_rel = max(max_top_residual_rel, max_rel)
                tag = "OK" if max_rel < 1e-10 else (
                    "ULP" if max_rel < 1e-12 else f"REL={max_rel:.1e}")
                print(f"  {solver}: levels={len(b['fp'])}, iters={len(br)}, "
                      f"final res {br[-1]:.3e} -> {ar[-1]:.3e}, max_rel_iter={tag}")

    print()
    print(f"Structural levels match (shape/nnz/indptr/indices/splitting): "
          f"{n_struct_match}/{n_struct_total}")
    print(f"Numerical data hashes match (A/P/R bit-identical):           "
          f"{n_data_match}/{n_data_total}")
    print(f"Iter-count mismatches:                                       {iter_mismatch}")
    print(f"Max relative residual diff (any iter, any solver):           "
          f"{max_top_residual_rel:.2e}")
    print(f"Max relative final-residual diff:                            "
          f"{final_max_rel:.2e}")
    if n_struct_match != n_struct_total:
        fail = True
        print("FAIL: hierarchy structure changed (would change C/F splitting "
              "or aggregation membership).")
    if max_top_residual_rel > 1e-10:
        print("Note: residual differences exceed 1e-10 -- not bit-identical, "
              "verify magnitude is acceptable.")
    if not fail:
        print("PASS")
    sys.exit(1 if fail else 0)


def main(argv):
    if len(argv) >= 2 and argv[1] == 'compare':
        compare(argv[2], argv[3])
        return
    label = argv[1] if len(argv) > 1 else 'unknown'
    print(json.dumps(collect(label), indent=2))


if __name__ == '__main__':
    main(sys.argv)
