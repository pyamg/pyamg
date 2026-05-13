"""Speedup benchmark for the threaded sparse BLAS PR.

Three sections, each swept across thread counts:

1. **Matvec micro** — scipy ``A @ x`` vs ``pyamg.util.sparse_blas.matvec`` for
   Poisson matrices spanning L1/L2/L3/main memory.
2. **Setup** — ``pyamg.ruge_stuben_solver(A)`` build time. Threaded RAP and
   SpGEMM affect setup, so this is a real sweep target.
3. **Solve** — full residual-tol V-cycle solve on a prebuilt hierarchy.

Workflow for before/after comparison:

    # On main (no threaded kernels):
    python bench/speedup.py --scipy-only --save /tmp/before.json

    # On the PR branch:
    python bench/speedup.py --threads 1,2,4,8 --save /tmp/after.json

    # Print delta tables (markdown, paste into PR):
    python bench/speedup.py --compare /tmp/before.json /tmp/after.json
"""
import argparse
import json
import platform
import subprocess
import time

import numpy as np
import pyamg
from pyamg.gallery import poisson


def best_of(fn, repeats=5, inner=10):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(inner):
            fn()
        times.append((time.perf_counter() - t0) / inner)
    return min(times)


def fmt_time(t):
    if t < 1e-6:
        return f"{t * 1e9:.1f}ns"
    if t < 1e-3:
        return f"{t * 1e6:.1f}us"
    if t < 1.0:
        return f"{t * 1e3:.2f}ms"
    return f"{t:.3f}s"


def matvec_cases():
    return [
        ('Poisson 1D    100k',     poisson((100_000,),       format='csr')),
        ('Poisson 2D    500x500',  poisson((500, 500),       format='csr')),
        ('Poisson 2D  1500x1500',  poisson((1500, 1500),     format='csr')),
        ('Poisson 3D 100x100x100', poisson((100, 100, 100),  format='csr')),
        ('Poisson 3D 200x200x200', poisson((200, 200, 200),  format='csr')),
    ]


def solver_cases():
    return [
        ('Poisson 2D 1500x1500',   poisson((1500, 1500),    format='csr')),
        ('Poisson 3D 50x50x50',    poisson((50, 50, 50),    format='csr')),
        ('Poisson 3D 100x100x100', poisson((100, 100, 100), format='csr')),
    ]


def build_solver(A):
    return pyamg.ruge_stuben_solver(
        A, max_coarse=200,
        CF=('RS', {'second_pass': True}),
        keep=False,
    )


def matvec_bench(threads, scipy_only):
    cases = matvec_cases()
    results = {}
    for name, A in cases:
        x = np.random.default_rng(0).random(A.shape[1])
        _ = A @ x  # warm
        scipy_t = best_of(lambda A=A, x=x: A @ x)
        record = {'nnz': int(A.nnz), 'scipy_time': scipy_t,
                  'kernel_times': None}

        if not scipy_only:
            from pyamg.util.sparse_blas import matvec, set_num_threads
            kt = {}
            _ = matvec(A, x)  # warm
            for t in threads:
                set_num_threads(t)
                _ = matvec(A, x)
                kt[str(t)] = best_of(lambda A=A, x=x: matvec(A, x))
            record['kernel_times'] = kt
        results[name] = record
    return results


def setup_bench(threads, scipy_only):
    cases = solver_cases()
    results = {}
    for name, A in cases:
        record = {'shape': list(A.shape), 'nnz': int(A.nnz),
                  'thread_times': {}, 'n_levels': None}

        def time_one():
            t0 = time.perf_counter()
            ml = build_solver(A)
            return time.perf_counter() - t0, len(ml.levels)

        if scipy_only:
            _ = build_solver(A)  # warm
            ts = [time_one()[0] for _ in range(3)]
            _, nl = time_one()
            record['thread_times']['scipy'] = min(ts)
            record['n_levels'] = nl
        else:
            from pyamg.util.sparse_blas import set_num_threads
            nl_seen = None
            for t in threads:
                set_num_threads(t)
                _ = build_solver(A)  # warm
                ts = []
                for _ in range(3):
                    bt, nl = time_one()
                    ts.append(bt)
                    nl_seen = nl
                record['thread_times'][str(t)] = min(ts)
            record['n_levels'] = nl_seen
        results[name] = record
    return results


def solve_bench(threads, scipy_only):
    cases = solver_cases()
    results = {}
    rng = np.random.default_rng(0)
    for name, A in cases:
        b = rng.random(A.shape[0])
        x0 = np.zeros(A.shape[0])
        ml = build_solver(A)
        record = {'shape': list(A.shape), 'nnz': int(A.nnz),
                  'n_levels': len(ml.levels), 'thread_times': {},
                  'iters': {}, 'final_residual': {}}

        def time_one():
            res = []
            t0 = time.perf_counter()
            ml.solve(b.copy(), x0=x0.copy(), tol=1e-10, maxiter=50,
                     residuals=res)
            return time.perf_counter() - t0, len(res) - 1, float(res[-1])

        if scipy_only:
            _ = time_one()
            samples = [time_one() for _ in range(3)]
            best = min(samples, key=lambda s: s[0])
            record['thread_times']['scipy']  = best[0]
            record['iters']['scipy']         = best[1]
            record['final_residual']['scipy'] = best[2]
        else:
            from pyamg.util.sparse_blas import set_num_threads
            for t in threads:
                set_num_threads(t)
                _ = time_one()
                samples = [time_one() for _ in range(3)]
                best = min(samples, key=lambda s: s[0])
                record['thread_times'][str(t)]  = best[0]
                record['iters'][str(t)]         = best[1]
                record['final_residual'][str(t)] = best[2]
        results[name] = record
    return results


def metadata(scipy_only, threads):
    sha = subprocess.run(['git', 'rev-parse', 'HEAD'],
                         capture_output=True, text=True,
                         cwd=__file__.rsplit('/', 2)[0]).stdout.strip()
    branch = subprocess.run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                            capture_output=True, text=True,
                            cwd=__file__.rsplit('/', 2)[0]).stdout.strip()
    return {
        'mode': 'scipy-only' if scipy_only else 'kernel',
        'threads': threads,
        'git_sha': sha,
        'git_branch': branch,
        'pyamg_version': pyamg.__version__,
        'numpy_version': np.__version__,
        'platform': platform.platform(),
        'cpu_count': platform.os.cpu_count(),
    }


def print_inline_matvec(results, threads, scipy_only):
    print('# Matvec micro-benchmark (best-of-5, inner=10)')
    header = f'{"problem":<26} {"nnz":>10} {"scipy":>9}'
    if not scipy_only:
        header += ' ' + ' '.join(f'thr={t:<5}' for t in threads)
    print(header)
    print('-' * len(header))
    for name, r in results.items():
        line = f'{name:<26} {r["nnz"]:>10} {fmt_time(r["scipy_time"]):>9}'
        if not scipy_only and r['kernel_times']:
            line += ' ' + ' '.join(
                f'{r["scipy_time"] / r["kernel_times"][str(t)]:>7.2f}x'
                for t in threads)
        print(line)
    print()


def print_inline_solver(label, results, threads, scipy_only):
    print(f'# {label} (best of 3 wall-clock measurements)')
    header = f'{"problem":<26} {"nnz":>10}'
    if scipy_only:
        header += f'  {"scipy":>10}'
    else:
        header += '  ' + ' '.join(f'thr={t:<7}' for t in threads)
    print(header)
    print('-' * len(header))
    for name, r in results.items():
        line = f'{name:<26} {r["nnz"]:>10}'
        if scipy_only:
            line += f'  {fmt_time(r["thread_times"]["scipy"]):>10}'
        else:
            line += '  ' + ' '.join(
                f'{fmt_time(r["thread_times"][str(t)]):>9}'
                for t in threads)
        print(line)
    print()


def cmd_run(args):
    threads = [int(t) for t in args.threads.split(',')]
    payload = {
        'metadata': metadata(args.scipy_only, threads),
        'matvec': matvec_bench(threads, args.scipy_only),
        'setup':  setup_bench(threads, args.scipy_only),
        'solve':  solve_bench(threads, args.scipy_only),
    }
    print_inline_matvec(payload['matvec'], threads, args.scipy_only)
    print_inline_solver('Setup',  payload['setup'],  threads, args.scipy_only)
    print_inline_solver('Solve',  payload['solve'],  threads, args.scipy_only)
    if args.save:
        with open(args.save, 'w') as f:
            json.dump(payload, f, indent=2)
        print(f'\nWrote {args.save}')


def cmd_compare(args):
    with open(args.before) as f:
        before = json.load(f)
    with open(args.after) as f:
        after = json.load(f)
    threads = after['metadata']['threads']

    print(f'# Before: `{before["metadata"]["git_branch"]}` '
          f'({before["metadata"]["git_sha"][:8]}, {before["metadata"]["mode"]})')
    print(f'# After:  `{after["metadata"]["git_branch"]}` '
          f'({after["metadata"]["git_sha"][:8]}, {after["metadata"]["mode"]})')
    print(f'# Host:   {after["metadata"]["platform"]}, '
          f'{after["metadata"]["cpu_count"]} CPUs\n')

    def base(rec, key):
        tt = rec.get('thread_times') or {}
        return tt.get('scipy') or tt.get('1') or rec.get('scipy_time')

    print('## Matvec speedup vs scipy A @ x\n')
    cols = ['problem', 'nnz', 'scipy'] + [f'thr={t}' for t in threads]
    print('| ' + ' | '.join(cols) + ' |')
    print('|' + '|'.join(['---'] * len(cols)) + '|')
    for name in after['matvec']:
        a = after['matvec'][name]
        b = before['matvec'].get(name, {})
        scipy_ref = b.get('scipy_time') or a['scipy_time']
        row = [name, f'{a["nnz"]}', fmt_time(scipy_ref)]
        for t in threads:
            kt = a['kernel_times'][str(t)]
            row.append(f'{scipy_ref / kt:.2f}x ({fmt_time(kt)})')
        print('| ' + ' | '.join(row) + ' |')
    print()

    for label, key in (('Setup', 'setup'), ('Solve', 'solve')):
        print(f'## {label} time and speedup vs `{before["metadata"]["mode"]}`\n')
        cols = ['problem', 'nnz', f'before ({before["metadata"]["mode"]})'] \
               + [f'thr={t}' for t in threads]
        print('| ' + ' | '.join(cols) + ' |')
        print('|' + '|'.join(['---'] * len(cols)) + '|')
        for name in after[key]:
            a = after[key][name]
            b = before[key].get(name)
            ref = base(b, key) if b else None
            row = [name, f'{a["nnz"]}',
                   fmt_time(ref) if ref else 'n/a']
            for t in threads:
                tt = a['thread_times'][str(t)]
                spd = f'{ref / tt:.2f}x' if ref else 'n/a'
                row.append(f'{fmt_time(tt)} ({spd})')
            print('| ' + ' | '.join(row) + ' |')
        print()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='subcommand')
    ap.set_defaults(subcommand='run')

    ap.add_argument('--threads', default='1,2,4,8',
                    help='comma-separated thread counts (kernel mode only)')
    ap.add_argument('--scipy-only', action='store_true',
                    help='record scipy/default-pyamg baseline; safe on main')
    ap.add_argument('--save', default=None,
                    help='write results to this JSON path')

    cmp_ap = sub.add_parser('compare', help='print before/after delta tables')
    cmp_ap.add_argument('before', help='JSON from --scipy-only run')
    cmp_ap.add_argument('after',  help='JSON from PR-branch run')

    args = ap.parse_args()
    if args.subcommand == 'compare':
        cmd_compare(args)
    else:
        cmd_run(args)


if __name__ == '__main__':
    main()
