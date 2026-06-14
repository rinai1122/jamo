"""Test whether more search budget recovers the short-case SEARCH failures.

The diagnostic showed cases 2, 9, 11 have the true key as a strictly higher-
fitness optimum that the search missed.  Throw more SA at short cases and see
if accuracy recovers.
"""
import io
import json
import sys
import time

if not isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from nospace_solver import NoSpaceSolver


def accuracy(plain, dec):
    n = min(len(plain), len(dec))
    return sum(a == b for a, b in zip(plain[:n], dec[:n])) / len(plain)


cases = json.load(open("benchmark_cv_test.json", encoding="utf-8"))[:15]
focus = [2, 9, 11, 10, 15, 1, 8]   # 1-indexed short/failing cases

configs = [
    ("base",  dict(sa_restarts=12,  sa_iters=4000)),
    ("big",   dict(sa_restarts=60,  sa_iters=8000)),
    ("huge",  dict(sa_restarts=150, sa_iters=12000)),
]

solvers = {name: NoSpaceSolver(kn_model_path="kn_model_cv.json", **kw)
           for name, kw in configs}

print("case len  " + "  ".join(f"{n:>14}" for n, _ in configs))
for idx in focus:
    c = cases[idx - 1]
    plain = c["plaintext"].replace(" ", "")
    cipher = c["ciphertext"].replace(" ", "")
    row = []
    for name, _ in configs:
        t0 = time.time()
        dec = solvers[name].solve(cipher)
        row.append(f"{accuracy(plain, dec):6.1%}({time.time()-t0:4.0f}s)")
    print(f"{idx:2d} {len(plain):4d}  " + "  ".join(f"{r:>14}" for r in row),
          flush=True)
