"""Parallel no-space solver evaluation on the FULL held-out test benchmark.

Runs `NoSpaceSolver` over every case in `benchmark_cv_test.json` (100 cases,
not the old 15) using a multiprocessing pool, and reports accuracy broken down
by length bucket -- the short-case zone is where no-space solving is hard, so
the breakdown is what matters, not just the global average.

Usage:
    python ns_eval.py [N] [workers] [tag]

`N` limits to the first N cases (default: all).  Results (per-case acc, time,
length) are written to  ns_result_<tag>.json  for diffing across solver
versions.
"""
import io
import json
import os
import sys
import time
from multiprocessing import Pool

if not isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# Solver kwargs -- override via SOLVER_KWARGS env (JSON) so the harness can be
# reused for ablations / tuning without editing this file.
SOLVER_KWARGS = json.loads(os.environ.get("SOLVER_KWARGS", "{}"))

_solver = None


def _init():
    global _solver
    from nospace_solver import NoSpaceSolver
    _solver = NoSpaceSolver(kn_model_path="kn_model_cv.json", **SOLVER_KWARGS)


def _accuracy(plain, dec):
    n = min(len(plain), len(dec))
    return sum(a == b for a, b in zip(plain[:n], dec[:n])) / len(plain)


def _true_key_decrypt(cipher, plain):
    """Decrypt `cipher` under the ground-truth key recovered from the aligned
    spaceless (cipher, plain) pair -- i.e. the best the solver could possibly
    output."""
    m = {}
    for c, p in zip(cipher, plain):
        m[c] = p
    return "".join(m.get(ch, ch) for ch in cipher)


def _solve_one(args):
    i, plain, cipher = args
    t0 = time.time()
    out = _solver.solve(cipher)
    dt = time.time() - t0
    # Search-vs-objective diagnostic: compare the fitness the solver landed on
    # to the fitness of the *true* key.  f_true > f_out  => the objective ranks
    # truth higher but the search missed it (a SEARCH failure, fixable by more
    # search);  f_out >= f_true with low acc => a wrong key out-scores truth (an
    # OBJECTIVE failure, only fixable by a better scoring function).
    f_out = _solver._fitness(out)
    f_true = _solver._fitness(_true_key_decrypt(cipher, plain))
    return i, len(plain), _accuracy(plain, out), dt, f_out, f_true, out


def _bucket(n):
    return ("<30" if n < 30 else "30-44" if n < 45 else "45-59" if n < 60
            else "60-94" if n < 95 else "95+")


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else min(8, os.cpu_count())
    tag = sys.argv[3] if len(sys.argv) > 3 else "baseline"

    cases = json.load(open("benchmark_cv_test.json", encoding="utf-8"))
    if N:
        cases = cases[:N]
    work = [(i, c["plaintext"].replace(" ", ""), c["ciphertext"].replace(" ", ""))
            for i, c in enumerate(cases)]

    print(f"tag={tag}  N={len(work)}  workers={workers}  kwargs={SOLVER_KWARGS}",
          flush=True)
    t_start = time.time()
    results = []
    with Pool(workers, initializer=_init) as pool:
        for rec in pool.imap_unordered(_solve_one, work):
            i, ln, acc, dt, f_out, f_true, out = rec
            results.append(rec)
            flag = "  SEARCH-fail" if (acc < 0.9 and f_true > f_out + 1e-6) else \
                   "  OBJ-fail" if acc < 0.9 else ""
            print(f"  [{len(results):3d}/{len(work)}] case={i:3d} len={ln:3d} "
                  f"acc={acc:6.2%} t={dt:5.1f}s dFit={f_out-f_true:+7.1f}{flag}",
                  flush=True)
    wall = time.time() - t_start

    results.sort()
    accs = [r[2] for r in results]
    avg = sum(accs) / len(accs)
    solved = sum(1 for a in accs if a >= 0.9)
    print(f"\n=== {tag} ===")
    print(f"avg={avg:.2%}  solved={solved}/{len(accs)}  wall={wall:.0f}s")

    # Per-bucket breakdown.
    buckets = {}
    for r in results:
        buckets.setdefault(_bucket(r[1]), []).append(r[2])
    print("bucket      n   avg     solved")
    for b in ["<30", "30-44", "45-59", "60-94", "95+"]:
        if b in buckets:
            a = buckets[b]
            print(f"  {b:6s} {len(a):3d}  {sum(a)/len(a):6.2%}  "
                  f"{sum(1 for x in a if x >= 0.9):3d}/{len(a)}")

    # Failure split: of the unsolved cases, how many are search vs objective?
    fails = [r for r in results if r[2] < 0.9]
    search_fail = [r for r in fails if r[5] > r[4] + 1e-6]
    obj_fail = [r for r in fails if r[5] <= r[4] + 1e-6]
    print(f"\nfailures (<90%): {len(fails)}  "
          f"search-fail(true key scores higher, fixable): {len(search_fail)}  "
          f"obj-fail(wrong key out-scores truth): {len(obj_fail)}")
    if search_fail:
        print("  search-fail cases:", sorted((r[0], r[1], round(r[2], 2),
              round(r[4] - r[5], 1)) for r in search_fail))
    if obj_fail:
        print("  obj-fail cases:   ", sorted((r[0], r[1], round(r[2], 2),
              round(r[4] - r[5], 1)) for r in obj_fail))

    json.dump({"tag": tag, "avg": avg, "solved": solved, "n": len(accs),
               "wall": wall, "kwargs": SOLVER_KWARGS,
               "cases": [{"i": i, "len": ln, "acc": acc, "t": dt,
                          "f_out": f_out, "f_true": f_true, "out": out}
                         for i, ln, acc, dt, f_out, f_true, out in results]},
              open(f"ns_result_{tag}.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
