"""Ablation study for the no-space solver: disable one stage / fitness term at
a time and measure the accuracy cost on the full benchmark.

All configs share one fixed *reduced* search budget (set below) so they are
directly comparable and the whole sweep finishes in a reasonable time -- the
goal is the *relative* contribution of each module, not absolute accuracy (the
shipped solver uses a larger budget; the winning config is re-validated there).

Run:  python ns_ablation.py [N] [workers]
Writes ns_ablation.json and prints a table sorted by accuracy cost.
"""
import io
import json
import os
import sys
import time
from multiprocessing import Pool

if not isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# Fixed reduced budget shared by every config (fast + comparable).
BUDGET = dict(beam_width=800, sa_restarts=6, sa_iters=1500,
              short_sa_mult=2, short_iter_mult=1, short_sa_passes=2,
              beam_seeds=5)

# (name, extra-kwargs-overriding-BUDGET).  "full" is the reference.
CONFIGS = [
    ("full",        {}),
    ("-kn",         {"disable": {"kn"}}),
    ("-struct",     {"disable": {"struct"}}),
    ("-ws",         {"disable": {"ws"}}),
    ("-beam",       {"disable": {"beam"}}),
    ("-greedy",     {"disable": {"greedy"}}),
    ("-base_sa",    {"disable": {"base_sa"}}),
    ("-ensemble",   {"disable": {"ensemble"}}),
    ("-multiseed",  {"beam_seeds": 1}),
]

_solver = None


def _init(kwargs):
    global _solver
    from nospace_solver import NoSpaceSolver
    _solver = NoSpaceSolver(kn_model_path="kn_model_cv.json", **kwargs)


def _acc(plain, dec):
    n = min(len(plain), len(dec))
    return sum(a == b for a, b in zip(plain[:n], dec[:n])) / len(plain)


def _solve_one(args):
    i, plain, cipher = args
    return _acc(plain, _solver.solve(cipher))


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else min(6, os.cpu_count())

    cases = json.load(open("benchmark_cv_test.json", encoding="utf-8"))
    if N:
        cases = cases[:N]
    work = [(i, c["plaintext"].replace(" ", ""), c["ciphertext"].replace(" ", ""))
            for i, c in enumerate(cases)]

    print(f"ablation: N={len(work)} workers={workers} budget={BUDGET}\n", flush=True)
    results = {}
    for name, extra in CONFIGS:
        kwargs = {**BUDGET, **extra}
        t0 = time.time()
        with Pool(workers, initializer=_init, initargs=(kwargs,)) as pool:
            accs = list(pool.imap_unordered(_solve_one, work))
        avg = sum(accs) / len(accs)
        solved = sum(1 for a in accs if a >= 0.9)
        results[name] = {"avg": avg, "solved": solved, "wall": time.time() - t0}
        print(f"  {name:12s} avg={avg:6.2%} solved={solved:3d}/{len(accs)} "
              f"({results[name]['wall']:.0f}s)", flush=True)

    full = results["full"]["avg"]
    print("\n=== ablation (accuracy cost of removing each module) ===")
    print(f"{'config':12s} {'avg':>7s} {'solved':>7s} {'Δ vs full':>10s}")
    rows = sorted(results.items(), key=lambda kv: kv[1]["avg"])
    for name, r in rows:
        d = "" if name == "full" else f"{r['avg'] - full:+.2%}"
        print(f"{name:12s} {r['avg']:7.2%} {r['solved']:5d}   {d:>10s}")

    json.dump({"budget": BUDGET, "n": len(work), "results": results},
              open("ns_ablation.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\nMost negative Δ = most essential module.  ~0 Δ = removable.")


if __name__ == "__main__":
    main()
