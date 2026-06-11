"""Evaluate a hill-climbing solver on the fixed benchmark set.

Usage: python eval_hc.py baseline|improved [num_cases]
"""
import json
import sys
import io
import time
import random

if not isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from benchmark import accuracy


def memoize_scorer(scorer):
    """Transparent cache for KN probability lookups (same values, just fast)."""
    cache = {}
    orig = scorer.get_prob

    def cached(ngram):
        p = cache.get(ngram)
        if p is None:
            p = orig(ngram)
            cache[ngram] = p
        return p

    scorer.get_prob = cached
    return scorer


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    with open("benchmark_set.json", "r", encoding="utf-8") as f:
        cases = json.load(f)
    if len(sys.argv) > 2:
        cases = cases[: int(sys.argv[2])]

    if which == "baseline":
        from lean_solver import AdvancedSolver
        solver = AdvancedSolver("kn_model.json", "corpus.txt")
        memoize_scorer(solver.scorer)
        solve = lambda ct: solver.solve(ct, restarts=30, iterations=5000)
    elif which == "v2":
        from improved_solver_v2 import ImprovedSolverV2
        solver = ImprovedSolverV2("kn_model.json", "corpus.txt")
        solve = lambda ct: solver.solve(ct, restarts=30, iterations=5000)
    else:
        from improved_solver import ImprovedSolver
        solver = ImprovedSolver("kn_model.json", "corpus.txt")
        solve = lambda ct: solver.solve(ct, restarts=30, iterations=5000)

    random.seed(42)
    accs = []
    t0 = time.time()
    for i, case in enumerate(cases):
        t1 = time.time()
        dec = solve(case["ciphertext"])
        acc = accuracy(case["plaintext"], dec)
        accs.append(acc)
        print(f"[{i+1}/{len(cases)}] acc={acc:.2%} ({time.time()-t1:.1f}s)")
        print(f"  orig: {case['plaintext']}")
        print(f"  dec:  {dec}")

    avg = sum(accs) / len(accs)
    solved = sum(1 for a in accs if a >= 0.9)
    print(f"\n=== {which} ===")
    print(f"Average accuracy: {avg:.2%}")
    print(f"Cases >=90% (solved): {solved}/{len(accs)}")
    print(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
