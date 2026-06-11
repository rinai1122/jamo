"""Unified evaluation across all solvers on the current benchmark_set.json.

Usage:
    python eval_all.py [--hc] [--nn] [--cases N] [--regen]

Flags:
    --hc        evaluate HC solvers (baseline, improved, v2)
    --nn        evaluate NN solvers (v4, v5, v6 if .pth exists)
    --cases N   only evaluate on first N cases (default: all)
    --regen     regenerate benchmark_set.json before evaluating

With no flags, evaluates all available solvers.
"""
import sys
import io
import json
import time
import random
import os

if not isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from benchmark import accuracy

BENCH_PATH = "benchmark_set.json"


def regen_benchmark():
    from benchmark import build
    build()


def load_cases(n=None):
    with open(BENCH_PATH, "r", encoding="utf-8") as f:
        cases = json.load(f)
    if n:
        cases = cases[:n]
    return cases


def eval_solver(name, solve_fn, cases):
    accs, lengths = [], []
    t0 = time.time()
    for i, case in enumerate(cases):
        t1 = time.time()
        try:
            dec = solve_fn(case["ciphertext"])
        except Exception as e:
            dec = ""
            print(f"  [{name}] case {i+1} error: {e}", file=sys.stderr)
        acc = accuracy(case["plaintext"], dec)
        accs.append(acc)
        lengths.append(len(case["ciphertext"]))
        if (i + 1) % 10 == 0 or (i + 1) == len(cases):
            avg_so_far = sum(accs) / len(accs)
            print(
                f"  [{name}] {i+1}/{len(cases)} avg={avg_so_far:.2%} "
                f"last={acc:.2%} ({time.time()-t1:.1f}s)",
                flush=True,
            )
    total = time.time() - t0
    avg = sum(accs) / len(accs)
    solved = sum(1 for a in accs if a >= 0.9)
    # Break down by length bucket
    buckets = [(30, 70), (70, 120), (120, 200)]
    bucket_stats = []
    for lo, hi in buckets:
        idxs = [i for i, l in enumerate(lengths) if lo <= l < hi]
        if idxs:
            b_avg = sum(accs[i] for i in idxs) / len(idxs)
            bucket_stats.append(f"len[{lo}-{hi}): {b_avg:.2%} ({len(idxs)})")
    return avg, solved, total, " | ".join(bucket_stats)


def main():
    args = sys.argv[1:]
    do_hc = "--hc" in args or "--nn" not in args
    do_nn = "--nn" in args or "--hc" not in args
    n_cases = None
    if "--cases" in args:
        n_cases = int(args[args.index("--cases") + 1])
    if "--regen" in args:
        print("Regenerating benchmark_set.json ...")
        regen_benchmark()

    cases = load_cases(n_cases)
    print(f"Evaluating on {len(cases)} cases from {BENCH_PATH}")
    lens = [len(c["ciphertext"]) for c in cases]
    print(f"  length: min={min(lens)} max={max(lens)} avg={sum(lens)/len(lens):.0f}")

    results = []

    if do_hc:
        random.seed(42)

        # HC baseline
        print("\n--- HC baseline ---")
        from lean_solver import AdvancedSolver
        sol = AdvancedSolver("kn_model.json", "corpus.txt")
        # memoize
        cache = {}
        orig_get = sol.scorer.get_prob
        def cached(ng, _orig=orig_get):
            p = cache.get(ng)
            if p is None:
                p = _orig(ng)
                cache[ng] = p
            return p
        sol.scorer.get_prob = cached
        avg, solved, t, bkts = eval_solver(
            "HC-baseline",
            lambda ct: sol.solve(ct, restarts=30, iterations=5000),
            cases,
        )
        results.append(("HC-baseline", avg, solved, t, bkts))

        # HC improved (v1)
        print("\n--- HC improved (v1) ---")
        from improved_solver import ImprovedSolver
        sol1 = ImprovedSolver("kn_model.json", "corpus.txt")
        avg, solved, t, bkts = eval_solver(
            "HC-v1",
            lambda ct: sol1.solve(ct, restarts=30, iterations=5000),
            cases,
        )
        results.append(("HC-v1", avg, solved, t, bkts))

        # HC v2
        print("\n--- HC v2 (elitism + reheating) ---")
        from improved_solver_v2 import ImprovedSolverV2
        sol2 = ImprovedSolverV2("kn_model.json", "corpus.txt")
        avg, solved, t, bkts = eval_solver(
            "HC-v2",
            lambda ct: sol2.solve(ct, restarts=30, iterations=5000),
            cases,
        )
        results.append(("HC-v2", avg, solved, t, bkts))

    if do_nn:
        import torch

        D_MODEL, NHEAD, LAYERS = 256, 4, 4

        for tag, pth_file, loader in [
            ("NN-v4", "jamo_nn_v4_fair.pth", "v4"),
            ("NN-v5", "jamo_nn_v5.pth", "v5"),
            ("NN-v6", "jamo_nn_v6.pth", "v6"),
        ]:
            if not os.path.exists(pth_file):
                print(f"\n  [{tag}] skipped — {pth_file} not found")
                continue
            print(f"\n--- {tag} ---")

            if loader == "v4":
                from nn_v4 import RankTransformerCracker, decode_constrained
                m = RankTransformerCracker(D_MODEL, NHEAD, LAYERS)
                m.load_state_dict(torch.load(pth_file, map_location="cpu"))
                m.eval()
                solve_fn = lambda ct, _m=m: decode_constrained(_m, ct)
            elif loader == "v5":
                from nn_v5 import RankTransformerCracker, decode_constrained
                m = RankTransformerCracker(D_MODEL, NHEAD, LAYERS, norm_first=True)
                m.load_state_dict(torch.load(pth_file, map_location="cpu"))
                m.eval()
                solve_fn = lambda ct, _m=m: decode_constrained(_m, ct)
            elif loader == "v6":
                from nn_v6 import DualRankTransformerCracker, decode_constrained
                m = DualRankTransformerCracker(d_model=256, nhead=8, num_layers=6,
                                               max_len=210, norm_first=True)
                m.load_state_dict(torch.load(pth_file, map_location="cpu"))
                m.eval()
                solve_fn = lambda ct, _m=m: decode_constrained(_m, ct)

            avg, solved, t, bkts = eval_solver(tag, solve_fn, cases)
            results.append((tag, avg, solved, t, bkts))

    # Summary table
    print("\n" + "=" * 72)
    print(f"{'Solver':<18} {'Avg Acc':>8} {'>=90%':>7} {'Time(s)':>8}  Buckets")
    print("-" * 72)
    for name, avg, solved, t, bkts in results:
        print(f"{name:<18} {avg:>8.2%} {solved:>6}/{len(cases)}  {t:>7.0f}s  {bkts}")
    print("=" * 72)


if __name__ == "__main__":
    main()
