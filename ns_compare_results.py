"""Diff two ns_eval result files (e.g. baseline vs improved no-space solver).

Usage:  python ns_compare_results.py ns_result_baseline.json ns_result_v2.json

Prints a per-length-bucket before/after table, the overall average / solved
deltas, every per-case regression and improvement, and the search-vs-objective
failure split for the second (improved) run -- so it is clear not just *that*
the number moved but *which* cases moved and what kind of failure remains.
"""
import io
import json
import sys

if not isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def bucket(n):
    return ("<30" if n < 30 else "30-44" if n < 45 else "45-59" if n < 60
            else "60-94" if n < 95 else "95+")


def load(path):
    d = json.load(open(path, encoding="utf-8"))
    return d, {c["i"]: c for c in d["cases"]}


def main():
    pa, pb = sys.argv[1], sys.argv[2]
    da, A = load(pa)
    db, B = load(pb)
    common = sorted(set(A) & set(B))

    print(f"A = {da['tag']:12s} ({pa})")
    print(f"B = {db['tag']:12s} ({pb})")
    print(f"common cases: {len(common)}\n")

    # Per-bucket before/after.
    order = ["<30", "30-44", "45-59", "60-94", "95+"]
    rows = {b: {"n": 0, "a": 0.0, "b": 0.0, "sa": 0, "sb": 0} for b in order}
    for i in common:
        bk = bucket(A[i]["len"])
        r = rows[bk]
        r["n"] += 1
        r["a"] += A[i]["acc"]
        r["b"] += B[i]["acc"]
        r["sa"] += A[i]["acc"] >= 0.9
        r["sb"] += B[i]["acc"] >= 0.9
    print(f"{'bucket':7s} {'n':>3s}  {'A avg':>7s} {'B avg':>7s} {'Δavg':>7s}  "
          f"{'A solv':>7s} {'B solv':>7s}")
    for b in order:
        r = rows[b]
        if r["n"]:
            print(f"{b:7s} {r['n']:3d}  {r['a']/r['n']:7.2%} {r['b']/r['n']:7.2%} "
                  f"{(r['b']-r['a'])/r['n']:+7.2%}  {r['sa']:4d}/{r['n']:<2d} "
                  f"{r['sb']:4d}/{r['n']:<2d}")

    aa = sum(A[i]["acc"] for i in common) / len(common)
    ab = sum(B[i]["acc"] for i in common) / len(common)
    sa = sum(A[i]["acc"] >= 0.9 for i in common)
    sb = sum(B[i]["acc"] >= 0.9 for i in common)
    print(f"\nOVERALL  A: avg={aa:.2%} solved={sa}/{len(common)}   "
          f"B: avg={ab:.2%} solved={sb}/{len(common)}   "
          f"Δavg={ab-aa:+.2%} Δsolved={sb-sa:+d}")

    # Per-case movers.
    movers = sorted(((B[i]["acc"] - A[i]["acc"], i) for i in common),
                    key=lambda x: x[0])
    regr = [(d, i) for d, i in movers if d < -0.01]
    impr = [(d, i) for d, i in movers if d > 0.01]
    print(f"\nregressions: {len(regr)}   improvements: {len(impr)}")
    if regr:
        print("  REGRESSED (case len: A%->B%):")
        for d, i in regr:
            print(f"    case {i:3d} len {A[i]['len']:3d}: "
                  f"{A[i]['acc']:.1%} -> {B[i]['acc']:.1%} ({d:+.1%})")
    if impr:
        print("  IMPROVED (case len: A%->B%):")
        for d, i in reversed(impr):
            print(f"    case {i:3d} len {A[i]['len']:3d}: "
                  f"{A[i]['acc']:.1%} -> {B[i]['acc']:.1%} ({d:+.1%})")

    # Failure split for B (needs f_out/f_true present).
    if all("f_out" in B[i] for i in common):
        fails = [i for i in common if B[i]["acc"] < 0.9]
        sfail = [i for i in fails if B[i]["f_true"] > B[i]["f_out"] + 1e-6]
        ofail = [i for i in fails if i not in sfail]
        print(f"\nB failures (<90%): {len(fails)}  "
              f"search-fail(fixable by more search): {len(sfail)}  "
              f"obj-fail(needs better objective): {len(ofail)}")
        if sfail:
            print("  search-fail:", [(i, B[i]["len"], round(B[i]["acc"], 2),
                  round(B[i]["f_out"] - B[i]["f_true"], 1)) for i in sfail])
        if ofail:
            print("  obj-fail:   ", [(i, B[i]["len"], round(B[i]["acc"], 2),
                  round(B[i]["f_out"] - B[i]["f_true"], 1)) for i in ofail])


if __name__ == "__main__":
    main()
