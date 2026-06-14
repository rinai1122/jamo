"""Flexible evaluator for the CV / ablation / no-space experiments.

Usage:
    python cv_eval.py --kn KN --corpus CORPUS --pmap PMAP --bench BENCH
                      [--n N] [--disable a,b,c] [--nospace] [--label NAME]

Prints per-config average accuracy and solved (>=90%) count.  With --nospace
the spaces are stripped from each ciphertext (and plaintext) before solving,
to test performance on text without 띄어쓰기.
"""
import argparse
import io
import json
import sys
import time

if not isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from crack import BeamSolver


def accuracy(plain, dec):
    n = min(len(plain), len(dec))
    return sum(a == b for a, b in zip(plain[:n], dec[:n])) / len(plain)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kn", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--pmap", required=True)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--disable", default="")
    ap.add_argument("--nospace", action="store_true")
    ap.add_argument("--label", default="run")
    args = ap.parse_args()

    disable = set(x.strip() for x in args.disable.split(",") if x.strip())
    solver = BeamSolver(kn_model_path=args.kn, corpus_path=args.corpus,
                        disable=disable, pattern_map_cache=args.pmap)

    with open(args.bench, "r", encoding="utf-8") as f:
        cases = json.load(f)
    if args.n:
        cases = cases[:args.n]

    accs = []
    t0 = time.time()
    for case in cases:
        plain = case["plaintext"]
        cipher = case["ciphertext"]
        if args.nospace:
            plain = plain.replace(" ", "")
            cipher = cipher.replace(" ", "")
        dec = solver.solve(cipher)
        accs.append(accuracy(plain, dec))

    avg = sum(accs) / len(accs)
    solved = sum(1 for a in accs if a >= 0.9)
    print(f"[{args.label}] n={len(accs)}  disable={sorted(disable)}  "
          f"nospace={args.nospace}")
    print(f"  avg_acc={avg:.4f}  solved>=90%={solved}/{len(accs)}  "
          f"time={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
