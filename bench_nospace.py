"""Benchmark the no-space hybrid solver on the held-out test set."""
import argparse
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--beam", type=int, default=2000)
    ap.add_argument("--order", type=int, default=6)
    ap.add_argument("--ext", default="greedy")       # first | freq | greedy
    ap.add_argument("--struct", type=float, default=10.0)
    ap.add_argument("--sa", type=int, default=1)       # 1 = use SA, 0 = beam+greedy only
    ap.add_argument("--kn", default="kn_model_cv.json")
    ap.add_argument("--bench", default="benchmark_cv_test.json")
    args = ap.parse_args()

    solver = NoSpaceSolver(kn_model_path=args.kn, beam_width=args.beam,
                           order_cap=args.order, ext_order=args.ext,
                           struct_penalty=args.struct, use_sa=bool(args.sa))
    cases = json.load(open(args.bench, encoding="utf-8"))[:args.n]

    accs = []
    t0 = time.time()
    for i, c in enumerate(cases):
        plain = c["plaintext"].replace(" ", "")
        cipher = c["ciphertext"].replace(" ", "")
        t1 = time.time()
        dec = solver.solve(cipher)
        a = accuracy(plain, dec)
        accs.append(a)
        print(f"[{i+1:2d}/{args.n}] len={len(plain):3d}  acc={a:6.2%}  "
              f"({time.time()-t1:.0f}s)", flush=True)

    avg = sum(accs) / len(accs)
    solved = sum(1 for a in accs if a >= 0.9)
    print(f"\nbeam={args.beam} ext={args.ext} struct={args.struct} sa={args.sa}")
    print(f"NO-SPACE hybrid: avg={avg:.2%}  solved>=90%={solved}/{len(accs)}  "
          f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
