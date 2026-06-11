"""Evaluate beam_solver (NSMC corpus) on the Wikipedia benchmark.

This is the out-of-distribution (OOD) overfitting check:
  - Solver knowledge base  : corpus.txt     (NSMC movie reviews)
  - Benchmark plaintext     : corpus_wiki.txt (Korean Wikipedia)

A high score means the method generalises; a big drop means it was
leaning on NSMC-specific vocabulary in the pattern map / KN model.
"""
import json
import sys
import time
import io

if not isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from benchmark import accuracy
from beam_solver import BeamSolver

BENCH_PATH = "benchmark_wiki.json"

def main():
    with open(BENCH_PATH, "r", encoding="utf-8") as f:
        cases = json.load(f)

    n = int(sys.argv[1]) if len(sys.argv) > 1 else len(cases)
    cases = cases[:n]

    solver = BeamSolver()   # uses corpus.txt (NSMC) + full_pattern_map.pkl
    accs = []
    t0 = time.time()
    for i, case in enumerate(cases):
        t1 = time.time()
        dec = solver.solve(case["ciphertext"], verbose=True)
        acc = accuracy(case["plaintext"], dec)
        accs.append(acc)
        print(f"[{i+1}/{n}] len={case['length']} acc={acc:.2%} ({time.time()-t1:.1f}s)")
        print(f"  orig: {case['plaintext']}")
        print(f"  dec : {dec}")

    avg  = sum(accs) / len(accs)
    sol  = sum(1 for a in accs if a >= 0.9)
    print(f"\n=== beam (NSMC solver) on Wikipedia benchmark ===")
    print(f"Average accuracy : {avg:.2%}")
    print(f"Cases >=90% (solved) : {sol}/{n}")
    print(f"Total time : {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
