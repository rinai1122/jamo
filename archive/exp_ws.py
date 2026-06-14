"""Sweep the word-segmentation weight on the held-out no-space test set."""
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
weights = [0.0, 0.3, 0.5, 1.0]
solvers = {w: NoSpaceSolver(kn_model_path="kn_model_cv.json", ws_weight=w)
           for w in weights}

results = {w: [] for w in weights}
print("case len  " + "  ".join(f"ws={w:<5}" for w in weights))
for i, c in enumerate(cases):
    plain = c["plaintext"].replace(" ", "")
    cipher = c["ciphertext"].replace(" ", "")
    row = []
    for w in weights:
        dec = solvers[w].solve(cipher)
        a = accuracy(plain, dec)
        results[w].append(a)
        row.append(f"{a:6.1%}")
    print(f"{i+1:2d} {len(plain):4d}  " + "  ".join(f"{r:>7}" for r in row),
          flush=True)

print("\n           " + "  ".join(f"ws={w:<5}" for w in weights))
avg = "  ".join(f"{sum(results[w])/len(results[w]):6.1%}" for w in weights)
sol = "  ".join(f"{sum(1 for a in results[w] if a>=0.9):>4d}/15" for w in weights)
print(f"avg        " + avg)
print(f"solved>=90 " + sol)
