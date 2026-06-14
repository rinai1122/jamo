"""Validate ws=0.3 vs ws=0 on a larger held-out no-space sample."""
import io
import json
import sys
import time

if not isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from nospace_solver import NoSpaceSolver

N = int(sys.argv[1]) if len(sys.argv) > 1 else 50


def accuracy(plain, dec):
    n = min(len(plain), len(dec))
    return sum(a == b for a, b in zip(plain[:n], dec[:n])) / len(plain)


cases = json.load(open("benchmark_cv_test.json", encoding="utf-8"))[:N]
weights = [0.0, 0.3]
solvers = {w: NoSpaceSolver(kn_model_path="kn_model_cv.json", ws_weight=w)
           for w in weights}

results = {w: [] for w in weights}
t0 = time.time()
for i, c in enumerate(cases):
    plain = c["plaintext"].replace(" ", "")
    cipher = c["ciphertext"].replace(" ", "")
    row = []
    for w in weights:
        dec = solvers[w].solve(cipher)
        a = accuracy(plain, dec)
        results[w].append(a)
        row.append(f"{a:6.1%}")
    print(f"[{i+1:3d}/{N}] len={len(plain):3d}  ws0={row[0]}  ws0.3={row[1]}",
          flush=True)

lines = []
for w in weights:
    avg = sum(results[w]) / len(results[w])
    sol = sum(1 for a in results[w] if a >= 0.9)
    lines.append(f"ws={w}: avg={avg:.4f}  solved>=90%={sol}/{N}")
out = f"N={N}  ({time.time()-t0:.0f}s)\n" + "\n".join(lines)
print("\n" + out, flush=True)
with open("result_nospace_wordseg.txt", "w", encoding="utf-8") as f:
    f.write(out + "\n")
