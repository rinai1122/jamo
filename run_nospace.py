"""Focused no-space (띄어쓰기 removed) test with per-case progress output.

Strips spaces from both ciphertext and plaintext, then solves.  Prints each
case as it completes so progress is visible.  Also reports the matched
full-pipeline WITH spaces on the same cases for direct comparison.
"""
import io
import json
import sys
import time

if not isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from crack import BeamSolver

N = 15

def accuracy(plain, dec):
    n = min(len(plain), len(dec))
    return sum(a == b for a, b in zip(plain[:n], dec[:n])) / len(plain)

solver = BeamSolver(kn_model_path="kn_model_cv.json",
                    corpus_path="corpus_cv_train.txt",
                    pattern_map_cache="full_pattern_map_cv.pkl")

cases = json.load(open("benchmark_cv_test.json", encoding="utf-8"))[:N]

ns_accs, sp_accs = [], []
for i, c in enumerate(cases):
    plain, cipher = c["plaintext"], c["ciphertext"]
    # with spaces (reference)
    t0 = time.time(); dec_sp = solver.solve(cipher); t_sp = time.time()-t0
    a_sp = accuracy(plain, dec_sp); sp_accs.append(a_sp)
    # no spaces
    p_ns, c_ns = plain.replace(" ", ""), cipher.replace(" ", "")
    t0 = time.time(); dec_ns = solver.solve(c_ns); t_ns = time.time()-t0
    a_ns = accuracy(p_ns, dec_ns); ns_accs.append(a_ns)
    print(f"[{i+1:2d}/{N}] len={len(p_ns):3d}  "
          f"spaced={a_sp:6.2%}({t_sp:.0f}s)  nospace={a_ns:6.2%}({t_ns:.0f}s)",
          flush=True)

print(f"\nWITH spaces : avg={sum(sp_accs)/len(sp_accs):.2%}  "
      f"solved={sum(1 for a in sp_accs if a>=0.9)}/{N}")
print(f"NO spaces   : avg={sum(ns_accs)/len(ns_accs):.2%}  "
      f"solved={sum(1 for a in ns_accs if a>=0.9)}/{N}")

with open("result_nospace.txt", "w", encoding="utf-8") as f:
    f.write(f"WITH spaces : avg={sum(sp_accs)/len(sp_accs):.4f} solved={sum(1 for a in sp_accs if a>=0.9)}/{N}\n")
    f.write(f"NO spaces   : avg={sum(ns_accs)/len(ns_accs):.4f} solved={sum(1 for a in ns_accs if a>=0.9)}/{N}\n")
