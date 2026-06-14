"""Diagnose no-space short-case failures.

For each test case, recover the *true* key (from plaintext vs ciphertext),
then compare the true key's fitness to the solver's output fitness.  If the
true key is NOT the fitness optimum, the failure is in the objective, not the
search -- and a dictionary-coverage term should help separate them.
"""
import io
import json
import pickle
import sys

if not isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from crack import KneserNeyScorer, structure_violations
from nospace_solver import NoSpaceSolver

STRUCT_PENALTY = 10.0


def true_key(cipher, plain):
    """cipher char -> plain char, from aligned spaceless strings."""
    m = {}
    for c, p in zip(cipher, plain):
        m[c] = p
    return m


def build_word_counts(pmap_path):
    with open(pmap_path, "rb") as f:
        pmap, _ = pickle.load(f)
    wc = {}
    for cands in pmap.values():
        for w, c in cands:
            wc[w] = c
    return wc


def seg_coverage(text, wc, min_count=5, max_len=12):
    """Best fraction of chars coverable by dict words (DP)."""
    n = len(text)
    b = [0] * (n + 1)
    for i in range(1, n + 1):
        b[i] = b[i - 1]
        for L in range(2, min(max_len, i) + 1):
            w = text[i - L:i]
            if wc.get(w, 0) >= min_count and b[i - L] + L > b[i]:
                b[i] = b[i - L] + L
    return b[n] / n if n else 0.0


def main():
    scorer = KneserNeyScorer("kn_model_cv.json")
    wc = build_word_counts("full_pattern_map_cv.pkl")
    solver = NoSpaceSolver(kn_model_path="kn_model_cv.json")

    def fitness(t):
        return scorer.log_prob(t) - STRUCT_PENALTY * structure_violations(t)

    cases = json.load(open("benchmark_cv_test.json", encoding="utf-8"))[:15]
    print("case len  trueFit   outFit  dFit  trueCov  outCov  acc")
    for i, c in enumerate(cases):
        plain = c["plaintext"].replace(" ", "")
        cipher = c["ciphertext"].replace(" ", "")
        tk = true_key(cipher, plain)
        true_dec = "".join(tk.get(ch, ch) for ch in cipher)
        out = solver.solve(cipher)
        n = min(len(plain), len(out))
        acc = sum(a == b for a, b in zip(plain[:n], out[:n])) / len(plain)
        f_true, f_out = fitness(true_dec), fitness(out)
        cov_true = seg_coverage(true_dec, wc)
        cov_out = seg_coverage(out, wc)
        flag = "  <-- true loses!" if f_out > f_true + 1e-6 else ""
        print(f"{i+1:2d} {len(plain):4d} {f_true:8.1f} {f_out:8.1f} "
              f"{f_out-f_true:6.1f} {cov_true:6.2%} {cov_out:6.2%} {acc:6.1%}{flag}",
              flush=True)


if __name__ == "__main__":
    main()
