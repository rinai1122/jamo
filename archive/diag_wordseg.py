"""Does a frequency-weighted word-segmentation score separate true from the
high-fitness wrong readings the search converges to?

For each case prints the jamo-LM gap (true - out) and the wordseg gap.
If wordseg gap is positive where jamo gap is negative, adding wordseg to the
objective would rerank the true key above the wrong one.
"""
import io
import json
import math
import pickle
import sys

if not isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from crack import KneserNeyScorer, structure_violations
from nospace_solver import NoSpaceSolver

STRUCT_PENALTY = 10.0


def build_word_logp(pmap_path, k=0.5):
    with open(pmap_path, "rb") as f:
        pmap, _ = pickle.load(f)
    wc = {}
    for cands in pmap.values():
        for w, c in cands:
            wc[w] = c
    T = sum(wc.values())
    V = len(wc)
    logp = {w: math.log10((c + k) / (T + k * V)) for w, c in wc.items()}
    floor = math.log10(k / (T + k * V))      # unseen word / unknown char
    maxlen = max(len(w) for w in wc)
    return logp, floor, min(maxlen, 16)


def wordseg_logp(text, logp, floor, maxlen):
    """Best per-char-normalised word-unigram segmentation log-prob."""
    n = len(text)
    if not n:
        return 0.0
    best = [(-math.inf)] * (n + 1)
    best[0] = 0.0
    for i in range(1, n + 1):
        best[i] = best[i - 1] + floor              # char i-1 as unknown
        for L in range(2, min(maxlen, i) + 1):
            lp = logp.get(text[i - L:i])
            if lp is not None and best[i - L] + lp > best[i]:
                best[i] = best[i - L] + lp
    return best[n] / n


def true_key(cipher, plain):
    m = {}
    for c, p in zip(cipher, plain):
        m[c] = p
    return m


def main():
    scorer = KneserNeyScorer("kn_model_cv.json")
    logp, floor, maxlen = build_word_logp("full_pattern_map_cv.pkl")
    solver = NoSpaceSolver(kn_model_path="kn_model_cv.json")

    def jamo_fit(t):
        return scorer.log_prob(t) - STRUCT_PENALTY * structure_violations(t)

    cases = json.load(open("benchmark_cv_test.json", encoding="utf-8"))[:15]
    print("case len   acc   jamoGap  wsTrue   wsOut   wsGap  verdict")
    for i, c in enumerate(cases):
        plain = c["plaintext"].replace(" ", "")
        cipher = c["ciphertext"].replace(" ", "")
        tk = true_key(cipher, plain)
        true_dec = "".join(tk.get(ch, ch) for ch in cipher)
        out = solver.solve(cipher)
        n = min(len(plain), len(out))
        acc = sum(a == b for a, b in zip(plain[:n], out[:n])) / len(plain)
        jgap = jamo_fit(true_dec) - jamo_fit(out)
        ws_t = wordseg_logp(true_dec, logp, floor, maxlen)
        ws_o = wordseg_logp(out, logp, floor, maxlen)
        wsgap = ws_t - ws_o
        # would wordseg flip a case where jamo prefers the wrong key?
        verdict = ""
        if jgap < 0 and wsgap > 0:
            verdict = "WORDSEG FIXES"
        elif jgap < 0 and wsgap <= 0:
            verdict = "still wrong"
        print(f"{i+1:2d} {len(plain):4d} {acc:6.1%} {jgap:8.2f} "
              f"{ws_t:7.3f} {ws_o:7.3f} {wsgap:7.3f}  {verdict}", flush=True)


if __name__ == "__main__":
    main()
