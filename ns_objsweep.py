"""Offline objective tuning against saved solver outputs (no re-solving).

Loads an ns_eval result file (which stores each case's solver `out` string),
reconstructs the true-key decryption from the benchmark, and re-scores BOTH
under a sweep of `word_penalty` (and optionally `ws_weight`) values.

Why this is valid and cheap: the expensive part is the *search*; re-scoring two
fixed strings is microseconds.  For an OBJECTIVE failure (a wrong key the search
found out-scores the truth) the necessary condition for a better objective to
help is that it flips the ranking -- f(true) > f(out).  At the same time the
change must not break the already-correct cases (where out == true, so the gap
stays ~0).  This script reports both, so a `word_penalty` can be chosen on
evidence before paying for a full solve run.
"""
import io
import json
import sys

if not isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from nospace_solver import NoSpaceSolver

RESULT = sys.argv[1] if len(sys.argv) > 1 else "ns_result_val.json"
WS_WEIGHT = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
PENALTIES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]


def true_key_decrypt(cipher, plain):
    m = {}
    for c, p in zip(cipher, plain):
        m[c] = p
    return "".join(m.get(ch, ch) for ch in cipher)


def main():
    res = json.load(open(RESULT, encoding="utf-8"))
    bench = json.load(open("benchmark_cv_test.json", encoding="utf-8"))

    # Build (case -> out, true, acc, len) from the saved run.
    cases = []
    for c in res["cases"]:
        i = c["i"]
        plain = bench[i]["plaintext"].replace(" ", "")
        cipher = bench[i]["ciphertext"].replace(" ", "")
        cases.append({"i": i, "len": c["len"], "acc": c["acc"],
                      "out": c["out"], "true": true_key_decrypt(cipher, plain)})

    # One solver per penalty (cheap; just changes the scalar).  ws_weight fixed.
    solvers = {wp: NoSpaceSolver(kn_model_path="kn_model_cv.json",
                                 ws_weight=WS_WEIGHT, word_penalty=wp, use_sa=False)
               for wp in PENALTIES}

    print(f"result={RESULT}  ws_weight={WS_WEIGHT}\n")
    print("Per-penalty objective health:")
    print(f"  {'wp':>4s}  {'obj-fails':>9s}  {'flipped(true>out)':>17s}  "
          f"{'broken-solved':>13s}")
    for wp in PENALTIES:
        s = solvers[wp]
        objfail = flipped = broken = 0
        for c in cases:
            f_out = s._fitness(c["out"])
            f_true = s._fitness(c["true"])
            is_solved = c["acc"] >= 0.9
            if not is_solved:
                # was an objective failure under the *original* objective?
                if c["out"] != c["true"]:
                    objfail += 1
                    if f_true > f_out + 1e-6:
                        flipped += 1
            else:
                # a currently-solved case must keep truth at/above its output
                if f_out > f_true + 1e-6:
                    broken += 1
        print(f"  {wp:4.1f}  {objfail:9d}  {flipped:17d}  {broken:13d}")

    # Detail: per failing case, the f(out)-f(true) gap across penalties.
    print("\nPer-failed-case gap  f(out)-f(true)  (negative = truth now wins):")
    hdr = "  case len  acc  " + "".join(f"{wp:>7.1f}" for wp in PENALTIES)
    print(hdr)
    for c in sorted(cases, key=lambda x: x["acc"]):
        if c["acc"] >= 0.9:
            continue
        gaps = []
        for wp in PENALTIES:
            s = solvers[wp]
            gaps.append(s._fitness(c["out"]) - s._fitness(c["true"]))
        print(f"  {c['i']:3d} {c['len']:3d} {c['acc']:4.0%} "
              + "".join(f"{g:7.1f}" for g in gaps))


if __name__ == "__main__":
    main()
