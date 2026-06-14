"""Continuation of the experiment suite (after the SA-only config proved too
slow at N=100).  Lighter ablations run at N=100; the SA-heavy configs
(-beam = pure SA, and no-space) run at N=30 with matched full-pipeline
references so the comparison is fair.
"""
import io
import json
import sys
import time

if not isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from crack import BeamSolver

KN, CORP, PMAP = "kn_model_cv.json", "corpus_cv_train.txt", "full_pattern_map_cv.pkl"
BENCH_TEST = "benchmark_cv_test.json"


def accuracy(plain, dec):
    n = min(len(plain), len(dec))
    return sum(a == b for a, b in zip(plain[:n], dec[:n])) / len(plain)


def load_cases(path, n, nospace=False):
    cases = json.load(open(path, encoding="utf-8"))[:n]
    out = []
    for c in cases:
        p, t = c["plaintext"], c["ciphertext"]
        if nospace:
            p, t = p.replace(" ", ""), t.replace(" ", "")
        out.append((p, t))
    return out


def run(solver, cases, label):
    t0 = time.time()
    accs = [accuracy(p, solver.solve(t)) for p, t in cases]
    avg = sum(accs) / len(accs)
    solved = sum(1 for a in accs if a >= 0.9)
    line = (f"{label:<32} avg={avg:6.2%}  solved>=90%={solved:3d}/{len(accs)}"
            f"  ({time.time()-t0:.0f}s)")
    print(line, flush=True)
    return line


def main():
    print("Loading TRAIN-built solver ...", flush=True)
    solver = BeamSolver(kn_model_path=KN, corpus_path=CORP, pattern_map_cache=PMAP)
    res = []

    res.append("=== ABLATIONS cont. (held-out test, N=100) ===")
    print(res[-1], flush=True)
    cases100 = load_cases(BENCH_TEST, 100)
    for label, dis in [("-edge augment", {"edge"}),
                       ("-sa_fallback", {"sa_fallback"}),
                       ("-sa_polish", {"sa_polish"}),
                       ("-greedy polish", {"greedy"})]:
        solver.disable = dis
        res.append(run(solver, cases100, label))

    res.append("")
    res.append("=== SA-heavy configs (held-out test, N=30) ===")
    print(res[-1], flush=True)
    cases30 = load_cases(BENCH_TEST, 30)
    solver.disable = set()
    res.append(run(solver, cases30, "full pipeline (N=30 ref)"))
    solver.disable = {"beam"}
    res.append(run(solver, cases30, "-beam (pure SA, N=30)"))

    res.append("")
    res.append("=== NO-SPACE (held-out test, 띄어쓰기 removed, N=30) ===")
    print(res[-1], flush=True)
    nospace30 = load_cases(BENCH_TEST, 30, nospace=True)
    solver.disable = set()
    res.append(run(solver, nospace30, "no-space, full pipeline"))

    with open("result_cv_experiments2.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(res) + "\n")
    print("\nWrote result_cv_experiments2.txt", flush=True)


if __name__ == "__main__":
    main()
