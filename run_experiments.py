"""Run the full experiment suite on the cross-validation split.

Loads the TRAIN-built solver once and reuses it for every config:
  1. Leakage test : same model vs train-drawn vs test-drawn benchmarks.
  2. Ablations    : disable each stage/term, evaluate on the test benchmark.
  3. No-space test: strip 띄어쓰기 from the test benchmark.

Writes a summary table to result_cv_experiments.txt.
"""
import io
import json
import sys
import time

if not isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from crack import BeamSolver

KN    = "kn_model_cv.json"
CORP  = "corpus_cv_train.txt"
PMAP  = "full_pattern_map_cv.pkl"
BENCH_TRAIN = "benchmark_cv_train.json"
BENCH_TEST  = "benchmark_cv_test.json"

N_MAIN = 100   # cases for leakage + no-space
N_ABL  = 100   # cases for ablations


def accuracy(plain, dec):
    n = min(len(plain), len(dec))
    return sum(a == b for a, b in zip(plain[:n], dec[:n])) / len(plain)


def load_cases(path, n, nospace=False):
    with open(path, "r", encoding="utf-8") as f:
        cases = json.load(f)[:n]
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
    line = (f"{label:<28} avg={avg:6.2%}  solved>=90%={solved:3d}/{len(accs)}"
            f"  ({time.time()-t0:.0f}s)")
    print(line, flush=True)
    return line


def main():
    print("Loading TRAIN-built solver ...", flush=True)
    solver = BeamSolver(kn_model_path=KN, corpus_path=CORP,
                        pattern_map_cache=PMAP)

    results = []

    # ---- 1. Leakage test ---------------------------------------------------
    results.append("=== 1. LEAKAGE (same train-built model, two sources) ===")
    print(results[-1], flush=True)
    solver.disable = set()
    results.append(run(solver, load_cases(BENCH_TRAIN, N_MAIN),
                       "in-sample (train-drawn)"))
    results.append(run(solver, load_cases(BENCH_TEST, N_MAIN),
                       "held-out (test-drawn)"))

    # ---- 2. Ablations (on held-out test benchmark) -------------------------
    results.append("")
    results.append("=== 2. ABLATIONS (held-out test, disable one part) ===")
    print(results[-1], flush=True)
    test_cases = load_cases(BENCH_TEST, N_ABL)
    ablations = [
        ("full pipeline",      set()),
        ("-kn (no LM score)",  {"kn"}),
        ("-struct (no DFA)",   {"struct"}),
        ("-dict (no word bonus)", {"dict"}),
        ("-beam (SA only)",    {"beam"}),
        ("-edge augment",      {"edge"}),
        ("-sa_fallback",       {"sa_fallback"}),
        ("-sa_polish",         {"sa_polish"}),
        ("-greedy polish",     {"greedy"}),
    ]
    for label, dis in ablations:
        solver.disable = dis
        results.append(run(solver, test_cases, label))

    # ---- 3. No-space test --------------------------------------------------
    results.append("")
    results.append("=== 3. NO-SPACE (held-out test, 띄어쓰기 removed) ===")
    print(results[-1], flush=True)
    solver.disable = set()
    results.append(run(solver, load_cases(BENCH_TEST, N_MAIN, nospace=True),
                       "no-space, full pipeline"))

    with open("result_cv_experiments.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(results) + "\n")
    print("\nWrote result_cv_experiments.txt", flush=True)


if __name__ == "__main__":
    main()
