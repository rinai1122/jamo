"""Hybrid solver: NN v6 predicts the key, HC v2 polishes it.

The NN gives a good starting mapping but can't exploit local KN statistics.
The HC v2 takes the NN key as the initial mapping for all restarts (with
light perturbation) so it starts near the correct solution rather than random.

Usage:
    python hybrid_solver.py [--cases N]
"""
import sys
import io
import json
import time
import random
import torch

if not isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from benchmark import accuracy
from nn_v6 import DualRankTransformerCracker, predict_key
from improved_solver_v2 import ImprovedSolverV2

NN_PTH = "jamo_nn_v6.pth"
BENCH_PATH = "benchmark_set.json"


class HybridSolver:
    def __init__(self, nn_pth=NN_PTH, kn_model="kn_model.json", corpus="corpus.txt"):
        self.nn_model = DualRankTransformerCracker(
            d_model=256, nhead=8, num_layers=6, max_len=210, norm_first=True
        )
        self.nn_model.load_state_dict(torch.load(nn_pth, map_location="cpu"))
        self.nn_model.eval()
        self.hc = ImprovedSolverV2(kn_model, corpus)

    def solve(self, ciphertext, restarts=20, iterations=3000):
        # Step 1: NN predicts the key
        try:
            nn_key = predict_key(self.nn_model, ciphertext)
        except Exception:
            nn_key = {}

        # Step 2: Inject NN key as a seed elite into HC v2
        symbols = sorted(set(ciphertext.replace(" ", "")))
        # Fill any missing symbols with unused targets
        used = set(nn_key.values())
        for s in symbols:
            if s not in nn_key:
                unused = [t for t in self.hc.target_jamos if t not in used]
                if unused:
                    nn_key[s] = unused[0]
                    used.add(unused[0])

        # Seed the HC elite pool with the NN key
        import collections
        import math

        cipher_counts = collections.Counter(ciphertext.replace(" ", ""))
        cipher_words = ciphertext.split(" ")
        from lean_solver import get_word_pattern
        word_candidates = [
            set(self.hc.dict_anchor.pattern_map.get(get_word_pattern(cw), []))
            if len(cw) > 2 else None
            for cw in cipher_words
        ]
        self.hc._ct = ciphertext
        self.hc._wc = word_candidates

        def decrypt(m):
            return "".join(m.get(c, c) for c in ciphertext)

        def fit(m):
            return self.hc.fitness(decrypt(m), word_candidates)

        nn_fitness = fit(nn_key)
        elite = [(nn_fitness, nn_key.copy())]

        rng = random.Random()
        best_mapping, best_fitness = nn_key.copy(), nn_fitness

        for r in range(restarts):
            mapping = self.hc._initial_mapping(
                symbols, cipher_counts, cipher_words, rng, elite=elite
            )
            current = fit(mapping)
            T = 2.0
            steps_since_improve = 0

            for _ in range(iterations):
                if len(symbols) < 2:
                    break
                new_mapping = mapping.copy()
                if rng.random() < 0.5:
                    s1, s2 = rng.sample(symbols, 2)
                    new_mapping[s1], new_mapping[s2] = new_mapping[s2], new_mapping[s1]
                else:
                    used = set(mapping.values())
                    unused = [t for t in self.hc.target_jamos if t not in used]
                    if unused:
                        s = rng.choice(symbols)
                        new_mapping[s] = rng.choice(unused)
                    else:
                        s1, s2 = rng.sample(symbols, 2)
                        new_mapping[s1], new_mapping[s2] = (
                            new_mapping[s2], new_mapping[s1]
                        )
                new_fit = fit(new_mapping)
                delta = new_fit - current
                if delta > 0 or (T > 1e-6 and rng.random() < math.exp(delta / T)):
                    if new_fit > current:
                        steps_since_improve = 0
                    mapping, current = new_mapping, new_fit
                else:
                    steps_since_improve += 1
                if steps_since_improve >= 300:
                    T = 2.0
                    steps_since_improve = 0
                else:
                    T *= 0.999

            if current > best_fitness:
                best_fitness, best_mapping = current, mapping.copy()
            elite.append((current, mapping.copy()))
            elite.sort(key=lambda x: -x[0])
            elite = elite[:5]

        best_mapping, _ = self.hc._greedy_polish(best_mapping, symbols, best_fitness)
        return decrypt(best_mapping)


def main():
    import os
    if not os.path.exists(NN_PTH):
        print(f"NN model {NN_PTH} not found. Run: python train_v6.py")
        sys.exit(1)

    args = sys.argv[1:]
    n_cases = None
    if "--cases" in args:
        n_cases = int(args[args.index("--cases") + 1])

    with open(BENCH_PATH, "r", encoding="utf-8") as f:
        cases = json.load(f)
    if n_cases:
        cases = cases[:n_cases]

    solver = HybridSolver()
    accs = []
    t0 = time.time()
    for i, case in enumerate(cases):
        t1 = time.time()
        dec = solver.solve(case["ciphertext"])
        acc = accuracy(case["plaintext"], dec)
        accs.append(acc)
        print(f"[{i+1}/{len(cases)}] acc={acc:.2%} ({time.time()-t1:.1f}s)")
        print(f"  orig: {case['plaintext']}")
        print(f"  dec:  {dec}")

    avg = sum(accs) / len(accs)
    solved = sum(1 for a in accs if a >= 0.9)
    print(f"\n=== Hybrid (NN-v6 + HC-v2) ===")
    print(f"Average accuracy: {avg:.2%}")
    print(f"Cases >=90%: {solved}/{len(accs)}")
    print(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
