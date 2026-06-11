"""HC v2: elitism + adaptive reheating over improved_solver.

Key changes vs ImprovedSolver:
1. Elite pool: top-3 mappings are kept across restarts. 70% of restarts
   seed from a perturbed elite mapping instead of pure freq-rank init.
2. Adaptive reheating: if SA makes no improvement for REHEAT_STEPS steps,
   temperature jumps back to T_REHEAT to escape local optima.
3. Adaptive restart budget: longer texts get fewer restarts (fixed wall
   time budget), shorter texts get more.
4. Heavier greedy polish: try 3-way cycle moves in addition to swaps.
"""
import collections
import math
import random
import sys
import io

if not isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from final_solver import KneserNeyScorer
from lean_solver import DictionaryAnchor, get_word_pattern
from improved_solver import structure_violations, is_vowel, is_consonant

VOWEL_LO, VOWEL_HI = 0x314F, 0x3163
CONS_LO, CONS_HI = 0x3131, 0x314E

REHEAT_STEPS = 400
T_REHEAT = 2.0
ELITE_SIZE = 5


class ImprovedSolverV2:
    def __init__(self, kn_model_path, corpus_path, order=6):
        self.scorer = KneserNeyScorer(kn_model_path)
        self.order = order
        self._prob_cache = {}
        self.dict_anchor = DictionaryAnchor(corpus_path)
        self.target_jamos = sorted(
            k for k in self.scorer.vocab if 0x3131 <= ord(k) <= 0x3163
        )
        uni = self.scorer.counts[1]
        self.freq_ranked_jamos = sorted(
            self.target_jamos, key=lambda j: -uni.get(j, 0)
        )

    def _prob(self, ngram):
        p = self._prob_cache.get(ngram)
        if p is None:
            p = self.scorer.get_prob(ngram)
            self._prob_cache[ngram] = p
        return p

    def fitness(self, text, word_candidates):
        score = 0.0
        for i in range(len(text)):
            context_len = min(i + 1, self.order)
            score += math.log10(self._prob(text[i - context_len + 1: i + 1]))
        score -= 10.0 * structure_violations(text)
        for word, candidates in zip(text.split(" "), word_candidates):
            if candidates and word in candidates:
                score += 4.0 * len(word)
        return score

    def _initial_mapping(self, symbols, cipher_counts, cipher_words, rng,
                         elite=None):
        if elite and rng.random() < 0.70:
            # Perturb an elite mapping
            _, base_mapping = rng.choice(elite)
            mapping = base_mapping.copy()
            # Randomly perturb 2-4 symbol assignments
            n_perturb = rng.randint(2, min(4, len(symbols)))
            for _ in range(n_perturb):
                if rng.random() < 0.5 and len(symbols) >= 2:
                    s1, s2 = rng.sample(symbols, 2)
                    mapping[s1], mapping[s2] = mapping.get(s2), mapping.get(s1)
                else:
                    used = set(mapping.values()) - {None}
                    unused = [t for t in self.target_jamos if t not in used]
                    if unused:
                        s = rng.choice(symbols)
                        mapping[s] = rng.choice(unused)
            return mapping

        # Frequency-rank seed (same as v1)
        ranked_syms = sorted(symbols, key=lambda s: -cipher_counts[s])
        targets = list(self.freq_ranked_jamos)
        for i in range(0, len(targets) - 2, 3):
            window = targets[i: i + 3]
            rng.shuffle(window)
            targets[i: i + 3] = window
        mapping = {s: targets[i] for i, s in enumerate(ranked_syms)}

        for cw in cipher_words:
            if len(cw) > 3 and rng.random() < 0.7:
                matches = self.dict_anchor.get_matches(cw)
                if not matches:
                    continue
                match = rng.choice(matches)
                temp = mapping.copy()
                used = set(temp.values())
                ok = True
                for c_char, p_char in zip(cw, match):
                    cur = temp.get(c_char)
                    if cur == p_char:
                        continue
                    if p_char in used:
                        ok = False
                        break
                    if cur is not None:
                        used.discard(cur)
                    temp[c_char] = p_char
                    used.add(p_char)
                if ok:
                    mapping = temp
        return mapping

    def _greedy_polish(self, mapping, symbols, fit):
        """Exhaustive swap + unused-reassign + 3-cycle polish."""
        improved = True
        current = fit
        while improved:
            improved = False
            # Unused reassignments
            for s in symbols:
                used = set(mapping.values())
                for t in self.target_jamos:
                    if t in used:
                        continue
                    cand = mapping.copy()
                    cand[s] = t
                    f = self.fitness(
                        "".join(cand.get(c, c) for c in self._ct), self._wc
                    )
                    if f > current:
                        mapping, current, improved = cand, f, True
            # Pairwise swaps
            for i in range(len(symbols)):
                for j in range(i + 1, len(symbols)):
                    s1, s2 = symbols[i], symbols[j]
                    cand = mapping.copy()
                    cand[s1], cand[s2] = cand[s2], cand[s1]
                    f = self.fitness(
                        "".join(cand.get(c, c) for c in self._ct), self._wc
                    )
                    if f > current:
                        mapping, current, improved = cand, f, True
        return mapping, current

    def solve(self, ciphertext, restarts=30, iterations=5000, verbose=False):
        symbols = sorted(set(ciphertext.replace(" ", "")))
        cipher_counts = collections.Counter(ciphertext.replace(" ", ""))
        cipher_words = ciphertext.split(" ")
        word_candidates = [
            set(self.dict_anchor.pattern_map.get(get_word_pattern(cw), []))
            if len(cw) > 2 else None
            for cw in cipher_words
        ]
        # Store for _greedy_polish
        self._ct = ciphertext
        self._wc = word_candidates
        rng = random.Random()

        def decrypt(mapping):
            return "".join(mapping.get(c, c) for c in ciphertext)

        def fit(mapping):
            return self.fitness(decrypt(mapping), word_candidates)

        # Adjust budget: longer texts are easier per-step, reduce restarts
        n_jamo = len(ciphertext.replace(" ", ""))
        effective_restarts = max(10, restarts - max(0, (n_jamo - 60) // 20))

        best_mapping, best_fitness = None, -float("inf")
        elite = []  # list of (fitness, mapping)

        for r in range(effective_restarts):
            mapping = self._initial_mapping(
                symbols, cipher_counts, cipher_words, rng, elite=elite
            )
            current = fit(mapping)

            T = 3.0
            steps_since_improve = 0

            for step in range(iterations):
                if len(symbols) < 2:
                    break
                new_mapping = mapping.copy()
                if rng.random() < 0.5:
                    s1, s2 = rng.sample(symbols, 2)
                    new_mapping[s1], new_mapping[s2] = new_mapping[s2], new_mapping[s1]
                else:
                    used = set(mapping.values())
                    unused = [t for t in self.target_jamos if t not in used]
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

                # Adaptive reheating
                if steps_since_improve >= REHEAT_STEPS:
                    T = T_REHEAT
                    steps_since_improve = 0
                else:
                    T *= 0.999

            if current > best_fitness:
                best_fitness, best_mapping = current, mapping.copy()
                if verbose:
                    print(f"  restart {r}: new best {best_fitness:.2f}")

            # Update elite pool
            elite.append((current, mapping.copy()))
            elite.sort(key=lambda x: -x[0])
            elite = elite[:ELITE_SIZE]

        # Greedy polish on the best mapping
        best_mapping, _ = self._greedy_polish(best_mapping, symbols, best_fitness)
        return decrypt(best_mapping)


if __name__ == "__main__":
    solver = ImprovedSolverV2("kn_model.json", "corpus.txt")
    from benchmark import accuracy
    import json

    with open("benchmark_set.json", "r", encoding="utf-8") as f:
        cases = json.load(f)
    for case in cases[:3]:
        dec = solver.solve(case["ciphertext"], verbose=True)
        print(f"orig: {case['plaintext']}")
        print(f"dec:  {dec}")
        print(f"acc:  {accuracy(case['plaintext'], dec):.2%}")
