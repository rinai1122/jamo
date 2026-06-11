"""Improved simulated-annealing solver for Korean jamo substitution ciphers.

Fixes over lean_solver.AdvancedSolver (same search budget):
1. Move set includes re-assigning a symbol to an UNUSED target jamo.
   The baseline could only swap targets chosen at init, so any key jamo
   missed by the initial random sample was unreachable for that restart.
2. KN probabilities are memoized, which makes order-6 context affordable
   (baseline capped context at 4 even though the model is 7-gram).
3. Restarts seed from cipher-frequency -> corpus-frequency rank mapping
   (perturbed per restart) instead of pure random.
4. Dictionary bonus scales with word length instead of a flat +1000 cliff,
   and candidate words per cipher word are precomputed once per solve.
5. Syllable-structure DFA penalty: jamo words are strictly (C V C?)+,
   which prunes most of the search space.
6. Greedy polish pass (all swaps + unused reassignments) after annealing.
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

VOWEL_LO, VOWEL_HI = 0x314F, 0x3163
CONS_LO, CONS_HI = 0x3131, 0x314E


def is_vowel(c):
    return VOWEL_LO <= ord(c) <= VOWEL_HI


def is_consonant(c):
    return CONS_LO <= ord(c) <= CONS_HI


def structure_violations(text):
    """Count violations of the (C V C?)+ jamo word structure."""
    violations = 0
    state = 0  # 0=expect C (word start), 1=expect V, 2=after V, 3=after VC
    for ch in text:
        if ch == " ":
            state = 0
            continue
        v = is_vowel(ch)
        if state == 0:
            if v:
                violations += 1
                state = 2
            else:
                state = 1
        elif state == 1:
            if v:
                state = 2
            else:
                violations += 1
                state = 1
        elif state == 2:
            if v:
                violations += 1
            else:
                state = 3
        else:  # state 3: after VC -> V means it was a new syllable, C means jong+cho
            if v:
                state = 2
            else:
                state = 1
    return violations


class ImprovedSolver:
    def __init__(self, kn_model_path, corpus_path, order=6):
        self.scorer = KneserNeyScorer(kn_model_path)
        self.order = order
        self._prob_cache = {}
        self.dict_anchor = DictionaryAnchor(corpus_path)
        self.target_jamos = sorted(
            k for k in self.scorer.vocab if 0x3131 <= ord(k) <= 0x3163
        )
        # Corpus jamo ranked by unigram frequency for seeding restarts
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
            score += math.log10(self._prob(text[i - context_len + 1 : i + 1]))
        score -= 10.0 * structure_violations(text)
        for word, candidates in zip(text.split(" "), word_candidates):
            if candidates and word in candidates:
                score += 4.0 * len(word)
        return score

    def _initial_mapping(self, symbols, cipher_counts, cipher_words, rng):
        # Frequency-rank seed, perturbed per restart
        ranked_syms = sorted(symbols, key=lambda s: -cipher_counts[s])
        targets = list(self.freq_ranked_jamos)
        # Local shuffle keeps ranks roughly aligned but varies restarts
        for i in range(0, len(targets) - 2, 3):
            window = targets[i : i + 3]
            rng.shuffle(window)
            targets[i : i + 3] = window
        mapping = {s: targets[i] for i, s in enumerate(ranked_syms)}

        # Dictionary seeding (consistent, injective overrides)
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

    def solve(self, ciphertext, restarts=30, iterations=5000, verbose=False):
        symbols = sorted(set(ciphertext.replace(" ", "")))
        cipher_counts = collections.Counter(ciphertext.replace(" ", ""))
        cipher_words = ciphertext.split(" ")
        # Candidate plaintext words per cipher word, computed once:
        # bijective substitution preserves the repetition pattern.
        word_candidates = [
            set(self.dict_anchor.pattern_map.get(get_word_pattern(cw), []))
            if len(cw) > 2
            else None
            for cw in cipher_words
        ]
        rng = random.Random()

        def decrypt(mapping):
            return "".join(mapping.get(c, c) for c in ciphertext)

        def fit(mapping):
            return self.fitness(decrypt(mapping), word_candidates)

        best_mapping, best_fitness = None, -float("inf")

        for r in range(restarts):
            mapping = self._initial_mapping(symbols, cipher_counts, cipher_words, rng)
            current = fit(mapping)

            T = 3.0
            for _ in range(iterations):
                if len(symbols) < 2:
                    break
                new_mapping = mapping.copy()
                if rng.random() < 0.5:
                    s1, s2 = rng.sample(symbols, 2)
                    new_mapping[s1], new_mapping[s2] = new_mapping[s2], new_mapping[s1]
                else:
                    # Reassign a symbol to an unused target (baseline lacked this)
                    used = set(mapping.values())
                    unused = [t for t in self.target_jamos if t not in used]
                    if unused:
                        s = rng.choice(symbols)
                        new_mapping[s] = rng.choice(unused)
                    else:
                        s1, s2 = rng.sample(symbols, 2)
                        new_mapping[s1], new_mapping[s2] = (
                            new_mapping[s2],
                            new_mapping[s1],
                        )
                new_fit = fit(new_mapping)
                delta = new_fit - current
                if delta > 0 or (T > 1e-6 and rng.random() < math.exp(delta / T)):
                    mapping, current = new_mapping, new_fit
                T *= 0.999

            if current > best_fitness:
                best_fitness, best_mapping = current, mapping.copy()
                if verbose:
                    print(f"  restart {r}: new best {best_fitness:.2f}")

        # Greedy polish on the best mapping
        improved = True
        while improved:
            improved = False
            current = fit(best_mapping)
            for s in symbols:
                used = set(best_mapping.values())
                for t in self.target_jamos:
                    if t in used:
                        continue
                    cand = best_mapping.copy()
                    cand[s] = t
                    f = fit(cand)
                    if f > current:
                        best_mapping, current, improved = cand, f, True
            for i in range(len(symbols)):
                for j in range(i + 1, len(symbols)):
                    s1, s2 = symbols[i], symbols[j]
                    cand = best_mapping.copy()
                    cand[s1], cand[s2] = cand[s2], cand[s1]
                    f = fit(cand)
                    if f > current:
                        best_mapping, current, improved = cand, f, True

        return decrypt(best_mapping)


if __name__ == "__main__":
    solver = ImprovedSolver("kn_model.json", "corpus.txt")
    from benchmark import accuracy
    import json

    with open("benchmark_set.json", "r", encoding="utf-8") as f:
        cases = json.load(f)
    for case in cases[:2]:
        dec = solver.solve(case["ciphertext"], verbose=True)
        print(f"orig: {case['plaintext']}")
        print(f"dec:  {dec}")
        print(f"acc:  {accuracy(case['plaintext'], dec):.2%}")
