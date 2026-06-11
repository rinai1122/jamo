"""Beam solver: dictionary-constrained beam search over word patterns.

Why this works: a bijective substitution preserves the repetition pattern
of every word, and spaces are preserved, so each cipher word indexes a
small candidate set of corpus words (a 10-jamo word has ~40 candidates,
a 12-jamo word ~3). Words are processed most-constrained-first; a beam
of partial key mappings is extended with every consistent candidate (or
a skip, for truncated fragment-edge words). Completed mappings are
reranked with the order-6 KN fitness, refined with a short seeded
anneal, and finished with the exhaustive greedy polish.

Usage:
    python beam_solver.py [num_cases]
"""
import collections
import math
import os
import pickle
import random
import sys
import io

if not isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from jamo import h2j, j2hcj
from lean_solver import get_word_pattern
from improved_solver import structure_violations
from improved_solver_v2 import ImprovedSolverV2

PATTERN_CACHE = "full_pattern_map.pkl"

BEAM_WIDTH = 500
BRANCH_CAP = 200       # consistent candidates tried per state per word
CAND_CAP = 6000        # candidates kept per pattern (sorted by count)
SKIP_LP = -10.0        # beam logprob for skipping a word
EDGE_LP_PENALTY = 1.0  # truncation candidates are slightly less trusted
EDGE_CAP = 800         # prefix/suffix candidates per edge word
SA_RESTARTS = 8
SA_ITERS = 1500


def build_full_pattern_map(corpus_path):
    """pattern -> list of (jamo_word, count), sorted by count desc.

    Unlike DictionaryAnchor (top-10k words only), this indexes every
    distinct corpus word so rare-but-real words still anchor the key.
    """
    if os.path.exists(PATTERN_CACHE):
        with open(PATTERN_CACHE, "rb") as f:
            return pickle.load(f)
    word_counts = collections.Counter()
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            for w in line.split():
                try:
                    jw = j2hcj(h2j(w))
                except Exception:
                    continue
                jw = "".join(c for c in jw if 0x3131 <= ord(c) <= 0x3163)
                if len(jw) >= 2:
                    word_counts[jw] += 1
    pmap = collections.defaultdict(list)
    for w, c in word_counts.items():
        pmap[get_word_pattern(w)].append((w, c))
    for p in pmap:
        pmap[p].sort(key=lambda wc: -wc[1])
        pmap[p] = pmap[p][:CAND_CAP]
    result = (dict(pmap), sum(word_counts.values()))
    with open(PATTERN_CACHE, "wb") as f:
        pickle.dump(result, f)
    return result


class BeamSolver:
    def __init__(self, kn_model_path="kn_model.json", corpus_path="corpus.txt"):
        self.hc = ImprovedSolverV2(kn_model_path, corpus_path)
        self.pattern_map, self.total_words = build_full_pattern_map(corpus_path)
        self.log_total = math.log10(self.total_words)
        # Flat word list (count desc) for edge-word truncation matching
        self.all_words = sorted(
            (wc for cands in self.pattern_map.values() for wc in cands),
            key=lambda wc: -wc[1],
        )

    # ---------- beam search over word candidates ----------

    def _edge_candidates(self, cw, side):
        """Candidates for a fragment-edge word truncated mid-word.

        The first word of a fragment may be the *suffix* of a corpus word,
        the last word a *prefix*. Scans the flat word list (count desc) and
        keeps matching truncations, summing counts of identical truncations.
        """
        pat = get_word_pattern(cw)
        L = len(cw)
        found = {}
        for w, c in self.all_words:
            if len(w) <= L:
                continue
            piece = w[-L:] if side == "suffix" else w[:L]
            if get_word_pattern(piece) == pat:
                found[piece] = found.get(piece, 0) + c
                if len(found) >= EDGE_CAP:
                    break
        return sorted(found.items(), key=lambda wc: -wc[1])

    def _beam(self, cipher_words, edge_augment=False):
        # Unique cipher words (len>=2) with multiplicity, most constrained first
        word_mult = collections.Counter(
            cw for cw in cipher_words if len(cw) >= 2
        )
        edge_sides = collections.defaultdict(list)  # cw -> ["suffix"|"prefix"]
        if edge_augment:
            real = [cw for cw in cipher_words if len(cw) >= 2]
            if real and len(real[0]) >= 3:
                edge_sides[real[0]].append("suffix")
            if real and len(real[-1]) >= 3:
                edge_sides[real[-1]].append("prefix")
        jobs = []
        for cw, mult in word_mult.items():
            cands = self.pattern_map.get(get_word_pattern(cw), [])
            if cw in edge_sides:
                merged = dict(cands)
                penalty = 10 ** (-EDGE_LP_PENALTY)
                for side in edge_sides[cw]:
                    for piece, c in self._edge_candidates(cw, side):
                        c_pen = max(int(c * penalty), 1)
                        merged[piece] = max(merged.get(piece, 0), c_pen)
                cands = sorted(merged.items(), key=lambda wc: -wc[1])
            jobs.append((cw, mult, cands))
        jobs.sort(key=lambda j: len(j[2]))

        # states: (score, mapping dict, used target set)
        states = [(0.0, {}, frozenset())]
        for cw, mult, cands in jobs:
            # (pos, plain_char) -> candidate-id set, for fast consistency filter
            pos_index = collections.defaultdict(set)
            for ci, (w, _) in enumerate(cands):
                for pos, ch in enumerate(w):
                    pos_index[(pos, ch)].add(ci)
            all_ids = range(len(cands))

            next_states = {}

            def push(score, mapping, used):
                key = frozenset(mapping.items())
                cur = next_states.get(key)
                if cur is None or score > cur[0]:
                    next_states[key] = (score, mapping, used)

            for score, mapping, used in states:
                # candidate ids consistent with already-mapped symbols
                fixed = [(i, mapping[c]) for i, c in enumerate(cw) if c in mapping]
                if fixed:
                    sets = [pos_index.get(fx) for fx in fixed]
                    if any(s is None for s in sets):
                        ids = ()
                    else:
                        ids = sorted(set.intersection(*sets))  # asc id = desc count
                else:
                    ids = all_ids

                n_branched = 0
                for ci in ids:
                    if n_branched >= BRANCH_CAP:
                        break
                    w, cnt = cands[ci]
                    new_pairs = {}
                    ok = True
                    for c_char, p_char in zip(cw, w):
                        if c_char in mapping:
                            continue  # consistency guaranteed by the filter
                        prev = new_pairs.get(c_char)
                        if prev is not None:
                            if prev != p_char:
                                ok = False
                                break
                            continue
                        if p_char in used:
                            ok = False
                            break
                        new_pairs[c_char] = p_char
                    if not ok or len(set(new_pairs.values())) != len(new_pairs):
                        continue
                    n_branched += 1
                    nm = dict(mapping)
                    nm.update(new_pairs)
                    nu = used | frozenset(new_pairs.values())
                    lp = math.log10(cnt) - self.log_total
                    push(score + mult * lp, nm, nu)

                # skip branch (word not in corpus / truncated edge word)
                push(score + mult * SKIP_LP, mapping, used)

            states = sorted(next_states.values(), key=lambda s: -s[0])[:BEAM_WIDTH]
        return states

    # ---------- completion, rerank, refine ----------

    def _complete(self, mapping, symbols, cipher_counts):
        """Assign remaining symbols by frequency rank among unused targets."""
        mapping = dict(mapping)
        used = set(mapping.values())
        free_syms = sorted(
            (s for s in symbols if s not in mapping),
            key=lambda s: -cipher_counts[s],
        )
        free_targets = [t for t in self.hc.freq_ranked_jamos if t not in used]
        for s, t in zip(free_syms, free_targets):
            mapping[s] = t
        return mapping

    def solve(self, ciphertext, verbose=False):
        symbols = sorted(set(ciphertext.replace(" ", "")))
        cipher_counts = collections.Counter(ciphertext.replace(" ", ""))
        cipher_words = ciphertext.split(" ")
        word_candidates = [
            set(w for w, _ in self.pattern_map.get(get_word_pattern(cw), ()))
            if len(cw) > 2 else None
            for cw in cipher_words
        ]
        self.hc._ct = ciphertext
        self.hc._wc = word_candidates

        def decrypt(m):
            return "".join(m.get(c, c) for c in ciphertext)

        def fit(m):
            return self.hc.fitness(decrypt(m), word_candidates)

        def rerank(beam_states):
            scored = []
            for score, mapping, _ in beam_states[:30]:
                full = self._complete(mapping, symbols, cipher_counts)
                scored.append((fit(full), full))
            scored.sort(key=lambda s: -s[0])
            return scored

        def dict_coverage(mapping):
            text = decrypt(mapping)
            matched = sum(
                len(w) for w, cands in zip(text.split(" "), word_candidates)
                if cands and w in cands
            )
            return matched / max(len(ciphertext.replace(" ", "")), 1)

        rng = random.Random(0)
        scored = rerank(self._beam(cipher_words))
        if not scored:  # no usable words: fall back to pure HC
            return self.hc.solve(ciphertext, restarts=20, iterations=4000)
        best_fitness, best_mapping = scored[0]
        coverage = dict_coverage(best_mapping)

        # Low coverage usually means a fragment-edge word was truncated
        # mid-word: rerun the beam letting edge words match word pieces.
        if coverage < 0.95:
            scored2 = rerank(self._beam(cipher_words, edge_augment=True))
            if scored2 and scored2[0][0] > best_fitness:
                scored = scored2
                best_fitness, best_mapping = scored[0]
                coverage = dict_coverage(best_mapping)

        text = decrypt(best_mapping)
        if verbose:
            print(f"  beam fitness {best_fitness:.1f}, dict coverage {coverage:.0%}")

        # Almost nothing matched (e.g. spaceless text): word anchors are
        # useless, so run the full HC search and keep the better result.
        if coverage < 0.3:
            hc_text = self.hc.solve(ciphertext, restarts=20, iterations=5000)
            if self.hc.fitness(hc_text, word_candidates) > best_fitness:
                return hc_text

        sa_restarts, sa_iters = SA_RESTARTS, SA_ITERS

        if coverage < 0.85 or structure_violations(text) > 0:
            # Short anneal seeded by the beam elites
            elite = [(f, m) for f, m in scored[:5]]
            for _ in range(sa_restarts):
                mapping = self.hc._initial_mapping(
                    symbols, cipher_counts, cipher_words, rng, elite=elite
                )
                current = fit(mapping)
                T = 1.5
                for _ in range(sa_iters):
                    if len(symbols) < 2:
                        break
                    nm = dict(mapping)
                    if rng.random() < 0.5:
                        s1, s2 = rng.sample(symbols, 2)
                        nm[s1], nm[s2] = nm[s2], nm[s1]
                    else:
                        used = set(mapping.values())
                        unused = [t for t in self.hc.target_jamos if t not in used]
                        if unused:
                            nm[rng.choice(symbols)] = rng.choice(unused)
                        else:
                            s1, s2 = rng.sample(symbols, 2)
                            nm[s1], nm[s2] = nm[s2], nm[s1]
                    nf = fit(nm)
                    d = nf - current
                    if d > 0 or (T > 1e-6 and rng.random() < math.exp(d / T)):
                        mapping, current = nm, nf
                    T *= 0.999
                if current > best_fitness:
                    best_fitness, best_mapping = current, dict(mapping)
                elite.append((current, dict(mapping)))
                elite.sort(key=lambda x: -x[0])
                elite = elite[:5]

        best_mapping, _ = self.hc._greedy_polish(best_mapping, symbols, best_fitness)
        return decrypt(best_mapping)


def main():
    import json
    import time
    from benchmark import accuracy

    n_cases = int(sys.argv[1]) if len(sys.argv) > 1 else None
    with open("benchmark_set.json", "r", encoding="utf-8") as f:
        cases = json.load(f)
    if n_cases:
        cases = cases[:n_cases]

    solver = BeamSolver()
    accs = []
    t0 = time.time()
    for i, case in enumerate(cases):
        t1 = time.time()
        dec = solver.solve(case["ciphertext"], verbose=True)
        acc = accuracy(case["plaintext"], dec)
        accs.append(acc)
        print(f"[{i+1}/{len(cases)}] acc={acc:.2%} ({time.time()-t1:.1f}s)")
        print(f"  orig: {case['plaintext']}")
        print(f"  dec:  {dec}")

    avg = sum(accs) / len(accs)
    solved = sum(1 for a in accs if a >= 0.9)
    print(f"\n=== beam ===")
    print(f"Average accuracy: {avg:.2%}")
    print(f"Cases >=90% (solved): {solved}/{len(accs)}")
    print(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
