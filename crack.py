"""
Korean jamo substitution cipher cracker — beam search solver.

A substitution cipher on Korean jamo (sub-character units) is bijective,
which means it preserves the character-repetition pattern of every word.
This lets us index corpus words by their pattern, so each cipher word has
a small candidate set (a 10-jamo word typically has ~40 matches, a 12-jamo
word ~3).  Beam search processes words most-constrained-first, growing a
beam of partial key hypotheses, then refines the winner with greedy polish.

Usage
-----
    python crack.py [num_cases]          # benchmark on benchmark_set.json
    python crack.py --solve "<cipher>"   # crack a single ciphertext

Prerequisites
-------------
    kn_model.json          Kneser-Ney 6-gram language model
                           (build with train_kn_final.py)
    corpus.txt             plain Korean sentences, one per line
                           (pattern map is built from this on first run)
"""
import collections
import io
import json
import math
import os
import pickle
import random
import sys

if not isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


# ---------------------------------------------------------------------------
# Hangul → jamo decomposition (standard Unicode arithmetic, no external libs)
# ---------------------------------------------------------------------------
# Hangul syllable block layout (U+AC00–U+D7A3):
#   code_point = 0xAC00 + 588*initial + 28*vowel + coda
#   (coda index 0 means no coda)

_INITIALS = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"       # 19 consonants
_VOWELS   = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"    # 21 vowels
_CODAS    = " ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"  # 28 (0 = none)

def decompose(text: str) -> str:
    """Hangul syllable blocks → compatibility jamo; spaces preserved."""
    out = []
    for ch in text:
        cp = ord(ch)
        if 0xAC00 <= cp <= 0xD7A3:
            n = cp - 0xAC00
            out.append(_INITIALS[n // 588])
            out.append(_VOWELS[(n % 588) // 28])
            coda = _CODAS[n % 28]
            if coda != " ":
                out.append(coda)
        elif ch == " ":
            out.append(" ")
        # non-Hangul, non-space characters are dropped
    return "".join(out)


# ---------------------------------------------------------------------------
# Jamo phonotactics
# ---------------------------------------------------------------------------

def is_vowel(c: str) -> bool:
    return 0x314F <= ord(c) <= 0x3163

def is_consonant(c: str) -> bool:
    return 0x3131 <= ord(c) <= 0x314E

def structure_violations(text: str) -> int:
    """Count deviations from the (C V C?)+ Korean syllable structure."""
    violations = 0
    # DFA states:  0 = expect initial consonant  (word start or after coda)
    #              1 = expect vowel
    #              2 = after vowel   (can take a coda or start a new syllable)
    #              3 = after coda    (next V continues, next C = coda+initial)
    state = 0
    for ch in text:
        if ch == " ":
            state = 0
            continue
        v = is_vowel(ch)
        if state == 0:
            violations += v          # word-initial vowel is illegal
            state = 2 if v else 1
        elif state == 1:
            if v:
                state = 2
            else:
                violations += 1      # two initials in a row
        elif state == 2:
            violations += v          # two vowels in a row
            if not v:
                state = 3
        else:                        # state 3
            state = 2 if v else 1
    return violations


# ---------------------------------------------------------------------------
# Word-pattern fingerprint
# ---------------------------------------------------------------------------

def word_pattern(word: str) -> str:
    """Canonical repetition pattern for substitution-cipher equivalence.

    "ㅎㅏㄴ"   → "0.1.2"
    "ㅇㅏㄴㅇㅏ" → "0.1.2.0.1"
    """
    seen: dict[str, str] = {}
    idx = 0
    parts = []
    for ch in word:
        if ch not in seen:
            seen[ch] = str(idx)
            idx += 1
        parts.append(seen[ch])
    return ".".join(parts)


# ---------------------------------------------------------------------------
# Kneser-Ney language model
# ---------------------------------------------------------------------------

class KneserNeyScorer:
    """Order-6 Kneser-Ney language model, loaded from a JSON file."""

    def __init__(self, model_path: str):
        with open(model_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.order          = data["order"]
        self.discount       = data["discount"]
        self.vocab          = set(data["vocab"])
        self.counts         = [collections.Counter(c) for c in data["counts"]]
        self.unique_follows = [collections.Counter(c) for c in data["unique_follows"]]
        self.total_unigrams = sum(self.counts[1].values())
        self._cache: dict[str, float] = {}
        # The cache is pure memoization (never affects results), but it would
        # otherwise grow without bound: a long solve scores millions of distinct
        # n-grams (every position of every decryption the search explores), and
        # a multiprocessing worker reuses one scorer across many cases.  Left
        # uncapped this exhausts RAM and the box swaps to a crawl, then OOMs.
        # Cap the size and drop the cache wholesale when it is hit -- cheaper
        # than per-entry LRU bookkeeping and the within-case hit rate recovers
        # almost immediately.
        self._cache_cap = 1_000_000

    def log_prob(self, sequence: str) -> float:
        """Sum of log₁₀ P(jamo_i | context) for every position."""
        total = 0.0
        for i in range(len(sequence)):
            ctx = min(i + 1, self.order)
            total += math.log10(self._prob(sequence[i - ctx + 1: i + 1]))
        return total

    def _prob(self, ngram: str) -> float:
        p = self._cache.get(ngram)
        if p is not None:
            return p
        n = len(ngram)
        if n == 1:
            p = (self.counts[1].get(ngram, 0) + 0.1) / (
                self.total_unigrams + 0.1 * len(self.vocab)
            )
        else:
            prefix    = ngram[:-1]
            count     = self.counts[n].get(ngram, 0)
            pfx_count = self.counts[n - 1].get(prefix, 0)
            if pfx_count > 0:
                lam = self.discount * self.unique_follows[n - 1].get(prefix, 0) / pfx_count
                p   = max(count - self.discount, 0) / pfx_count + lam * self._prob(ngram[1:])
                p   = max(p, 1e-12)
            else:
                p = self._prob(ngram[1:])
        if len(self._cache) >= self._cache_cap:
            self._cache.clear()
        self._cache[ngram] = p
        return p

    @property
    def jamos_by_freq(self) -> list:
        """All jamo characters in the vocabulary, sorted most-frequent first."""
        return sorted(
            (j for j in self.vocab if 0x3131 <= ord(j) <= 0x3163),
            key=lambda j: -self.counts[1].get(j, 0),
        )


# ---------------------------------------------------------------------------
# Pattern map  (corpus index: word_pattern → candidate plaintext words)
# ---------------------------------------------------------------------------

_PATTERN_MAP_CACHE = "full_pattern_map.pkl"
_CAND_CAP = 6000  # top candidates stored per pattern

def _build_pattern_map(corpus_path: str,
                       cache_path: str = _PATTERN_MAP_CACHE) -> tuple:
    """Build and cache  pattern → [(jamo_word, count), …]  sorted by count.

    Every distinct corpus word is indexed (not just the top 10k) so that
    rare-but-real words can still anchor the substitution key.
    """
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    word_counts: collections.Counter = collections.Counter()
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            for w in line.split():
                jw = decompose(w)
                jw = "".join(c for c in jw if 0x3131 <= ord(c) <= 0x3163)
                if len(jw) >= 2:
                    word_counts[jw] += 1

    pmap: dict = collections.defaultdict(list)
    for w, count in word_counts.items():
        pmap[word_pattern(w)].append((w, count))
    for p in pmap:
        pmap[p].sort(key=lambda wc: -wc[1])
        pmap[p] = pmap[p][:_CAND_CAP]

    result = (dict(pmap), sum(word_counts.values()))
    with open(cache_path, "wb") as f:
        pickle.dump(result, f)
    return result


# ---------------------------------------------------------------------------
# Beam solver
# ---------------------------------------------------------------------------

# ── tuning constants ────────────────────────────────────────────────────────
BEAM_WIDTH   = 500   # hypotheses kept after each word
BRANCH_CAP   = 200   # candidates tried per (state, cipher_word) pair
EDGE_CAP     = 800   # prefix/suffix candidates per truncated edge word
SKIP_LP      = -10.0 # beam score awarded for skipping an unknown word
EDGE_PENALTY = 1.0   # log₁₀ penalty applied to edge-word candidates

SA_RESTARTS  = 8     # SA restarts in the beam-seeded polish pass
SA_ITERS     = 1500  # SA iterations per restart in the polish pass

STRUCT_PENALTY = 10.0  # fitness penalty per syllable-structure violation
DICT_BONUS     = 4.0   # fitness bonus per jamo char in a matched corpus word


class BeamSolver:
    """
    Solve Korean jamo substitution ciphers.

    Pipeline
    --------
    1. Beam search       — most-constrained-first word expansion; keeps the
                           top BEAM_WIDTH partial key hypotheses.
    2. Complete & rank   — fill unmapped symbols by frequency rank; score
                           with the KN language model + phonotactics + dict.
    3. Edge augment      — if word-level coverage is low, retry allowing
                           corpus words to match as prefixes/suffixes of
                           truncated fragment-edge words.
    4. SA polish         — short beam-seeded simulated annealing when coverage
                           is still weak after steps 1–3.
    5. Greedy polish     — exhaustive pairwise-swap + unused-reassign pass.
    """

    def __init__(self, kn_model_path: str = "kn_model.json",
                 corpus_path: str = "corpus.txt",
                 disable: set = None,
                 pattern_map_cache: str = _PATTERN_MAP_CACHE):
        # `disable` names stages/terms to switch off (for ablation studies):
        #   "kn", "struct", "dict", "beam", "edge", "sa_fallback",
        #   "sa_polish", "greedy".  Empty set = full pipeline (default).
        self.disable      = disable or set()
        self.scorer       = KneserNeyScorer(kn_model_path)
        self.target_jamos = self.scorer.jamos_by_freq   # most-frequent first
        self.pattern_map, total = _build_pattern_map(corpus_path, pattern_map_cache)
        self.log_total = math.log10(total)
        # Flat list sorted by count — used for edge-word truncation matching.
        self._all_words = sorted(
            (wc for cands in self.pattern_map.values() for wc in cands),
            key=lambda wc: -wc[1],
        )

    # ── fitness ──────────────────────────────────────────────────────────────

    def _fitness(self, text: str, word_cands: list) -> float:
        score = 0.0 if "kn" in self.disable else self.scorer.log_prob(text)
        if "struct" not in self.disable:
            score -= STRUCT_PENALTY * structure_violations(text)
        if "dict" not in self.disable:
            for word, cands in zip(text.split(" "), word_cands):
                if cands and word in cands:
                    score += DICT_BONUS * len(word)
        return score

    # ── beam search ──────────────────────────────────────────────────────────

    def _edge_candidates(self, cipher_word: str, side: str) -> list:
        """Candidates for a fragment-edge word that was cut mid-word.

        The first word of a corpus fragment may be a word *suffix*; the last
        word a word *prefix*.  Scans the full word list and collects pieces
        whose pattern matches the cipher word.
        """
        pat = word_pattern(cipher_word)
        L   = len(cipher_word)
        found: dict[str, int] = {}
        for w, count in self._all_words:
            if len(w) <= L:
                continue
            piece = w[-L:] if side == "suffix" else w[:L]
            if word_pattern(piece) == pat:
                found[piece] = max(found.get(piece, 0), count)
                if len(found) >= EDGE_CAP:
                    break
        return sorted(found.items(), key=lambda wc: -wc[1])

    def _beam(self, cipher_words: list, edge_augment: bool = False) -> list:
        """Return beam states: [(score, mapping_dict, used_targets_frozenset)]."""
        word_mult = collections.Counter(cw for cw in cipher_words if len(cw) >= 2)

        # Identify which edge words (first/last in fragment) may be truncated.
        edge_sides: dict = collections.defaultdict(list)
        if edge_augment:
            real = [cw for cw in cipher_words if len(cw) >= 2]
            if real and len(real[0]) >= 3:
                edge_sides[real[0]].append("suffix")
            if len(real) > 1 and len(real[-1]) >= 3:
                edge_sides[real[-1]].append("prefix")

        # Build (cipher_word, multiplicity, candidates) jobs.
        # Sorting by fewest candidates first means the beam prunes most
        # aggressively at the start, where states are most numerous.
        jobs = []
        for cw, mult in word_mult.items():
            cands = list(self.pattern_map.get(word_pattern(cw), []))
            if cw in edge_sides:
                # Merge in edge-word (truncated) candidates at a slight penalty.
                penalty = 10 ** (-EDGE_PENALTY)
                merged  = dict(cands)
                for side in edge_sides[cw]:
                    for piece, c in self._edge_candidates(cw, side):
                        merged[piece] = max(merged.get(piece, 0), int(c * penalty) or 1)
                cands = sorted(merged.items(), key=lambda wc: -wc[1])
            jobs.append((cw, mult, cands))
        jobs.sort(key=lambda j: len(j[2]))

        # States: list of (score, cipher→plain mapping, frozenset of used targets).
        states: list = [(0.0, {}, frozenset())]

        for cw, mult, cands in jobs:
            # Position index: (position, plain_char) → {candidate_ids}
            # Used to quickly filter candidates consistent with the current mapping.
            pos_index: dict = collections.defaultdict(set)
            for ci, (w, _) in enumerate(cands):
                for pos, ch in enumerate(w):
                    pos_index[(pos, ch)].add(ci)

            # next_states: key → best (score, mapping, used) for that key.
            next_states: dict = {}

            def push(score, mapping, used, _ns=next_states):
                key = frozenset(mapping.items())
                cur = _ns.get(key)
                if cur is None or score > cur[0]:
                    _ns[key] = (score, mapping, used)

            for score, mapping, used in states:
                # Narrow candidates to those consistent with already-mapped chars.
                fixed = [(i, mapping[c]) for i, c in enumerate(cw) if c in mapping]
                if fixed:
                    sets = [pos_index.get(fx) for fx in fixed]
                    ids  = (
                        sorted(set.intersection(*sets))
                        if all(s is not None for s in sets)
                        else []
                    )
                else:
                    ids = range(len(cands))

                n_branched = 0
                for ci in ids:
                    if n_branched >= BRANCH_CAP:
                        break
                    w, cnt = cands[ci]
                    # Try to extend the mapping with this candidate word.
                    new_pairs: dict = {}
                    ok = True
                    for c_ch, p_ch in zip(cw, w):
                        if c_ch in mapping:
                            continue  # already mapped; consistency guaranteed by pos_index
                        prev = new_pairs.get(c_ch)
                        if prev is not None:
                            if prev != p_ch:
                                ok = False; break
                            continue
                        if p_ch in used:
                            ok = False; break  # bijectivity violated
                        new_pairs[c_ch] = p_ch
                    if not ok or len(set(new_pairs.values())) != len(new_pairs):
                        continue
                    n_branched += 1
                    push(
                        score + mult * (math.log10(cnt) - self.log_total),
                        {**mapping, **new_pairs},
                        used | frozenset(new_pairs.values()),
                    )

                # Skip branch: this word has no corpus match (or is an edge fragment).
                push(score + mult * SKIP_LP, mapping, used)

            states = sorted(next_states.values(), key=lambda s: -s[0])[:BEAM_WIDTH]

        return states

    # ── completion and re-ranking ─────────────────────────────────────────────

    def _complete(self, mapping: dict, symbols: list,
                  cipher_counts: collections.Counter) -> dict:
        """Assign unmapped symbols by unigram-frequency rank among unused targets."""
        used      = set(mapping.values())
        free_syms = sorted((s for s in symbols if s not in mapping),
                           key=lambda s: -cipher_counts[s])
        free_tgts = [t for t in self.target_jamos if t not in used]
        return {**mapping, **dict(zip(free_syms, free_tgts))}

    def _rerank(self, beam_states: list, symbols: list,
                cipher_counts: collections.Counter, ciphertext: str,
                word_cands: list, top: int = 30) -> list:
        """Complete partial keys, score with full fitness, return sorted list."""
        scored = []
        for _, mapping, _ in beam_states[:top]:
            m    = self._complete(mapping, symbols, cipher_counts)
            text = "".join(m.get(c, c) for c in ciphertext)
            scored.append((self._fitness(text, word_cands), m))
        scored.sort(key=lambda x: -x[0])
        return scored

    # ── greedy polish ─────────────────────────────────────────────────────────

    def _greedy_polish(self, mapping: dict, symbols: list,
                       ciphertext: str, word_cands: list) -> tuple:
        """Exhaustive pairwise-swap + unused-reassign until no improvement."""
        def score(m):
            return self._fitness("".join(m.get(c, c) for c in ciphertext), word_cands)

        best_m, best_f = mapping, score(mapping)
        improved = True
        while improved:
            improved = False
            used = set(best_m.values())
            for s in symbols:
                for t in self.target_jamos:
                    if t in used:
                        continue
                    cand = {**best_m, s: t}
                    f = score(cand)
                    if f > best_f:
                        best_m, best_f, improved = cand, f, True
            for i in range(len(symbols)):
                for j in range(i + 1, len(symbols)):
                    s1, s2 = symbols[i], symbols[j]
                    cand = dict(best_m)
                    cand[s1], cand[s2] = cand[s2], cand[s1]
                    f = score(cand)
                    if f > best_f:
                        best_m, best_f, improved = cand, f, True
        return best_m, best_f

    # ── simulated annealing (fallback / seeded polish) ────────────────────────

    def _sa_solve(self, ciphertext: str, word_cands: list,
                  restarts: int, iters: int,
                  T_init: float = 1.5,
                  elite: list = None,
                  rng: random.Random = None) -> tuple:
        """Simulated-annealing key search.

        When *elite* is a list of (fitness, mapping) pairs, 70 % of restarts
        are seeded by a lightly perturbed elite mapping; the rest use a
        corpus-frequency initialisation.

        Returns (best_mapping, best_fitness).
        """
        if rng is None:
            rng = random.Random()
        symbols = sorted(set(ciphertext.replace(" ", "")))
        counts  = collections.Counter(ciphertext.replace(" ", ""))
        elite   = list(elite or [])

        def score(m):
            return self._fitness("".join(m.get(c, c) for c in ciphertext), word_cands)

        def freq_rank_init() -> dict:
            ranked = sorted(symbols, key=lambda s: -counts[s])
            tgts   = list(self.target_jamos)
            for i in range(0, len(tgts) - 2, 3):   # small local shuffle for variety
                w = tgts[i: i + 3]; rng.shuffle(w); tgts[i: i + 3] = w
            return {s: tgts[i] for i, s in enumerate(ranked)}

        def perturb(m: dict) -> dict:
            m = dict(m)
            for _ in range(rng.randint(2, min(4, len(symbols)))):
                if rng.random() < 0.5 and len(symbols) >= 2:
                    s1, s2 = rng.sample(symbols, 2)
                    m[s1], m[s2] = m.get(s2), m.get(s1)
                else:
                    unused = [t for t in self.target_jamos if t not in set(m.values())]
                    if unused:
                        m[rng.choice(symbols)] = rng.choice(unused)
            return m

        best_m, best_f = None, -math.inf

        for _ in range(restarts):
            if elite and rng.random() < 0.70:
                mapping = perturb(rng.choice(elite)[1])
            else:
                mapping = freq_rank_init()

            current_f = score(mapping)
            T         = T_init
            stale     = 0  # steps without improvement (used for adaptive reheat)

            for _ in range(iters):
                if len(symbols) < 2:
                    break
                nm = dict(mapping)
                if rng.random() < 0.5:
                    s1, s2 = rng.sample(symbols, 2)
                    nm[s1], nm[s2] = nm[s2], nm[s1]
                else:
                    unused = [t for t in self.target_jamos if t not in set(mapping.values())]
                    if unused:
                        nm[rng.choice(symbols)] = rng.choice(unused)
                    else:
                        s1, s2 = rng.sample(symbols, 2)
                        nm[s1], nm[s2] = nm[s2], nm[s1]

                nf    = score(nm)
                delta = nf - current_f
                if delta > 0 or (T > 1e-6 and rng.random() < math.exp(delta / T)):
                    mapping, current_f = nm, nf
                    stale = 0 if delta > 0 else stale + 1
                else:
                    stale += 1

                # Reheat if stuck — helps escape deep local optima.
                if stale >= 400:
                    T, stale = 2.0, 0
                else:
                    T *= 0.999

            if current_f > best_f:
                best_m, best_f = mapping, current_f
            elite.append((current_f, dict(mapping)))
            elite.sort(key=lambda x: -x[0])
            elite = elite[:5]

        return best_m, best_f

    # ── main entry point ──────────────────────────────────────────────────────

    def solve(self, ciphertext: str, verbose: bool = False) -> str:
        symbols       = sorted(set(ciphertext.replace(" ", "")))
        cipher_counts = collections.Counter(ciphertext.replace(" ", ""))
        cipher_words  = ciphertext.split(" ")
        word_cands    = [
            set(w for w, _ in self.pattern_map.get(word_pattern(cw), ()))
            if len(cw) > 2 else None
            for cw in cipher_words
        ]

        def decrypt(m):
            return "".join(m.get(c, c) for c in ciphertext)

        def coverage(m) -> float:
            text    = decrypt(m)
            matched = sum(
                len(w)
                for w, cands in zip(text.split(" "), word_cands)
                if cands and w in cands
            )
            return matched / max(len(ciphertext.replace(" ", "")), 1)

        rng = random.Random(0)

        # ── 1. Beam search ────────────────────────────────────────────────────
        if "beam" in self.disable:
            scored = []   # force the SA fallback path below (no word anchors)
        else:
            scored = self._rerank(
                self._beam(cipher_words), symbols, cipher_counts, ciphertext, word_cands
            )
        if not scored:
            # No usable words (very short / space-less input): fall back to SA.
            m, _ = self._sa_solve(ciphertext, word_cands, restarts=20, iters=5000,
                                  T_init=3.0)
            m, _ = self._greedy_polish(m, symbols, ciphertext, word_cands)
            return decrypt(m)

        best_f, best_m = scored[0]
        cov = coverage(best_m)

        # ── 2. Edge augment (retry if fragment words may be cut mid-word) ─────
        if cov < 0.95 and "edge" not in self.disable:
            scored2 = self._rerank(
                self._beam(cipher_words, edge_augment=True),
                symbols, cipher_counts, ciphertext, word_cands,
            )
            if scored2 and scored2[0][0] > best_f:
                scored = scored2
                best_f, best_m = scored[0]
                cov = coverage(best_m)

        if verbose:
            print(f"  beam: fitness={best_f:.1f}  coverage={cov:.0%}")

        # ── 3. Full SA fallback (nearly nothing matched → word anchors useless) ─
        if cov < 0.3 and "sa_fallback" not in self.disable:
            sa_m, sa_f = self._sa_solve(ciphertext, word_cands,
                                        restarts=20, iters=5000, T_init=3.0)
            if sa_f > best_f:
                best_m = sa_m
                best_f = sa_f

        # ── 4. Beam-seeded SA polish (when coverage is still shaky) ──────────
        if ("sa_polish" not in self.disable
                and (cov < 0.85 or structure_violations(decrypt(best_m)) > 0)):
            sa_m, sa_f = self._sa_solve(
                ciphertext, word_cands,
                restarts=SA_RESTARTS, iters=SA_ITERS,
                elite=scored[:5], rng=rng,
            )
            if sa_f > best_f:
                best_m, best_f = sa_m, sa_f

        # ── 5. Greedy polish ──────────────────────────────────────────────────
        if "greedy" not in self.disable:
            best_m, _ = self._greedy_polish(best_m, symbols, ciphertext, word_cands)
        return decrypt(best_m)


# ---------------------------------------------------------------------------
# CLI / benchmark
# ---------------------------------------------------------------------------

def _accuracy(plaintext: str, decrypted: str) -> float:
    n = min(len(plaintext), len(decrypted))
    return sum(a == b for a, b in zip(plaintext, decrypted)) / len(plaintext)


def main():
    import time

    args = sys.argv[1:]

    if args and args[0] == "--solve":
        solver = BeamSolver()
        print(solver.solve(args[1], verbose=True))
        return

    n_cases = int(args[0]) if args else None
    with open("benchmark_set.json", "r", encoding="utf-8") as f:
        cases = json.load(f)
    if n_cases:
        cases = cases[:n_cases]

    solver = BeamSolver()
    accs   = []
    t0     = time.time()
    for i, case in enumerate(cases):
        t1  = time.time()
        dec = solver.solve(case["ciphertext"], verbose=True)
        acc = _accuracy(case["plaintext"], dec)
        accs.append(acc)
        print(f"[{i+1}/{len(cases)}] acc={acc:.2%}  ({time.time()-t1:.1f}s)")
        print(f"  orig: {case['plaintext']}")
        print(f"  dec:  {dec}")

    avg     = sum(accs) / len(accs)
    solved  = sum(1 for a in accs if a >= 0.9)
    elapsed = time.time() - t0
    print(f"\n=== Results ===")
    print(f"Average accuracy : {avg:.2%}")
    print(f"Solved (≥90 %)   : {solved}/{len(accs)}")
    print(f"Total time       : {elapsed:.1f}s")


if __name__ == "__main__":
    main()
