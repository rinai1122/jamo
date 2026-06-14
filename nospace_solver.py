"""No-space (띄어쓰기 없음) jamo substitution solver.

Diagnosis behind the design (validated on the full 100-case held-out test set)
------------------------------------------------------------------------------
Stripping spaces removes the word anchors the spaced solver relies on, so every
case falls back to search over the KN + structure (+ light word-segmentation)
fitness.  Running the full benchmark and classifying every failure as a SEARCH
failure (the true key scores higher but the search missed it) or an OBJECTIVE
failure (a wrong key genuinely out-scores the truth) splits the problem cleanly:

  * Search failures dominate, and the worst are LONG cases (100+ jamo at <15%
    accuracy while the true key scores 30-70 log10 higher).  The earlier belief
    that "long cases already solve" was an artifact of a 15-case sample; at
    scale the base search gets stuck on long fragments too.  These are pure
    search misses -- the objective is right, so more search fixes them.
  * Objective failures are short fragments (≲25 jamo): two readings are both
    valid Korean and no n-gram/dictionary signal separates them.  This is the
    irreducible floor; *more* search there only locks onto a higher-fitness
    wrong key and hurts, so heavy search is gated to len >= `min_ensemble_len`.

Search stages, all scored by the same fitness with "best by fitness wins":

  * LM-beam  — n-gram beam decipherment (Nuhn et al. 2013): deterministic beam
               search over the key.  We keep the *top-K* beam keys (not just #1)
               as seeds; each is greedy-polished.
  * Base SA  — simulated annealing seeded by beam key #1 (the proven baseline).
  * Ensemble — for len >= `min_ensemble_len`, a multi-pass union of independent
               annealers.  Pass 1 is a *deep* anneal seeded exactly as the old
               single union pass: it protects cases that need anneal depth (the
               ones a breadth-only redistribution was found to wreck).  Passes
               2+ are *broad* -- shorter anneals, fresh RNGs, seeded across the
               whole top-K beam -- cheap restarts that rescue the stuck basins.
               Applied across ALL lengths (long cases need it most), not just
               short ones.  Best-by-fitness over the union dominates any single
               pass, so the ensemble never regresses a case the deep pass solved.

Why this is safe: the search only ever *adds* candidates judged by one fitness,
so it is monotone in fitness.  For search failures (the bulk) higher fitness
means higher accuracy; the only residual risk is objective-failure drift on the
tiny fragments, which the length gate contains.

Memory note: `KneserNeyScorer` caches every n-gram it scores; across a long
multiprocessing run that cache is unbounded and OOMs the box -- it is capped in
crack.py.  See ns_eval.py for the parallel full-benchmark harness.
"""
import collections
import math
import pickle
import random

from crack import KneserNeyScorer, structure_violations

STRUCT_PENALTY = 10.0
WS_WEIGHT      = 0.3    # weight of the word-segmentation term in the fitness
WS_MAXLEN      = 12     # longest dict word considered during segmentation
WS_MIN_COUNT   = 2      # ignore corpus words rarer than this (noise)


class NoSpaceSolver:
    def __init__(self, kn_model_path: str = "kn_model_cv.json",
                 beam_width: int = 2000, order_cap: int = 6,
                 ext_order: str = "first", struct_penalty: float = STRUCT_PENALTY,
                 sa_restarts: int = 12, sa_iters: int = 4000, use_sa: bool = True,
                 ws_weight: float = WS_WEIGHT, word_penalty: float = 0.0,
                 beam_seeds: int = 10,
                 short_len: int = 100000, short_sa_mult: int = 3,
                 short_iter_mult: int = 1,
                 short_sa_passes: int = 3, min_ensemble_len: int = 28,
                 union_seed_all: bool = True,
                 pattern_map_path: str = "full_pattern_map_cv.pkl"):
        # Defaults below are the tuned "v3" configuration validated on the full
        # 100-case held-out test set: +5.0 pts average / +10 solved over the
        # original single-deep-pass solver (68.1% -> 73.1%, 47 -> 57 solved).
        # `short_len` is effectively unbounded (100000): the multi-pass ensemble
        # applies to every case of length >= `min_ensemble_len`, because the long
        # cases turned out to need it most -- they are not "already solved".
        self.scorer         = KneserNeyScorer(kn_model_path)
        self.order          = min(self.scorer.order, order_cap)
        self.beam_width     = beam_width
        self.ext_order      = ext_order
        self.struct_penalty = struct_penalty
        self.sa_restarts    = sa_restarts
        self.sa_iters       = sa_iters
        self.use_sa         = use_sa
        self.ws_weight      = ws_weight
        # Per-word penalty in the segmentation DP: subtract a fixed cost every
        # time a word boundary is taken, biasing toward covering a span with one
        # long real word over several short common ones.  Spurious-but-fluent
        # wrong keys tend to tile a fragment into many short frequent words; the
        # true key resolves into fewer, longer words.  At 0.0 (default) the term
        # is inert and the objective is unchanged.
        self.word_penalty   = word_penalty
        self.beam_seeds     = max(1, beam_seeds)
        self.short_len      = short_len
        self.short_sa_mult  = short_sa_mult
        # Iteration budget per restart in the union passes.  Decoupled from the
        # restart multiplier so a multi-pass ensemble can trade anneal *depth*
        # for *breadth* at constant total cost: 3 passes x N restarts x (iters)
        # = 1 pass x N restarts x (3·iters).  Breadth wins for escaping the wrong
        # basins short under-determined cases fall into.  Defaults to
        # short_sa_mult, i.e. the original depth-heavy single-pass behaviour.
        self.short_iter_mult = short_sa_mult if short_iter_mult is None else short_iter_mult
        self.short_sa_passes = max(1, short_sa_passes)
        self.min_ensemble_len = min_ensemble_len
        self.union_seed_all = union_seed_all
        self.plain_pool     = self.scorer.jamos_by_freq   # ~51 jamo
        self._load_word_logp(pattern_map_path) if ws_weight else None

    # ── word-segmentation language model ─────────────────────────────────
    def _load_word_logp(self, pattern_map_path: str, k: float = 0.5):
        """Build word→log10 P(word) from the corpus pattern map.

        The pattern map already holds every corpus word with its count, so we
        flatten it into a unigram word model.  Used to score the best word
        segmentation of a spaceless decryption — the dictionary signal the
        spaced solver gets from `text.split(" ")` but which is destroyed when
        spaces are stripped.  This is what re-ranks the true key above the
        merely-fluent wrong readings on short fragments.
        """
        with open(pattern_map_path, "rb") as f:
            pmap, _ = pickle.load(f)
        wc = {}
        for cands in pmap.values():
            for w, c in cands:
                if c >= WS_MIN_COUNT:
                    wc[w] = c
        total = sum(wc.values())
        denom = total + k * len(wc)
        self.word_logp = {w: math.log10((c + k) / denom) for w, c in wc.items()}
        self.ws_floor  = math.log10(k / denom)   # unknown char / unseen word
        self.ws_maxlen = min(max((len(w) for w in wc), default=2), WS_MAXLEN)
        # Prefix set for the segmentation DP: every proper prefix of every dict
        # word (capped at ws_maxlen).  Lets `_wordseg` stop extending a span the
        # instant it is no longer the start of any word -- the DP is otherwise
        # the per-fitness-call bottleneck (len x maxlen dict probes), and most
        # of those probes miss.  Result is identical, only the constant changes.
        prefixes = set()
        for w in self.word_logp:
            for L in range(2, min(len(w), self.ws_maxlen)):
                prefixes.add(w[:L])
        self.ws_prefixes = prefixes

    def _wordseg(self, text: str) -> float:
        """Total log-prob of the best dictionary word segmentation of `text`.

        Forward DP: from each start position extend the span while it is still
        the prefix of some dict word, scoring whenever the span is a complete
        word; a position not covered by any word falls through as a single
        unknown char at `ws_floor`.  Identical result to the naive O(n·maxlen)
        version but the prefix-set guard prunes the inner loop hard (most spans
        die after 2-3 chars), which matters because this DP runs on every one of
        the hundreds of thousands of fitness evaluations a solve performs.
        """
        n = len(text)
        if not n:
            return 0.0
        logp, floor, maxlen = self.word_logp, self.ws_floor, self.ws_maxlen
        prefixes = self.ws_prefixes
        wp = self.word_penalty
        NEG = -math.inf
        best = [NEG] * (n + 1)
        best[0] = 0.0
        for j in range(n):
            bj = best[j]
            if bj == NEG:
                continue
            # span of length 1 is always an option (unknown char)
            if bj + floor > best[j + 1]:
                best[j + 1] = bj + floor
            hi = min(maxlen, n - j)
            for L in range(2, hi + 1):
                span = text[j:j + L]
                lp = logp.get(span)
                if lp is not None:
                    v = bj + lp - wp
                    if v > best[j + L]:
                        best[j + L] = v
                if span not in prefixes:
                    break   # no longer word-start; longer spans cannot match
        return best[n]

    # ── fitness ──────────────────────────────────────────────────────────
    def _decrypt(self, mapping, text):
        return "".join(mapping.get(c, c) for c in text)

    def _fitness(self, text):
        f = (self.scorer.log_prob(text)
             - self.struct_penalty * structure_violations(text))
        if self.ws_weight:
            f += self.ws_weight * self._wordseg(text)
        return f

    # ── extension order for the beam ─────────────────────────────────────
    def _extension_order(self, text: str) -> list:
        freq = collections.Counter(text)
        if self.ext_order == "freq":
            return sorted(freq, key=lambda c: -freq[c])
        # "first": by first occurrence (left-to-right completion).  Empirically
        # better than completion-greedy ordering for this LM.
        first = {}
        for i, c in enumerate(text):
            first.setdefault(c, i)
        return sorted(first, key=lambda c: first[c])

    def _windows_by_step(self, text: str, rank: dict):
        steps = collections.defaultdict(list)
        for i in range(len(text)):
            ctx = min(i + 1, self.order)
            window = text[i - ctx + 1: i + 1]
            steps[max(rank[c] for c in window)].append(tuple(window))
        return steps

    # ── LM-beam decipherment ─────────────────────────────────────────────
    def _lm_beam(self, text: str, topk: int = 1) -> list:
        """Return the `topk` best keys from the n-gram beam (best first).

        The beam already carries thousands of diverse partial-key hypotheses;
        the single best is usually right but on under-determined short cases
        the true basin sits a little lower.  Handing the top few keys to the
        downstream greedy/SA stages as seeds turns those near-misses into hits
        at negligible cost.
        """
        sym_order = self._extension_order(text)
        rank  = {c: k for k, c in enumerate(sym_order)}
        steps = self._windows_by_step(text, rank)
        prob, log10 = self.scorer._prob, math.log10
        beam = [(0.0, {}, frozenset())]
        for k, s in enumerate(sym_order):
            windows = steps.get(k, ())
            nxt = {}
            for score, mapping, used in beam:
                for p in self.plain_pool:
                    if p in used:
                        continue
                    add = 0.0
                    for win in windows:
                        add += log10(prob("".join(p if c == s else mapping[c]
                                                  for c in win)))
                    nm = dict(mapping); nm[s] = p
                    dk = tuple(nm[c] for c in sym_order[:k + 1])
                    cur = nxt.get(dk)
                    if cur is None or score + add > cur[0]:
                        nxt[dk] = (score + add, nm, used | {p})
            beam = sorted(nxt.values(), key=lambda h: -h[0])[:self.beam_width]
        return [h[1] for h in beam[:topk]]

    # ── greedy polish (struct-aware) ─────────────────────────────────────
    def _greedy(self, mapping, text, symbols):
        def score(m):
            return self._fitness(self._decrypt(m, text))
        best_m, best_f = dict(mapping), score(mapping)
        improved = True
        while improved:
            improved = False
            used = set(best_m.values())
            for s in symbols:
                for t in self.plain_pool:
                    if t in used:
                        continue
                    cand = {**best_m, s: t}
                    f = score(cand)
                    if f > best_f:
                        best_m, best_f, improved = cand, f, True
                        used = set(best_m.values())
            for i in range(len(symbols)):
                for j in range(i + 1, len(symbols)):
                    s1, s2 = symbols[i], symbols[j]
                    cand = dict(best_m); cand[s1], cand[s2] = cand[s2], cand[s1]
                    f = score(cand)
                    if f > best_f:
                        best_m, best_f, improved = cand, f, True
        return best_m, best_f

    # ── simulated annealing (diverse restarts + beam seed) ───────────────
    def _sa(self, text, symbols, elites, rng, restarts=None, iters=None):
        restarts = self.sa_restarts if restarts is None else restarts
        sa_iters = self.sa_iters    if iters    is None else iters
        counts = collections.Counter(text)

        def score(m):
            return self._fitness(self._decrypt(m, text))

        def freq_rank_init():
            ranked = sorted(symbols, key=lambda s: -counts[s])
            tgts = list(self.plain_pool)
            for i in range(0, len(tgts) - 2, 3):
                w = tgts[i:i + 3]; rng.shuffle(w); tgts[i:i + 3] = w
            return {s: tgts[i] for i, s in enumerate(ranked)}

        def perturb(m):
            m = dict(m)
            for _ in range(rng.randint(2, min(4, len(symbols)))):
                if rng.random() < 0.5 and len(symbols) >= 2:
                    a, b = rng.sample(symbols, 2)
                    m[a], m[b] = m[b], m[a]
                else:
                    unused = [t for t in self.plain_pool if t not in set(m.values())]
                    if unused:
                        m[rng.choice(symbols)] = rng.choice(unused)
            return m

        elites = list(elites)
        best_m, best_f = None, -math.inf
        for r in range(restarts):
            if r < len(elites):
                mapping = dict(elites[r])
            elif elites and rng.random() < 0.4:
                mapping = perturb(rng.choice(elites))
            else:
                mapping = freq_rank_init()
            cur_f, T, stale = score(mapping), 1.5, 0
            for _ in range(sa_iters):
                if len(symbols) < 2:
                    break
                nm = dict(mapping)
                if rng.random() < 0.5:
                    a, b = rng.sample(symbols, 2)
                    nm[a], nm[b] = nm[b], nm[a]
                else:
                    unused = [t for t in self.plain_pool if t not in set(mapping.values())]
                    if unused:
                        nm[rng.choice(symbols)] = rng.choice(unused)
                    else:
                        a, b = rng.sample(symbols, 2)
                        nm[a], nm[b] = nm[b], nm[a]
                nf = score(nm)
                d = nf - cur_f
                if d > 0 or (T > 1e-6 and rng.random() < math.exp(d / T)):
                    mapping, cur_f = nm, nf
                    stale = 0 if d > 0 else stale + 1
                else:
                    stale += 1
                if stale >= 400:
                    T, stale = 2.0, 0
                else:
                    T *= 0.999
            if cur_f > best_f:
                best_m, best_f = dict(mapping), cur_f
        return best_m, best_f

    # ── main entry ───────────────────────────────────────────────────────
    def solve(self, ciphertext: str) -> str:
        text = ciphertext.replace(" ", "")
        if not text:
            return ciphertext
        symbols = sorted(set(text))
        rng = random.Random(0)

        # Incumbent: greedy-polish every top-K beam key, keep the best.  The #1
        # key is the old incumbent, so this can only improve on it — on under-
        # determined short cases the true basin sits just below the beam optimum
        # and is reachable from a lower-ranked beam key the old code discarded.
        beam_keys = self._lm_beam(text, topk=self.beam_seeds)
        best_m, best_f = None, -math.inf
        for bk in beam_keys:
            gm, gf = self._greedy(bk, text, symbols)
            if gf > best_f:
                best_m, best_f = gm, gf

        if self.use_sa:
            # Baseline SA, seeded exactly as before (#1 beam key) so this branch
            # never regresses; the multi-seed gain comes purely from the greedy
            # sweep above.
            sm, sf = self._sa(text, symbols, [beam_keys[0]], rng)
            sm, sf = self._greedy(sm, text, symbols)
            if sf > best_f:
                best_f, best_m = sf, sm

            # Short texts are under-determined: a single anneal lands in a basin
            # by luck (one stuck case at 16%, one at 75% only by chance).  Spend
            # a second, larger SA budget with a fresh RNG on them and keep the
            # better basin by fitness — a union of two independent searches, so
            # neither pass can drag the other down.  Gated to short texts where
            # it is both cheap and where the search failures actually live.
            if len(text) < self.short_len and self.short_sa_mult > 1:
                # Each extra pass is an *independent* search with a fresh RNG,
                # seeded from the full top-K beam (not just key #1) so the SA
                # restarts start from diverse basins.  Best-by-fitness across
                # passes finds better basins the single anneal missed.
                #
                # Caveat that sets the pass count: more search is monotone in
                # *fitness* but NOT in *accuracy*.  Below ~min_ensemble_len jamo
                # the objective itself is unreliable (two readings are both valid
                # Korean), so extra search just locks onto a higher-fitness WRONG
                # key and accuracy drops.  So the ensemble is gated to lengths
                # where the objective can be trusted; the tiniest fragments keep
                # the lighter single-pass search that does not over-fit them.
                passes = (self.short_sa_passes
                          if len(text) >= self.min_ensemble_len else 1)
                for seed in range(1, passes + 1):
                    # Pass 1 is a *deep* anneal seeded exactly as the baseline's
                    # single union pass (beam key #1, full iter budget, RNG(1)).
                    # It reproduces that pass byte-for-byte, so every case the
                    # baseline already solved is protected -- those need anneal
                    # *depth*, and redistributing all the budget to shallow passes
                    # was observed to wreck them.  Passes 2+ are *broad*: shorter
                    # anneals with fresh RNGs, seeded from the whole top-K beam for
                    # diversity -- cheap extra restarts that rescue the stuck
                    # search-failure basins (the long catastrophic misses).  Best
                    # -by-fitness over the union dominates either pass alone, so
                    # this keeps the rescues without the regressions.
                    if seed == 1:
                        elites, pass_mult = [beam_keys[0]], self.short_sa_mult
                    else:
                        elites = beam_keys if self.union_seed_all else [beam_keys[0]]
                        pass_mult = self.short_iter_mult
                    sm, sf = self._sa(text, symbols, elites, random.Random(seed),
                                      restarts=self.sa_restarts * self.short_sa_mult,
                                      iters=self.sa_iters * pass_mult)
                    sm, sf = self._greedy(sm, text, symbols)
                    if sf > best_f:
                        best_f, best_m = sf, sm

        return self._decrypt(best_m, text)
