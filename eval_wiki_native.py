"""Eval beam solver trained entirely on Korean Wikipedia.

Builds:
  kn_model_wiki.json          -- KN-7 trained on corpus_wiki.txt
  full_pattern_map_wiki.pkl   -- pattern map from corpus_wiki.txt

Then runs beam_solver on benchmark_wiki.json using these wiki-only artifacts.
This tests whether the METHOD generalises, not whether NSMC is special.
"""
import io, json, sys, time
if not isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

WIKI_CORPUS  = "corpus_wiki.txt"
WIKI_KN      = "kn_model_wiki.json"
WIKI_PMAP    = "full_pattern_map_wiki.pkl"
BENCH_PATH   = "benchmark_wiki.json"
KN_LINES     = 50_000

# ── 1. build KN model if needed ──────────────────────────────────────────────
import os
if not os.path.exists(WIKI_KN):
    import collections, math
    from jamo import h2j, j2hcj

    def get_jamo(text):
        return j2hcj(h2j(text))

    class KneserNeyLM:
        def __init__(self, order=7, discount=0.75):
            self.order = order
            self.discount = discount
            self.counts = [collections.Counter() for _ in range(order + 1)]
            self.prefix_unique_follows = [collections.Counter() for _ in range(order)]
            self.suffix_unique_precedes = [collections.Counter() for _ in range(order + 1)]
            self.vocab = set()

        def train(self, sentences):
            print(f"Training order-{self.order} KN on {len(sentences):,} lines …")
            for text in sentences:
                jamos = get_jamo(text)
                self.vocab.update(jamos)
                for n in range(1, self.order + 1):
                    for i in range(len(jamos) - n + 1):
                        ngram = jamos[i:i+n]
                        self.counts[n][ngram] += 1
                        if n > 1:
                            self.prefix_unique_follows[n-1][ngram[:-1]] += 1
                            self.suffix_unique_precedes[n][ngram[1:]] += 1

        def save(self, path):
            data = {
                "order": self.order,
                "discount": self.discount,
                "vocab": list(self.vocab),
                "counts": [{k: v for k, v in c.items() if v > 1 or len(k) == 1}
                           for c in self.counts],
                "unique_follows": [dict(c) for c in self.prefix_unique_follows],
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            print(f"Saved {path}")

    with open(WIKI_CORPUS, encoding="utf-8") as f:
        sentences = [l.strip() for l in f if l.strip()][:KN_LINES]
    kn = KneserNeyLM(order=7)
    kn.train(sentences)
    kn.save(WIKI_KN)
else:
    print(f"Using cached {WIKI_KN}")

# ── 2. build pattern map if needed ───────────────────────────────────────────
import beam_solver as bs
_orig_cache = bs.PATTERN_CACHE
bs.PATTERN_CACHE = WIKI_PMAP          # redirect cache path before first call

if not os.path.exists(WIKI_PMAP):
    print("Building wiki pattern map …")
    bs.build_full_pattern_map(WIKI_CORPUS)
else:
    print(f"Using cached {WIKI_PMAP}")

# ── 3. instantiate solver with wiki artifacts ─────────────────────────────────
solver = bs.BeamSolver(kn_model_path=WIKI_KN, corpus_path=WIKI_CORPUS)

# ── 4. run benchmark ──────────────────────────────────────────────────────────
from benchmark import accuracy

with open(BENCH_PATH, encoding="utf-8") as f:
    cases = json.load(f)

n = int(sys.argv[1]) if len(sys.argv) > 1 else len(cases)
cases = cases[:n]

accs = []
t0 = time.time()
for i, case in enumerate(cases):
    t1 = time.time()
    dec = solver.solve(case["ciphertext"], verbose=True)
    acc = accuracy(case["plaintext"], dec)
    accs.append(acc)
    print(f"[{i+1}/{n}] len={case['length']} acc={acc:.2%} ({time.time()-t1:.1f}s)")
    print(f"  orig: {case['plaintext']}")
    print(f"  dec : {dec}")

avg = sum(accs) / len(accs)
sol = sum(1 for a in accs if a >= 0.9)
print(f"\n=== beam (wiki solver) on Wikipedia benchmark ===")
print(f"Average accuracy     : {avg:.2%}")
print(f"Cases >=90% (solved) : {sol}/{n}")
print(f"Total time           : {time.time()-t0:.1f}s")
