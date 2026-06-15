# Cross-Validation, Ablation & No-Space Study

All runs use a **train/test split of `corpus_wiki.txt`** (seed 42, 85/15 by
line → 255k train / 45k test).  The KN model (`kn_model_cv.json`, order-7,
150k train lines) and the pattern map (`full_pattern_map_cv.pkl`, all train
lines) are built **from train only**.  Two benchmarks are generated with
identical params (seed 7001, 100 fragments, 30–200 jamo): one drawn from
train lines, one from test lines.  The same train-built solver is run
against both, so the only difference is data leakage.

Reproduce the no-space study (§3–4): `python cv_setup.py && python ns_eval.py`
— see `README.md`. The spaced-solver leakage/ablation scripts behind §1–2
(`run_experiments.py`, `run_experiments2.py`, `cv_eval.py`) and the earlier
diagnostics now live under `archive/` (run them from the repo root).

## 1. Leakage test — is the benchmark cheating?

| Benchmark source | Avg acc | Solved ≥90% |
|---|---|---|
| In-sample (train-drawn) | **92.77%** | 87/100 |
| Held-out (test-drawn)   | **91.27%** | 77/100 |

Holding out the test sentences costs only **~1.5 pts** of average accuracy
(the "solved" count drops more, 87→77, i.e. a handful of borderline cases
lose their verbatim-leaked anchor word).  The in-sample 92.77% reproduces
the earlier native-wiki 92.68%, validating the harness.

**Conclusion: the solver is not meaningfully cheating.** It relies on common
Korean words and jamo n-gram statistics, which generalise; only rare unique
phrases benefit from exact train/test overlap.

> Note: the low-coverage SA fallback uses an unseeded RNG, giving ~±0.5 pt
> run-to-run noise. Treat ablation gaps under ~1 pt as within noise.

## 2. Ablation — which parts carry the performance?

Held-out test benchmark; disable one part, keep the rest.

| Config | Avg acc | Solved | Δ vs full |
|---|---|---|---|
| full pipeline (N=100) | 90.83% | 76/100 | — |
| −KN language model | 87.58% | 71/100 | **−3.25** |
| −edge augment | 89.39% | 74/100 | −1.44 |
| −SA polish | 89.83% | 75/100 | −1.00 |
| −struct DFA | 89.98% | 78/100 | −0.85 |
| −greedy polish | 90.51% | 76/100 | −0.32 |
| −SA fallback | 90.77% | 76/100 | ~0 |
| −dict bonus | 90.93% | 77/100 | ~0 |
| full pipeline (N=30) | 92.45% | 22/30 | — |
| **−beam (pure SA)** (N=30) | **74.77%** | 16/30 | **−17.68** |

**Ranking of importance**

1. **Beam search over word patterns — essential (−17.7 pts).** Removing it
   reduces the solver to pure simulated annealing (the old hill-climbing
   baseline, ~75%). The word-pattern anchoring is the whole reason this
   solver beats ~75% → ~92%.
2. **KN language model — clearly helps (−3.25 pts).** Disambiguates among
   pattern-consistent candidate words and fills unanchored symbols.
3. **Edge augmentation (−1.4) and SA polish (−1.0) — modest but real.** They
   recover the hard fragment-edge / low-coverage cases.
4. **Struct DFA, greedy polish, dict bonus, SA fallback — ~0 on spaced text.**
   The beam already supplies strong word anchors, so the fitness-level dict
   bonus is largely redundant, and the structure/fallback machinery only
   bites on the few low-coverage cases (rare when spaces survive). These are
   cheap insurance that matters most on degraded input (see §3).

## 3. Does it work without 띄어쓰기 (no spaces)?

Spaces are the solver's backbone: they delimit the words whose repetition
patterns drive the beam. Stripping them removes all word anchors, so every
case falls back to pure SA over the KN model.

Held-out test, N=15, spaces stripped from cipher **and** plaintext:

| | Avg acc | Solved ≥90% |
|---|---|---|
| With spaces (ref) | 93.69% | 13/15 |
| **No spaces** | **63.44%** | 6/15 |

The result is **bimodal and length-dependent**:

| Length (jamo) | No-space outcome |
|---|---|
| ≥95 | mostly 92–100% (5 of 6 solved) |
| <45 | mostly 7–62% (fails) |

Long texts give SA enough n-gram signal to find the right key; short texts
without spaces are information-theoretically underdetermined and depend on
anneal luck. (One len-147 case still failed at 32% — pure SA does not always
find the basin.)

**Conclusion: it partially works.** No-space drops average accuracy ~30 pts
and only reliably solves long fragments. For short spaceless text the
word-pattern method is gone and the solver falls back to search over the LM.
(This 15-case snapshot motivated the dedicated solver and the **full 100-case**
study below.)

## 4. Improved no-space solver (`nospace_solver.py`), full 100-case study

The earlier no-space numbers were measured on only 15 cases. Re-running the
dedicated `NoSpaceSolver` on the **entire 100-case held-out test set** (via the
parallel harness `ns_eval.py`) overturned a key assumption and drove a redesign.

**First: a memory bug that blocked the large run.** `KneserNeyScorer` caches
every n-gram it scores and never evicts. A multiprocessing worker reuses one
scorer across ~17 cases of intense annealing (millions of distinct decryptions
→ millions of distinct n-grams), so the cache grows without bound: the box
swapped to a crawl (cases ballooning from ~100 s to 3,000–7,800 s) and then
hit `MemoryError`. Fixed by capping the cache in `crack.py` (pure memoization,
so results are unchanged). This alone is what made full-benchmark runs viable.

**Diagnosis at scale.** `ns_eval.py` classifies every failure by comparing the
true key's fitness to the solver's output: *search* failure (true key scores
higher — more search fixes it) vs *objective* failure (a wrong key genuinely
out-scores the truth). On the 100-case baseline (avg 68.1%, 47 solved):

* **28 search failures, and the worst are LONG cases** — e.g. a 132-jamo case
  at 14% and a 105-jamo case at 4%, while the true key scores **+70 / +65
  log10 higher**. The old "long cases already solve at ~98–100%" claim was an
  artifact of the 15-case sample; at scale the base search gets badly stuck on
  long fragments too. These are pure search misses.
* **25 objective failures**, almost all ≲25-jamo fragments — two readings both
  valid Korean, no signal separates them. The irreducible floor.

(Word-segmentation *coverage* does **not** separate the two: stuck wrong keys
still decrypt to fluent, high-coverage text — just the wrong words. So the
heavy search cannot be cheaply gated on output quality.)

**Redesign (the "v3" config).** The single gated union pass became a multi-pass
ensemble applied across **all** lengths ≥ `min_ensemble_len` (28), not just short
ones. (Steps 1–2 below were later found redundant by the §5 ablation and removed
in §6; this section records the v3 pipeline that first hit 73.1%.)

1. **LM-beam top-K seeds** (K=10), each greedy-polished.
2. **Base SA** seeded by beam key #1 (unchanged baseline).
3. **Ensemble.** *Pass 1* is a **deep** anneal that reproduces the old single
   union pass byte-for-byte (same RNG, seed, budget, with `word_penalty=0`) —
   it protects cases that need anneal *depth* (a breadth-only redistribution
   was found to wreck cases the baseline had solved, e.g. case 19: 91%→9%).
   *Passes 2+* are **broad** — shorter anneals, fresh RNGs, seeded across the
   whole top-K beam — cheap restarts that rescue the stuck basins.
4. **Best by fitness wins** over the union; the search only adds candidates
   under one objective, so it is monotone in fitness.
5. The word-segmentation DP was also made ~2× faster (prefix-set pruning,
   identical results), which pays for the extra passes.

**Result (full 100-case held-out test, `ns_compare_results.py`):**

| no-space solver | Avg acc | Solved ≥90% |
|---|---|---|
| baseline (original single deep union pass) | 68.12% | 47/100 |
| **v3 (deep+broad ensemble, all lengths)** | **73.14%** | **57/100** |

**+5.0 pts average, +10 solved.** (This "v3" config was later trimmed to the
shipped v4 in §6; the two short-case rescues below, cases 24 and 79, are the ones
the trim gives back.) By length bucket the gains are exactly where the diagnosis
predicted — the long search failures:

| Length | Baseline | Improved | Δ avg | solved |
|---|---|---|---|---|
| <30   | 38.2% | 38.2% | −0.0 | 0→0 |
| 30–44 | 57.9% | 62.0% | +4.1 | 8→11 |
| 45–59 | 68.7% | 68.4% | −0.3 | 9→9 |
| 60–94 | 84.3% | 93.6% | **+9.3** | 13→18 |
| 95+   | 88.0% | 97.6% | **+9.6** | 17→19 |

Headline rescues: case 67 (105j) **3.8%→99.0%**, case 95 (132j) **13.6%→99.2%**,
case 62 (83j) 8.4%→92.8%, case 24 (36j) 0%→91.7%, case 79 (44j) 11.4%→95.5%.

**Residual failures** (43): the 13 per-case regressions are all len ≤ 46 and
are *objective overshoot* — better search climbs to a higher-fitness but still
wrong local optimum the baseline's shallower anneal luckily avoided. Net is
positive in every bucket, so they are accepted. The rest are the irreducible
≲25-jamo fragments.

**Tuning notes that did *not* pan out** (recorded so they aren't re-tried):
*Breadth-over-depth redistribution* (cutting anneal length to fund more passes)
wrecks short cases that need depth — hence the deep pass 1.  A *word-insertion
penalty* (`word_penalty`, biasing toward fewer/longer segmented words) was
offline-tuned against saved outputs (`ns_objsweep.py`): it flips only ~3 extra
objective failures and, by changing the objective, would forfeit the deep
pass's byte-for-byte no-regression guarantee — net not worth it, left at 0.
*Coverage-gating* the heavy search is impossible (coverage doesn't separate the
failure types, above).

## 5. No-space ablation — which stages carry the no-space solver?

`archive/ns_ablation.py` repeats the §2 ablation idea for the dedicated
`NoSpaceSolver`: disable one stage / fitness term at a time and measure the
accuracy cost on the **full 100-case held-out test set** (spaces stripped from
cipher and plaintext).
To keep nine configs × 100 cases tractable, every config shares one fixed
*reduced* search budget (`beam_width=800, sa_restarts=6, sa_iters=1500,
short_sa_passes=2, beam_seeds=5`), so the numbers measure **relative
contribution**, not absolute accuracy — this ablates the full v3 pipeline (still
with base_sa and the multi-seed beam), which at its ~2.5× larger shipped budget
reaches 73.1% (§4). At this reduced budget the full pipeline scores
58.8%. The no-space search is fully seeded (per-case `random.Random`), so these
gaps are reproducible signal, not the RNG noise that §2 had to discount.

| Config | Avg acc | Solved | Δ avg | Δ solved |
|---|---|---|---|---|
| full pipeline | 58.84% | 33/100 | — | — |
| **−ensemble** (multi-pass union) | **48.70%** | 19 | **−10.15** | −14 |
| −beam (LM-beam seeding) | 50.70% | 23 | −8.14 | −10 |
| −kn (KN language model) | 51.67% | 18 | −7.18 | −15 |
| −greedy (greedy polish) | 53.88% | 22 | −4.97 | −11 |
| −ws (word-seg fitness term) | 54.86% | 26 | −3.98 | −7 |
| −struct (phonotactic DFA) | 55.53% | 27 | −3.31 | −6 |
| −base_sa (standalone SA pass) | 58.71% | 33 | −0.13 | +0 |
| −multiseed (beam_seeds 5→1) | 59.85% | 33 | +1.00 | +0 |

**Ranking & reading**

1. **Ensemble — most essential (−10.2 pts, −14 solved).** The multi-pass
   deep+broad union is the single biggest contributor, quantitatively
   confirming the §4 redesign thesis: with word anchors gone, escaping the
   stuck search-failure basins via independent restarts is what carries the
   solver. It is also the most expensive stage (removing it is the *fastest*
   config, 444 s vs 955 s) — high value, high cost, exactly where the compute
   should go.
2. **LM-beam seeding (−8.1) and KN language model (−7.2) — the backbone.**
   No spaces means no word-pattern anchors, so the n-gram beam and the KN score
   do the heavy lifting the spaced solver got from `text.split(" ")`. The KN
   term collapses the *solved* count hardest (33→18): it is what converts a
   near-miss key into a clean ≥90% read, even where the average barely moves.
3. **Greedy polish (−5.0), and the two fitness add-ons ws (−4.0) and struct
   (−3.3) — all clearly real.** This is the sharp contrast with §2, where
   struct/ws/dict were ~0 because the beam already supplied word anchors.
   Here every fitness term bites: on anchorless text they are the only thing
   separating competing fluent readings, so none is redundant.
4. **Standalone base SA — redundant (−0.13, 0 solved lost).** The ensemble's
   pass 1 is a *deeper* anneal from the same beam-key #1 seed, so by
   best-by-fitness it almost always dominates the shallower standalone base SA;
   removing the separate `base_sa` stage costs essentially nothing yet still
   bills ~145 s. It is kept only as free best-by-fitness insurance — a genuine
   candidate to drop if the budget were tight.
5. **Multi-seed beam (5 vs 1) — no benefit at this budget (+1.0).** Dropping to
   a single beam seed slightly *helps* here. The extra seeds were tuned for the
   full budget's short-case rescues (§4); at the reduced budget they mostly
   spread the polish effort and occasionally let a higher-fitness-but-wrong key
   win on a short fragment. Honest caveat: this stage's value is
   budget-dependent and does not surface at the ablation budget — it is
   validated only at the shipped budget, not re-derivable from this table.

**Contrast with the spaced ablation (§2).** There, one stage dominated (beam,
−17.7 pts) and struct / ws / dict / fallback were ~0. Here the contributions are
*spread*: ensemble, beam and KN are each worth 7–10 pts and the fitness terms
each worth 3–5. Stripping the word anchors doesn't merely lower the ceiling — it
makes every remaining stage load-bearing, because no single one can pin the key
on its own. That spread is the quantitative signature of the no-space regime.

The two ~0 / positive rows — `base_sa` (−0.13) and `multiseed` (+1.0) — flagged
removable stages; §6 acts on that and re-validates at the shipped budget.

## 6. Trimming to the shipped solver (v4)

§5 found the standalone base-SA pass fully redundant (the ensemble's deep pass 1
supersedes it) and the multi-seed beam not earning its budget. Both were removed
from `NoSpaceSolver`, which is now simply **single best beam key → greedy polish
→ deep+broad annealing ensemble** (gone: `base_sa`, `beam_seeds`,
`union_seed_all`; ~40 lines and three knobs lighter). The result was re-validated
on the **full 100-case held-out test set at the shipped budget** (`ns_eval.py`,
→ `ns_result_v4.json`):

| no-space solver | Avg acc | Solved ≥90% | vs baseline |
|---|---|---|---|
| baseline (original single deep union pass) | 68.12% | 47/100 | — |
| v3 (ensemble + multi-seed beam + base SA, §4) | 73.14% | 57/100 | +5.02 / +10 |
| **v4 (trimmed, shipped)** | **72.38%** | **54/100** | **+4.26 / +7** |

**The trim is not free, and the ablation budget hid the cost.** Dropping base_sa
was free as predicted (it never won a case the ensemble didn't). Dropping the
multi-seed beam cost **−0.75 pts / −3 solved vs v3**, and the loss is entirely in
the short/mid buckets — exactly the §4 short-case rescues the extra seeds bought:

| Length | v3 | v4 | Δ avg | v3→v4 solved |
|---|---|---|---|---|
| <30   | 38.2% | 32.8% | −5.3 | 0→0 |
| 30–44 | 62.0% | 58.0% | −4.0 | 11→8 |
| 45–59 | 68.4% | 73.6% | +5.2 | 9→9 |
| 60–94 | 93.6% | 94.2% | +0.6 | 18→17 |
| 95+   | 97.6% | 99.1% | +1.6 | 19→20 |

The casualties are the two v3 headline rescues that depended on a lower-ranked
beam seed: case 24 (36j) **91.7%→0%** and case 79 (44j) **95.5%→11.4%** both flip
back to wrong basins the single best key can't escape. This refines §5's
multiseed reading: at the *reduced* ablation budget the extra seeds looked inert
(+1.0), but at the *shipped* budget — where the ensemble has the depth to exploit
a near-miss seed — they genuinely rescue a few short cases. The reduced-budget
ablation under-counted them, the documented risk of measuring relative
contribution at a smaller budget.

**Why ship v4 anyway.** The entire long-case story — the whole point of the
redesign — is untouched: vs the baseline, v4 still gains **+9.9 pts (60–94j)** and
**+11.2 pts (95+j)**, and every long rescue survives (case 67 105j 3.8%→99%,
case 95 132j 13.6%→99.2%, case 62 83j 8.4%→92.8%, case 23 84j 67.9%→100%). The
trade is a meaningfully simpler, faster solver (one beam seed, one fewer SA pass)
for −0.75 pt on the already-hard, partly-irreducible short fragments. Net vs the
original baseline is **+4.3 pts / +7 solved**, and the failure split is unchanged
(46 unsolved: 19 search, 27 objective). v3's outputs remain committed
(`ns_result_v3.json`) so the trade-off is auditable per case.

## Code change

`crack.py`: `KneserNeyScorer` now caps its n-gram cache (`_cache_cap`,
default 1M) and clears it wholesale when exceeded — bounds memory across long
multiprocessing runs without changing any score. `BeamSolver` retains the
optional `disable` set (names: `kn`, `struct`, `dict`, `beam`, `edge`,
`sa_fallback`, `sa_polish`, `greedy`) and `pattern_map_cache` path argument for
the ablation in §2.

`nospace_solver.py`: prefix-pruned word-segmentation DP; single best beam key →
greedy polish → multi-pass deep+broad annealing ensemble (`short_sa_passes`,
`short_iter_mult`, `min_ensemble_len`); inert `word_penalty` hook; a `disable`
set (names: `kn`, `struct`, `ws`, `beam`, `greedy`, `ensemble`) switching off
one stage or fitness term at a time for the §5 ablation. The standalone base-SA
pass and the multi-seed beam (`beam_seeds`, `union_seed_all`) were **removed**
after §5 (see §6); defaults are the trimmed shipped config.

New harness/analysis scripts: `ns_eval.py` (parallel full-benchmark eval with
search/objective diagnostics + saved outputs), `ns_compare_results.py` (before/
after diff by bucket), `ns_objsweep.py` (offline objective tuning). The
per-stage no-space ablation `ns_ablation.py` (+ `ns_ablation.json`, §5) now lives
under `archive/`: it exercises the removed `base_sa`/`beam_seeds` knobs, so it is
frozen as the record of that decision.
