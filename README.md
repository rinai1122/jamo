# Korean jamo substitution-cipher cracker — no-space study

Cracks monoalphabetic substitution ciphers over Korean **jamo** (the text is
decomposed into 자모 first, e.g. `한글` → `ㅎㅏㄴㄱㅡㄹ`). This repository is
centered on the hardest variant: **no-space (띄어쓰기 없음)** ciphertext, where
the word boundaries that normally anchor the attack are gone.

## Headline result

On the full **100-case held-out test set** (spaces stripped from both cipher and
plaintext), the shipped `NoSpaceSolver` improves on the original single-pass
solver by **+4.3 pts average accuracy and +7 solved cases**:

| no-space solver | avg acc | solved ≥90% |
|---|---|---|
| baseline (single deep union SA pass) | 68.12% | 47/100 |
| **shipped (deep+broad SA ensemble, all lengths)** | **72.38%** | **54/100** |

The gains land on long fragments that the baseline search got stuck on (60–94
jamo +9.9 pts, 95+ jamo +11.2 pts; e.g. a 105-jamo case 3.8%→99%, a 132-jamo
case 14%→99%). The ensemble redesign first reached 73.1% / 57 with a multi-seed
beam and a separate base-SA pass; a per-stage ablation then showed both
removable, and trimming them (simpler solver) settled at the 72.4% / 54 above.
Full analysis, per-bucket tables and the design rationale are in
[`SUMMARY_CV.md`](SUMMARY_CV.md) (§4 redesign, §5 ablation, §6 trim).

## Core files

| file | role |
|---|---|
| `crack.py` | shared core: `KneserNeyScorer`, phonotactic `structure_violations`, `decompose`, word-pattern index, and the spaced `BeamSolver` |
| `nospace_solver.py` | **the no-space solver** (`NoSpaceSolver`): LM-beam seeds → deep+broad SA ensemble, best-by-fitness |
| `build_wiki_corpus.py` | downloads Korean Wikipedia → `corpus_wiki.txt` |
| `train_kn_final.py` | Kneser-Ney n-gram LM trainer (used by `cv_setup.py`) |
| `benchmark.py` | generates cipher/plaintext benchmark fragments (used by `cv_setup.py`) |
| `cv_setup.py` | train/test split + builds KN model, pattern map and both benchmarks |
| `ns_eval.py` | parallel evaluation of `NoSpaceSolver` over the 100-case benchmark |
| `ns_compare_results.py` | diffs two `ns_eval` runs (before/after, by length bucket) |
| `ns_objsweep.py` | offline objective tuning against saved outputs (no re-solving) |
| `benchmark_cv_{train,test}.json` | the committed 100-case benchmarks |
| `ns_result_{baseline,v4}.json` | the two runs behind the headline table |

Historical / superseded scripts (earlier phases, the spaced-solver
leakage+ablation study, the no-space per-stage ablation `ns_ablation.py`, one-off
experiments and diagnostics) are parked under [`archive/`](archive/).

## Replicate the no-space result

Requires Python 3.12. The large artifacts (`corpus_wiki.txt`, `kn_model_cv.json`,
`full_pattern_map_cv.pkl`) are git-ignored and rebuilt by the steps below.

```bash
# 1. Build the corpus (needs the `datasets` package; one-time, downloads Wikipedia)
pip install datasets
python build_wiki_corpus.py            # -> corpus_wiki.txt

# 2. Train/test split + KN model + pattern map + benchmarks (from TRAIN only)
python cv_setup.py                     # -> kn_model_cv.json, full_pattern_map_cv.pkl,
                                       #    benchmark_cv_{train,test}.json
                                       # (benchmarks are also committed, so this
                                       #  reproduces them identically)

# 3. Evaluate the shipped solver on all 100 held-out cases
python ns_eval.py 0 6 v4               # args: N(0=all) workers tag
                                       # -> ns_result_v4.json

# 4. Diff against the committed baseline (the original single-deep-pass solver)
python ns_compare_results.py ns_result_baseline.json ns_result_v4.json
```

`ns_result_baseline.json` is the committed reference run of the original
single-deep-pass solver; it predates the ensemble redesign and the knobs it was
generated with (`beam_seeds`, `union_seed_all`) no longer exist on
`NoSpaceSolver`, so it is kept as a frozen artifact rather than regenerated.

`ns_eval.py` prints a per-bucket accuracy table plus a **search-vs-objective
failure split** for every unsolved case, and saves per-case outputs to
`ns_result_<tag>.json`. The solver config is overridable via the
`SOLVER_KWARGS` env var (JSON); with no override, `NoSpaceSolver`'s defaults are
the shipped configuration.

### Notes

- **Workers / memory.** Each worker loads the KN model + pattern map (~700 MB).
  On a 16 GB machine use ~6 workers. `KneserNeyScorer` caps its n-gram cache so
  long parallel runs stay memory-bounded (uncapped it OOMs the box).
- **Runtime.** A full 100-case run of the shipped solver is search-heavy
  (~1.8 h on 6 cores; the committed `ns_result_v4.json` run was 6469 s). Pass a
  small `N` to spot-check quickly.
