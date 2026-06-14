# archive/ — historical & superseded scripts

These files are kept for provenance but are **not** part of the no-space
replication path (see the top-level `README.md`). They import the core modules
(`crack`, `nospace_solver`) by name, so run them from the **repo root**, e.g.
`python archive/run_experiments.py` may need `PYTHONPATH=.` — or copy the file
up one level.

- `SUMMARY_PHASE_1.md`, `SUMMARY_PHASE_2.md`, `SUMMARY_PHASE_3.md` — earlier
  project-phase writeups (pre-CV study).
- `run_experiments.py`, `run_experiments2.py`, `cv_eval.py` — the spaced-solver
  cross-validation leakage + ablation study behind `SUMMARY_CV.md` §1–2.
- `run_nospace.py`, `bench_nospace.py` — first no-space evaluators, superseded
  by the parallel `ns_eval.py`.
- `exp_search.py`, `exp_validate.py`, `exp_ws.py` — one-off no-space tuning
  experiments (search budget, ws weight).
- `diag_nospace.py`, `diag_wordseg.py` — per-case diagnostics that fed the
  no-space design.
- `train_kn.py` — earlier KN trainer, superseded by `train_kn_final.py`.
- `build_freq.py`, `freq_table.json` — jamo frequency table from an early phase.
- `extract_data.py`, `validate_corpus.py`, `verify_limit.py`, `jamo_test.py` —
  early data-prep and sanity utilities.
- `benchmark_set.json`, `benchmark_wiki.json` — earlier benchmark sets.
- `ns_result_val.json`, `ns_result_v2.json` — intermediate no-space runs (the
  headline comparison uses `ns_result_baseline.json` vs `ns_result_v3.json` at
  the repo root).
