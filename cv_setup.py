"""Cross-validation setup: split corpus_wiki.txt into train/test by line,
build KN model + pattern map from TRAIN ONLY, and generate two benchmarks
(one drawn from train lines, one from test lines).

Running the same train-built solver against both benchmarks isolates the
effect of data leakage: the only difference is whether the test fragments
come from sentences the model was trained on.
"""
import io
import json
import os
import pickle
import random
import sys

if not isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from crack import decompose, _build_pattern_map
from train_kn_final import KneserNeyLM
from benchmark import build as build_benchmark

SRC          = "corpus_wiki.txt"
TRAIN_TXT    = "corpus_cv_train.txt"
TEST_TXT     = "corpus_cv_test.txt"
KN_CV        = "kn_model_cv.json"
PMAP_CV      = "full_pattern_map_cv.pkl"
BENCH_TRAIN  = "benchmark_cv_train.json"
BENCH_TEST   = "benchmark_cv_test.json"

SPLIT_SEED   = 42
TEST_FRAC    = 0.15
KN_TRAIN_CAP = 150_000   # lines fed to the KN model (memory/time bound)


def split_corpus():
    with open(SRC, "r", encoding="utf-8") as f:
        lines = [l.rstrip("\n") for l in f]
    rng = random.Random(SPLIT_SEED)
    idx = list(range(len(lines)))
    rng.shuffle(idx)
    n_test = int(len(idx) * TEST_FRAC)
    test_idx = set(idx[:n_test])
    train, test = [], []
    for i, l in enumerate(lines):
        (test if i in test_idx else train).append(l)
    with open(TRAIN_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(train) + "\n")
    with open(TEST_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(test) + "\n")
    print(f"Split: {len(train):,} train / {len(test):,} test lines")


def build_kn():
    if os.path.exists(KN_CV):
        print(f"{KN_CV} exists, skipping")
        return
    with open(TRAIN_TXT, "r", encoding="utf-8") as f:
        sents = [l.strip() for l in f if l.strip()][:KN_TRAIN_CAP]
    print(f"Training KN on {len(sents):,} train lines ...")
    kn = KneserNeyLM(order=7)
    kn.train(sents)
    kn.save(KN_CV)


def build_pmap():
    if os.path.exists(PMAP_CV):
        print(f"{PMAP_CV} exists, skipping")
        return
    print("Building pattern map from train ...")
    _build_pattern_map(TRAIN_TXT, cache_path=PMAP_CV)


def build_benchmarks():
    # Same generation params for both; only the source corpus differs.
    build_benchmark(corpus_path=TRAIN_TXT, out_path=BENCH_TRAIN,
                    num_tests=100, min_len=30, max_len=200, seed=7001)
    build_benchmark(corpus_path=TEST_TXT, out_path=BENCH_TEST,
                    num_tests=100, min_len=30, max_len=200, seed=7001)


if __name__ == "__main__":
    split_corpus()
    build_kn()
    build_pmap()
    build_benchmarks()
    print("CV setup complete.")
