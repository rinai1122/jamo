"""Download Korean Wikipedia and build corpus_wiki.txt + benchmark_wiki.json.

Steps:
  1. Load Korean Wikipedia via HuggingFace datasets (streaming to avoid OOM).
  2. Write plain-text sentences to corpus_wiki.txt (up to MAX_LINES lines).
  3. Generate benchmark_wiki.json using benchmark.build() pointed at that file.
"""
import io
import sys

if not isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

MAX_LINES = 300_000   # ~20 MB of Korean text; more than enough for a pattern map
CORPUS_PATH = "corpus_wiki.txt"
BENCH_PATH  = "benchmark_wiki.json"

def build_corpus():
    from datasets import load_dataset
    print("Streaming Korean Wikipedia …")
    ds = load_dataset("wikimedia/wikipedia", "20231101.ko", split="train", streaming=True)
    written = 0
    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        for article in ds:
            for line in article["text"].splitlines():
                line = line.strip()
                if len(line) > 15:          # skip headings / stub lines
                    f.write(line + "\n")
                    written += 1
                    if written >= MAX_LINES:
                        break
            if written >= MAX_LINES:
                break
    print(f"Wrote {written:,} lines to {CORPUS_PATH}")

def build_benchmark():
    from benchmark import build
    build(corpus_path=CORPUS_PATH, out_path=BENCH_PATH,
          num_tests=100, min_len=30, max_len=200, seed=9999)

if __name__ == "__main__":
    build_corpus()
    build_benchmark()
