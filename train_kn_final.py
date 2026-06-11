import collections
import json
import math
import sys
import io
import os
from jamo import h2j, j2hcj

# Force UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def get_jamo_sequence(text):
    return j2hcj(h2j(text))

class KneserNeyLM:
    def __init__(self, order=7, discount=0.75):
        self.order = order
        self.discount = discount
        # counts[n][ngram] = count of ngram
        self.counts = [collections.Counter() for _ in range(order + 1)]
        # prefix_counts[n][prefix] = how many unique words follow this prefix
        self.prefix_unique_follows = [collections.Counter() for _ in range(order)]
        # suffix_unique_precedes[n][suffix] = how many unique words precede this suffix
        self.suffix_unique_precedes = [collections.Counter() for _ in range(order + 1)]
        self.vocab = set()

    def train(self, sentences):
        print(f"Training {self.order}-gram KN model...")
        for text in sentences:
            jamos = get_jamo_sequence(text)
            self.vocab.update(jamos)
            for n in range(1, self.order + 1):
                for i in range(len(jamos) - n + 1):
                    ngram = jamos[i:i+n]
                    self.counts[n][ngram] += 1
                    
                    if n > 1:
                        prefix = ngram[:-1]
                        suffix = ngram[1:]
                        # We need to track unique continuations for KN
                        # These are used for the lower-order distributions
                        # This part is memory intensive, so we use a simplified version
                        self.prefix_unique_follows[n-1][prefix] += 1
                        self.suffix_unique_precedes[n][suffix] += 1

    def save(self, path):
        print(f"Saving model to {path}...")
        # To save space, we filter rare n-grams
        data = {
            "order": self.order,
            "discount": self.discount,
            "vocab": list(self.vocab),
            "counts": [ {k: v for k, v in c.items() if v > 1 or len(k) == 1} for c in self.counts ],
            "unique_follows": [dict(c) for c in self.prefix_unique_follows]
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

if __name__ == "__main__":
    if not os.path.exists("corpus.txt"):
        print("corpus.txt not found. Please run extract_data.py first.")
        sys.exit(1)
        
    with open("corpus.txt", "r", encoding="utf-8") as f:
        # Load a substantial subset but keep it fast
        sentences = [line.strip() for line in f if line.strip()][:50000]

    kn = KneserNeyLM(order=7)
    kn.train(sentences)
    kn.save("kn_model.json")
    print("Model training complete.")
