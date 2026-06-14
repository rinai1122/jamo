import collections
import json
import math
import sys
import io
from jamo import h2j, j2hcj
from Korpora import Korpora

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def get_jamo_sequence(text):
    return j2hcj(h2j(text))

class KneserNeyModel:
    def __init__(self, order=3, discount=0.75):
        self.order = order
        self.discount = discount
        self.counts = [collections.Counter() for _ in range(order + 1)]
        self.vocab = set()

    def train_on_corpus(self, corpus_iterator, max_chars=100_000_000):
        total_chars = 0
        print(f"Starting training on up to {max_chars} characters...")
        
        for text in corpus_iterator:
            if total_chars >= max_chars:
                break
            
            jamos = get_jamo_sequence(text)
            total_chars += len(jamos)
            self.vocab.update(jamos)
            
            # Count N-grams
            for n in range(1, self.order + 1):
                for i in range(len(jamos) - n + 1):
                    ngram = jamos[i:i+n]
                    self.counts[n][ngram] += 1
            
            if total_chars % 1_000_000 < 5000: # Progress report approx every 1M chars
                print(f"Processed {total_chars} characters...")

    def save_model(self, file_path):
        # We only save the necessary stats to keep the JSON manageable
        # Using a more compact format or just the top N-grams if needed
        # For now, let's try to save all but filter very rare ones to save space
        filtered_counts = []
        for n in range(1, self.order + 1):
            # Keep n-grams that appear more than once (except for unigrams)
            if n > 1:
                filtered = {k: v for k, v in self.counts[n].items() if v > 1}
            else:
                filtered = dict(self.counts[n])
            filtered_counts.append(filtered)

        model_data = {
            "order": self.order,
            "discount": self.discount,
            "counts": filtered_counts,
            "vocab": list(self.vocab)
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(model_data, f, ensure_ascii=False)

if __name__ == "__main__":
    print("Loading kowikitext...")
    corpus = Korpora.load("kowikitext")
    
    kn = KneserNeyModel(order=3)
    # Use 50MB first to see performance, can increase to 100MB
    kn.train_on_corpus(corpus.train.get_all_texts(), max_chars=20_000_000)
    
    print("Saving model...")
    kn.save_model("kn_model.json")
    print("Done.")
