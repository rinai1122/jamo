import random
import os
from jamo import h2j, j2hcj

def get_jamo_sequence(text):
    return j2hcj(h2j(text))

class DataGenerator:
    def __init__(self, corpus_path):
        self.corpus_path = corpus_path
        self.jamo_chars = [chr(i) for i in range(0x3131, 0x314f)] + [chr(i) for i in range(0x314f, 0x3164)]
        self.lines = []
        if os.path.exists(corpus_path):
            with open(corpus_path, "r", encoding="utf-8") as f:
                self.lines = [line.strip() for line in f if len(line.strip()) > 5]

    def generate_batch(self, batch_size=32, seq_len=40):
        batch_pairs = []
        for _ in range(batch_size):
            line = random.choice(self.lines)
            jamo_seq = get_jamo_sequence(line)
            
            if len(jamo_seq) < seq_len:
                plaintext = jamo_seq.ljust(seq_len)
            else:
                start = random.randint(0, len(jamo_seq) - seq_len)
                plaintext = jamo_seq[start:start+seq_len]
            
            # Create a random substitution key
            # We only shuffle the Jamo characters, keeping spaces as is
            unique_jamos_in_text = sorted(list(set(plaintext.replace(" ", ""))))
            target_pool = list(self.jamo_chars)
            random.shuffle(target_pool)
            
            key = {j: target_pool[i] for i, j in enumerate(unique_jamos_in_text)}
            ciphertext = "".join(key.get(c, c) for c in plaintext)
            
            batch_pairs.append((ciphertext, plaintext))
        return batch_pairs

if __name__ == "__main__":
    gen = DataGenerator("corpus.txt")
    batch = gen.generate_batch(5)
    for c, p in batch:
        print(f"P: {p}")
        print(f"C: {c}")
        print("-" * 10)
