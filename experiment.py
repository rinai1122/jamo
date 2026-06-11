import collections
import math
import random
import sys
import io
from jamo import h2j, j2hcj

# Force UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def get_jamo_sequence(text):
    return j2hcj(h2j(text))

class JamoLM:
    def __init__(self, order=3):
        self.order = order
        self.counts = [collections.Counter() for _ in range(order + 1)]
        self.vocab = set()

    def train(self, sentences):
        for text in sentences:
            jamos = get_jamo_sequence(text)
            self.vocab.update(jamos)
            for n in range(1, self.order + 1):
                for i in range(len(jamos) - n + 1):
                    ngram = jamos[i:i+n]
                    self.counts[n][ngram] += 1

    def score(self, text):
        score = 0
        for n in range(1, self.order + 1):
            weight = n
            for i in range(len(text) - n + 1):
                ngram = text[i:i+n]
                count = self.counts[n].get(ngram, 0)
                total = sum(self.counts[n].values())
                prob = (count + 0.01) / (total + 0.01 * len(self.vocab))
                score += weight * math.log10(prob)
        return score

class Evaluator:
    def __init__(self, lm):
        self.lm = lm
        self.target_jamos = [k for k in lm.vocab if 0x3131 <= ord(k) <= 0x3163]

    def encrypt(self, text):
        jamos = get_jamo_sequence(text)
        unique_jamos = sorted(list(set(jamos.replace(" ", ""))))
        shuffled = list(self.target_jamos)
        random.shuffle(shuffled)
        key = {j: shuffled[i] for i, j in enumerate(unique_jamos)}
        ciphertext = "".join(key.get(c, c) for c in jamos)
        return ciphertext, key

    def fitness(self, text):
        score = self.lm.score(text)
        def is_vowel(c): return 0x314F <= ord(c) <= 0x3163
        def is_consonant(c): return 0x3131 <= ord(c) <= 0x314E
        for i in range(len(text)):
            if i > 0 and is_consonant(text[i-1]) and is_consonant(text[i]):
                if i > 1 and is_consonant(text[i-2]): score -= 200
            if i > 0 and is_vowel(text[i-1]) and is_vowel(text[i]):
                score -= 100
        return score

    def solve(self, ciphertext, iterations=10000):
        symbols = sorted(list(set(ciphertext.replace(" ", ""))))
        targets = random.sample(self.target_jamos, len(symbols))
        mapping = {s: targets[i] for i, s in enumerate(symbols)}
        
        current_text = "".join(mapping.get(c, c) for c in ciphertext)
        current_fitness = self.fitness(current_text)
        
        best_mapping = mapping.copy()
        best_fitness = current_fitness
        
        T = 10.0
        for i in range(iterations):
            s1, s2 = random.sample(symbols, 2)
            new_mapping = mapping.copy()
            new_mapping[s1], new_mapping[s2] = new_mapping[s2], new_mapping[s1]
            
            new_text = "".join(new_mapping.get(c, c) for c in ciphertext)
            new_fitness = self.fitness(new_text)
            
            if new_fitness > current_fitness or random.random() < math.exp((new_fitness - current_fitness) / T):
                mapping = new_mapping
                current_fitness = new_fitness
                if current_fitness > best_fitness:
                    best_fitness = current_fitness
                    best_mapping = mapping.copy()
            T *= 0.9997
        
        return "".join(best_mapping.get(c, c) for c in ciphertext)

if __name__ == "__main__":
    print("Loading corpus.txt...")
    with open("corpus.txt", "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]
    
    random.shuffle(texts)
    split = int(len(texts) * 0.9)
    train_texts = texts[:split][:5000] # Reduced for speed
    test_texts = texts[split:][:3]      # Reduced for speed
    
    print(f"Training on {len(train_texts)} sentences...")
    lm = JamoLM(order=3)
    lm.train(train_texts)
    
    evaluator = Evaluator(lm)
    
    print("\n--- Validation ---")
    for i, original in enumerate(test_texts):
        orig_jamo = get_jamo_sequence(original)
        ciphertext, _ = evaluator.encrypt(original)
        
        print(f"Test {i+1} Length: {len(orig_jamo)}")
        decrypted = evaluator.solve(ciphertext, iterations=10000) # Reduced for speed
        
        matches = sum(1 for a, b in zip(orig_jamo, decrypted) if a == b)
        acc = matches / len(orig_jamo)
        
        print(f"  Acc:  {acc:.2%}")
        print(f"  Orig: {orig_jamo}")
        print(f"  Decr: {decrypted}")
    print("\nDone.")
