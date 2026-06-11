import collections
import json
import math
import random
import sys
import io
from jamo import h2j, j2hcj

# Force UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def get_jamo_sequence(text):
    return j2hcj(h2j(text))

class KneserNeyScorer:
    def __init__(self, model_path):
        with open(model_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.order = data["order"]
        self.discount = data["discount"]
        self.vocab = set(data["vocab"])
        self.counts = [collections.Counter(c) for c in data["counts"]]
        self.unique_follows = [collections.Counter(c) for c in data["unique_follows"]]
        self.total_unigrams = sum(self.counts[1].values())

    def get_prob(self, ngram):
        n = len(ngram)
        epsilon = 1e-12
        if n == 1:
            return (self.counts[1].get(ngram, 0) + 0.1) / (self.total_unigrams + 0.1 * len(self.vocab))
        
        prefix = ngram[:-1]
        count = self.counts[n].get(ngram, 0)
        prefix_count = self.counts[n-1].get(prefix, 0)
        
        if prefix_count > 0:
            lambda_weight = (self.discount * self.unique_follows[n-1].get(prefix, 0)) / prefix_count
            prob = max(count - self.discount, 0) / prefix_count + lambda_weight * self.get_prob(ngram[1:])
            return max(prob, epsilon)
        else:
            return self.get_prob(ngram[1:])

    def score(self, text):
        total_log_prob = 0
        for i in range(len(text)):
            # Use max available context up to order
            context_len = min(i + 1, self.order)
            ngram = text[i - context_len + 1 : i + 1]
            prob = self.get_prob(ngram)
            total_log_prob += math.log10(prob)
        return total_log_prob

class Evaluator:
    def __init__(self, scorer):
        self.scorer = scorer
        self.target_jamos = [k for k in scorer.vocab if 0x3131 <= ord(k) <= 0x3163]

    def encrypt(self, text):
        jamos = get_jamo_sequence(text)
        unique_jamos = sorted(list(set(jamos.replace(" ", ""))))
        shuffled = list(self.target_jamos)
        random.shuffle(shuffled)
        key = {j: shuffled[i] for i, j in enumerate(unique_jamos)}
        ciphertext = "".join(key.get(c, c) for c in jamos)
        return ciphertext, key

    def fitness(self, text):
        score = self.scorer.score(text)
        # Add basic phonotactic penalty for stability
        def is_vowel(c): return 0x314F <= ord(c) <= 0x3163
        def is_consonant(c): return 0x3131 <= ord(c) <= 0x314E
        for i in range(len(text)):
            if i > 0 and is_consonant(text[i-1]) and is_consonant(text[i]):
                if i > 1 and is_consonant(text[i-2]): score -= 10
            if i > 0 and is_vowel(text[i-1]) and is_vowel(text[i]):
                score -= 5
        return score

    def solve(self, ciphertext, restarts=10, iterations=20000):
        symbols = sorted(list(set(ciphertext.replace(" ", ""))))
        best_overall_mapping = None
        best_overall_fitness = -float('inf')

        for r in range(restarts):
            # Start with unigram-frequency initialization for a better head start
            targets = random.sample(self.target_jamos, len(symbols))
            mapping = {s: targets[i] for i, s in enumerate(symbols)}
            
            current_text = "".join(mapping.get(c, c) for c in ciphertext)
            current_fitness = self.fitness(current_text)
            
            T = 5.0
            cooling_rate = 0.9998
            
            for i in range(iterations):
                s1, s2 = random.sample(symbols, 2)
                new_mapping = mapping.copy()
                new_mapping[s1], new_mapping[s2] = new_mapping[s2], new_mapping[s1]
                
                new_text = "".join(new_mapping.get(c, c) for c in ciphertext)
                new_fitness = self.fitness(new_text)
                
                delta = new_fitness - current_fitness
                if delta > 0 or (T > 0 and random.random() < math.exp(delta / T)):
                    mapping = new_mapping
                    current_fitness = new_fitness
                
                T *= cooling_rate
            
            print(f"Restart {r}, Fitness: {current_fitness:.4f}")
            if current_fitness > best_overall_fitness:
                best_overall_fitness = current_fitness
                best_overall_mapping = mapping.copy()

        return "".join(best_overall_mapping.get(c, c) for c in ciphertext)

if __name__ == "__main__":
    print("Loading 7-gram KN model...")
    scorer = KneserNeyScorer("kn_model.json")
    evaluator = Evaluator(scorer)
    
    # Test on a few short sentences
    test_sentences = [
        "세종대왕이 한글을 창제하셨다",
        "대한민국은 민주공화국입니다",
        "오늘 점심은 무엇을 먹을까요"
    ]
    
    print("\n--- Cracking Ciphers (40-60 characters) ---")
    for i, original in enumerate(test_sentences):
        orig_jamo = get_jamo_sequence(original)
        ciphertext, _ = evaluator.encrypt(original)
        
        print(f"\nTest {i+1} [Len={len(orig_jamo)}]: {ciphertext}")
        decrypted = evaluator.solve(ciphertext, iterations=15000)
        
        matches = sum(1 for a, b in zip(orig_jamo, decrypted) if a == b)
        acc = matches / len(orig_jamo)
        
        print(f"  Accuracy:  {acc:.2%}")
        print(f"  Decrypted: {decrypted}")
        print(f"  Original:  {orig_jamo}")
