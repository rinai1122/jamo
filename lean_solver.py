import collections
import json
import math
import random
import sys
import io
import os
from jamo import h2j, j2hcj

# Force UTF-8
if not isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def get_jamo_sequence(text):
    return j2hcj(h2j(text))

def get_word_pattern(word):
    pattern = []
    seen = {}
    curr = 0
    for char in word:
        if char not in seen:
            seen[char] = str(curr)
            curr += 1
        pattern.append(seen[char])
    return ".".join(pattern)

class DictionaryAnchor:
    def __init__(self, corpus_path):
        self.pattern_map = collections.defaultdict(list)
        print("Building Lean Pattern Map...")
        if os.path.exists(corpus_path):
            with open(corpus_path, "r", encoding="utf-8") as f:
                words = []
                for line in f:
                    words.extend(line.strip().split())
                # Top 10k words is often enough for common sentences
                word_counts = collections.Counter(words)
                common_words = [w for w, _ in word_counts.most_common(10000)]
                for word in common_words:
                    jamo_word = get_jamo_sequence(word)
                    if len(jamo_word) > 2:
                        pattern = get_word_pattern(jamo_word)
                        self.pattern_map[pattern].append(jamo_word)
        print(f"Lean Pattern map built.")

    def get_matches(self, cipher_word):
        pattern = get_word_pattern(cipher_word)
        return self.pattern_map.get(pattern, [])

class AdvancedSolver:
    def __init__(self, kn_model_path, corpus_path):
        from final_solver import KneserNeyScorer
        self.scorer = KneserNeyScorer(kn_model_path)
        self.order = 4 # Limit context for speed
        self.dict_anchor = DictionaryAnchor(corpus_path)
        self.target_jamos = [k for k in self.scorer.vocab if 0x3131 <= ord(k) <= 0x3163]

    def fitness(self, text):
        # Optimized fitness using the existing scorer's logic but with depth limit
        score = 0
        for i in range(len(text)):
            context_len = min(i + 1, self.order)
            ngram = text[i - context_len + 1 : i + 1]
            score += math.log10(self.scorer.get_prob(ngram))
        
        words = text.split()
        for w in words:
            if len(w) > 2:
                pattern = get_word_pattern(w)
                if w in self.dict_anchor.pattern_map.get(pattern, []):
                    score += 1000
        return score

    def solve(self, ciphertext, restarts=50, iterations=3000):
        symbols = sorted(list(set(ciphertext.replace(" ", ""))))
        best_overall_mapping = None
        best_overall_fitness = -float('inf')

        cipher_words = ciphertext.split()
        
        for r in range(restarts):
            if r % 10 == 0: print(f"Restart {r}...")
            
            mapping = {}
            available_targets = list(self.target_jamos)
            random.shuffle(available_targets)
            
            # Seeding with patterns
            for cw in cipher_words:
                if len(cw) > 3 and random.random() < 0.7:
                    matches = self.dict_anchor.get_matches(cw)
                    if matches:
                        match = random.choice(matches)
                        temp_map = mapping.copy()
                        possible = True
                        for c_char, p_char in zip(cw, match):
                            if c_char in temp_map and temp_map[c_char] != p_char: possible = False; break
                            if p_char in temp_map.values() and (c_char not in temp_map or temp_map[c_char] != p_char): possible = False; break
                            temp_map[c_char] = p_char
                        if possible: mapping = temp_map

            for s in symbols:
                if s not in mapping:
                    mapping[s] = available_targets.pop() if available_targets else random.choice(self.target_jamos)

            current_text = "".join(mapping.get(c, c) for c in ciphertext)
            current_fitness = self.fitness(current_text)
            
            T = 5.0
            for i in range(iterations):
                if len(symbols) < 2: break
                s1, s2 = random.sample(symbols, 2)
                new_mapping = mapping.copy()
                new_mapping[s1], new_mapping[s2] = new_mapping[s2], new_mapping[s1]
                
                new_text = "".join(new_mapping.get(c, c) for c in ciphertext)
                new_fitness = self.fitness(new_text)
                
                if new_fitness > current_fitness or (T > 0 and random.random() < math.exp((new_fitness - current_fitness) / T)):
                    mapping = new_mapping
                    current_fitness = new_fitness
                T *= 0.999
            
            if current_fitness > best_overall_fitness:
                best_overall_fitness = current_fitness
                best_overall_mapping = mapping.copy()
                print(f"  New best: {best_overall_fitness:.2f}")

        return "".join(best_overall_mapping.get(c, c) for c in ciphertext)

if __name__ == "__main__":
    solver = AdvancedSolver("kn_model.json", "corpus.txt")
    test_cases = ["안녕하세요", "밥 먹었니", "지금 어디야"]
    
    from final_solver import Evaluator
    evaluator = Evaluator(solver.scorer)
    
    for original in test_cases:
        orig_jamo = get_jamo_sequence(original)
        ciphertext, _ = evaluator.encrypt(original)
        print(f"\nOriginal: {original}")
        result = solver.solve(ciphertext)
        print(f"Decrypted: {result}")
