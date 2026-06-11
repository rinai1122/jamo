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
        print("Building Dictionary Pattern Map...")
        if os.path.exists(corpus_path):
            with open(corpus_path, "r", encoding="utf-8") as f:
                words = []
                for line in f:
                    words.extend(line.strip().split())
                word_counts = collections.Counter(words)
                # Keep top 50k words for pattern matching
                common_words = [w for w, _ in word_counts.most_common(50000)]
                for word in common_words:
                    jamo_word = get_jamo_sequence(word)
                    if len(jamo_word) > 2:
                        pattern = get_word_pattern(jamo_word)
                        self.pattern_map[pattern].append(jamo_word)
        print(f"Pattern map built with {len(self.pattern_map)} unique patterns.")

    def get_matches(self, cipher_word):
        pattern = get_word_pattern(cipher_word)
        return self.pattern_map.get(pattern, [])

class AdvancedSolver:
    def __init__(self, kn_model_path, corpus_path):
        from final_solver import KneserNeyScorer
        self.scorer = KneserNeyScorer(kn_model_path)
        self.dict_anchor = DictionaryAnchor(corpus_path)
        self.target_jamos = [k for k in self.scorer.vocab if 0x3131 <= ord(k) <= 0x3163]

    def fitness(self, text):
        score = self.scorer.score(text)
        words = text.split()
        for w in words:
            # Reward actual common words
            if len(w) > 2 and w in self.dict_anchor.pattern_map.get(get_word_pattern(w), []):
                score += 2000 
        return score

    def solve(self, ciphertext, restarts=30, iterations=15000):
        symbols = sorted(list(set(ciphertext.replace(" ", ""))))
        best_overall_mapping = None
        best_overall_fitness = -float('inf')

        cipher_words = ciphertext.split()
        cipher_words.sort(key=len, reverse=True)
        
        print(f"Starting solver with {restarts} restarts...")
        for r in range(restarts):
            # Print every few restarts to keep output flowing
            if r % 5 == 0:
                print(f"  Working on restart {r}/{restarts}...")
            
            mapping = {}
            available_targets = list(self.target_jamos)
            random.shuffle(available_targets)
            
            # Seeding: try to anchor patterns for the longest words
            if r > 0:
                # 50% chance to use seeding, 50% random start
                if random.random() < 0.5:
                    for cw in cipher_words[:2]:
                        matches = self.dict_anchor.get_matches(cw)
                        if matches:
                            match = random.choice(matches)
                            temp_map = mapping.copy()
                            possible = True
                            for c_char, p_char in zip(cw, match):
                                if c_char in temp_map and temp_map[c_char] != p_char:
                                    possible = False; break
                                if p_char in temp_map.values() and (c_char not in temp_map or temp_map[c_char] != p_char):
                                    possible = False; break
                                temp_map[c_char] = p_char
                            if possible: mapping = temp_map

            # Fill remaining symbols
            remaining = [s for s in symbols if s not in mapping]
            for s in remaining:
                if available_targets:
                    mapping[s] = available_targets.pop()
                else:
                    mapping[s] = random.choice(self.target_jamos)

            current_text = "".join(mapping.get(c, c) for c in ciphertext)
            current_fitness = self.fitness(current_text)
            
            # Simulated Annealing
            T = 10.0
            cooling_rate = 0.9997
            
            for i in range(iterations):
                if len(symbols) < 2: break
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
            
            if current_fitness > best_overall_fitness:
                best_overall_fitness = current_fitness
                best_overall_mapping = mapping.copy()
                print(f"Restart {r}: New Best Fitness = {best_overall_fitness:.4f}")

        return "".join(best_overall_mapping.get(c, c) for c in ciphertext)

if __name__ == "__main__":
    solver = AdvancedSolver("kn_model.json", "corpus.txt")
    
    # Updated test set with very short sentences
    test_cases = [
        "안녕하세요",
        "밥 먹었니?",
        "지금 어디야",
        "날씨가 맑다"
    ]
    
    from final_solver import Evaluator
    evaluator = Evaluator(solver.scorer)
    
    print("\n--- TEST RUN ---")
    for original in test_cases:
        orig_jamo = get_jamo_sequence(original)
        # Handle punctuation in jamo sequence for ground truth comparison
        clean_orig_jamo = "".join(c for c in orig_jamo if 0x3131 <= ord(c) <= 0x3163 or c == ' ')
        
        ciphertext, _ = evaluator.encrypt(original)
        print(f"\nOriginal:   {original}")
        print(f"Ciphertext: {ciphertext}")
        
        result = solver.solve(ciphertext, restarts=50, iterations=20000)
        
        matches = sum(1 for a, b in zip(clean_orig_jamo, result) if a == b)
        acc = matches / len(clean_orig_jamo)
        
        print(f"Decrypted:  {result}")
        print(f"Accuracy:   {acc:.2%}")
