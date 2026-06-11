import json
import random
import sys
import io
import math
import collections
from jamo import h2j, j2hcj, j2h

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

class JamoSolver:
    def __init__(self, freq_table_path):
        with open(freq_table_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.ref_unigrams = data["unigrams"]
            self.ref_bigrams = data["bigrams"]
            self.ref_trigrams = data["trigrams"]
            self.dictionary = set(data.get("dictionary", []))
            # Decompose dictionary into Jamo
            self.jamo_dictionary = {j2hcj(h2j(w)) for w in self.dictionary}
        
        self.sorted_targets = [k for k, v in sorted(self.ref_unigrams.items(), key=lambda x: x[1], reverse=True) if k != ' ' and k != '\n']

    def is_vowel(self, c):
        return 0x314F <= ord(c) <= 0x3163
    
    def is_consonant(self, c):
        return 0x3131 <= ord(c) <= 0x314E

    def fitness(self, text):
        score = 0
        # Bigrams and Trigrams (higher weight)
        for i in range(len(text) - 1):
            bg = text[i:i+2]
            score += 2 * math.log10(self.ref_bigrams.get(bg, 1e-8))
        
        for i in range(len(text) - 2):
            tg = text[i:i+3]
            score += 3 * math.log10(self.ref_trigrams.get(tg, 1e-9))
            
        # Strict Phonotactic constraints
        # 1. Every syllable starts with a consonant (mostly)
        # 2. Vowels must follow consonants
        # 3. No more than 2 consonants in a row (tail + next head)
        for i in range(len(text)):
            c = text[i]
            if c == ' ': continue
            
            if i == 0 or text[i-1] == ' ':
                if self.is_vowel(c): score -= 100 # Syllable cannot start with vowel
            
            if i > 0 and text[i-1] != ' ':
                if self.is_consonant(text[i-1]) and self.is_consonant(c):
                    # C-C is okay (tail-head), but C-C-C is bad
                    if i > 1 and self.is_consonant(text[i-2]):
                        score -= 200
                if self.is_vowel(text[i-1]) and self.is_vowel(c):
                    score -= 100 # V-V is rare/complex
        
        # Word dictionary fitness
        words_in_text = text.split()
        for w in words_in_text:
            if w in self.jamo_dictionary:
                score += 500 # Massively favor real words
        
        return score

    def decrypt(self, ciphertext, mapping):
        return "".join(mapping.get(c, c) for c in ciphertext)

    def get_initial_mapping(self, cipher_symbols, ciphertext):
        counts = collections.Counter(ciphertext.replace(" ", "").replace("\n", ""))
        sorted_cipher = [k for k, v in sorted(counts.items(), key=lambda x: x[1], reverse=True)]
        
        mapping = {}
        targets = list(self.sorted_targets)
        for i, s in enumerate(sorted_cipher):
            if i < len(targets):
                mapping[s] = targets[i]
            else:
                mapping[s] = random.choice(targets)
        for s in cipher_symbols:
            if s not in mapping: mapping[s] = random.choice(targets)
        return mapping

    def solve(self, ciphertext, restarts=20, iterations=20000):
        cipher_symbols = sorted(list(set(ciphertext)))
        for char in [' ', '\n', '.', ',', '!', '?']:
            if char in cipher_symbols: cipher_symbols.remove(char)
        
        best_overall_fitness = -float('inf')
        best_overall_mapping = None

        for r in range(restarts):
            current_mapping = self.get_initial_mapping(cipher_symbols, ciphertext)
            
            # For restarts after the first, add more randomness
            if r > 0:
                keys = list(current_mapping.keys())
                for _ in range(random.randint(1, len(keys))):
                    k1, k2 = random.sample(keys, 2)
                    current_mapping[k1], current_mapping[k2] = current_mapping[k2], current_mapping[k1]

            current_text = self.decrypt(ciphertext, current_mapping)
            current_fitness = self.fitness(current_text)
            
            T = 10.0
            cooling_rate = 0.9998
            
            for i in range(iterations):
                s1, s2 = random.sample(cipher_symbols, 2)
                new_mapping = current_mapping.copy()
                new_mapping[s1], new_mapping[s2] = new_mapping[s2], new_mapping[s1]
                
                new_text = self.decrypt(ciphertext, new_mapping)
                new_fitness = self.fitness(new_text)
                
                delta = new_fitness - current_fitness
                if delta > 0 or (T > 0 and random.random() < math.exp(delta / T)):
                    current_fitness = new_fitness
                    current_mapping = new_mapping
                
                T *= cooling_rate
            
            if r % 2 == 0:
                print(f"Restart {r}, Best Fitness: {current_fitness:.2f}")
            if current_fitness > best_overall_fitness:
                best_overall_fitness = current_fitness
                best_overall_mapping = current_mapping

        return self.decrypt(ciphertext, best_overall_mapping), best_overall_mapping

if __name__ == "__main__":
    solver = JamoSolver("freq_table.json")
    
    # "세종대왕이 한글을 만드셨다" -> decomposed
    original_text = "ㅅㅔㅈㅗㅇㄷㅐㅇㅘㅇㅇㅣ ㅎㅏㄴㄱㅡㄹㅇㅡㄹ ㅁㅏㄴㄷㅡㅅㅕㅅㄷㅏ"
    
    jamo_list = list(set(original_text.replace(" ", "")))
    symbols = "abcdefghijklmnopqrstuvwxyz1234567890"
    sub_map = {j: symbols[i] for i, j in enumerate(jamo_list)}
    
    ciphertext = "".join(sub_map.get(c, c) for c in original_text)
    print(f"Ciphertext: {ciphertext}")
    
    decrypted_jamo, mapping = solver.solve(ciphertext, restarts=30, iterations=30000)
    
    print("\n--- RESULTS ---")
    print(f"Original Jamo Hex:  {' '.join(hex(ord(c)) for c in original_text[:10])}")
    print(f"Decrypted Jamo Hex: {' '.join(hex(ord(c)) for c in decrypted_jamo[:10])}")
    
    matches = original_text.replace(" ", "") == decrypted_jamo.replace(" ", "")
    print(f"Direct Match (ignoring spaces): {matches}")
    
    if matches:
        print("SUCCESS! The cipher was cracked.")
    else:
        print("FAILED to crack the cipher fully.")
