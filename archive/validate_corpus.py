import random
import sys
import io
import os
from jamo import h2j, j2hcj
from lean_solver import AdvancedSolver
from final_solver import Evaluator

# Force UTF-8
if not isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def get_jamo_sequence(text):
    return j2hcj(h2j(text))

def run_corpus_validation(corpus_path, num_tests=5, fragment_len=40):
    if not os.path.exists(corpus_path):
        print(f"Error: {corpus_path} not found.")
        return

    print(f"Loading corpus for validation from {corpus_path}...")
    with open(corpus_path, "r", encoding="utf-8") as f:
        all_text = f.read()
    
    # Pre-calculate jamo for the whole corpus to sample easily
    # (Or just sample lines and take fragments)
    lines = [line.strip() for line in all_text.split('\n') if len(line.strip()) > 10]
    
    solver = AdvancedSolver("kn_model.json", corpus_path)
    evaluator = Evaluator(solver.scorer)
    
    print(f"\n--- Starting Corpus Fragment Validation ({num_tests} tests) ---")
    
    total_acc = 0
    for i in range(num_tests):
        # Pick a random line and a fragment from it
        line = random.choice(lines)
        full_jamo = get_jamo_sequence(line)
        
        if len(full_jamo) <= fragment_len:
            fragment = full_jamo
        else:
            start = random.randint(0, len(full_jamo) - fragment_len)
            fragment = full_jamo[start:start+fragment_len]
        
        # Clean fragment to only include Jamo and space
        clean_fragment = "".join(c for c in fragment if 0x3131 <= ord(c) <= 0x3163 or c == ' ')
        if not clean_fragment.strip():
            continue

        ciphertext, _ = evaluator.encrypt(clean_fragment)
        
        print(f"\nTest {i+1} Original:  {clean_fragment}")
        print(f"Test {i+1} Ciphertext: {ciphertext}")
        
        # Use more iterations for corpus fragments
        decrypted = solver.solve(ciphertext, restarts=30, iterations=5000)
        
        matches = sum(1 for a, b in zip(clean_fragment, decrypted) if a == b)
        acc = matches / len(clean_fragment)
        total_acc += acc
        
        print(f"Test {i+1} Decrypted: {decrypted}")
        print(f"Test {i+1} Accuracy:  {acc:.2%}")

    print(f"\nAverage Accuracy over {num_tests} tests: {total_acc/num_tests:.2%}")

if __name__ == "__main__":
    # Run validation on fragments from corpus.txt
    run_corpus_validation("corpus.txt", num_tests=5, fragment_len=50)
