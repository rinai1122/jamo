import sys
import io
import random
from final_solver import KneserNeyScorer, Evaluator, get_jamo_sequence

# Do not wrap sys.stdout if already wrapped in final_solver or just use it once
if not isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def analyze_convergence(original_text, iterations=20000):
    print(f"Analyzing convergence for: {original_text}")
    scorer = KneserNeyScorer("kn_model.json")
    evaluator = Evaluator(scorer)
    
    # 1. Get true plaintext Jamo and its fitness
    true_jamo = get_jamo_sequence(original_text)
    true_fitness = evaluator.fitness(true_jamo)
    
    # 2. Encrypt and solve
    ciphertext, true_key = evaluator.encrypt(original_text)
    decrypted_jamo = evaluator.solve(ciphertext, iterations=iterations)
    decrypted_fitness = evaluator.fitness(decrypted_jamo)
    
    # 3. Compare
    print("\n--- Comparative Analysis ---")
    print(f"True Plaintext Fitness: {true_fitness:.4f}")
    print(f"Solver Result Fitness:  {decrypted_fitness:.4f}")
    print(f"Difference:             {decrypted_fitness - true_fitness:.4f}")
    
    if decrypted_fitness < true_fitness:
        print("\nCONCLUSION: Search Limitation.")
        print("The scoring system knows the truth is better, but the solver failed to find it.")
        print("Try increasing iterations, restarts, or adjusting the temperature schedule.")
    elif decrypted_fitness > true_fitness:
        print("\nCONCLUSION: Scoring System Limitation.")
        print("The solver found a 'better' score than the truth. The Language Model is misleading the search.")
        print("This means the model needs more data or a higher N-gram order to distinguish real Korean from 'Korean-sounding' gibberish.")
    else:
        print("\nCONCLUSION: Exact Convergence.")
        print("The solver reached the same fitness as the truth.")

    # Show text comparison
    print(f"\nTrue Jamo: {true_jamo}")
    print(f"Decrypted: {decrypted_jamo}")

if __name__ == "__main__":
    test_sentence = "세종대왕이 한글을 창제하셨다"
    analyze_convergence(test_sentence)
