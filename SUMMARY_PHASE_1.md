# Korean Jamo Substitution Cipher Cracker - Phase 1 Summary

## Overview
Phase 1 focused on a statistical and evolutionary approach to crack Korean Jamo-based substitution ciphers, specifically targeting short texts (40-60 characters).

## Key Components Implemented
1.  **7-Gram Kneser-Ney Language Model**: A high-order probabilistic model trained on a ~13MB Korean corpus (NSMC). It provides a robust "language instinct" by scoring Jamo sequences based on long-range dependencies.
2.  **Dictionary Pattern Anchoring**: A pre-computed map of patterns for the top 50,000 Korean words. It allows the solver to "seed" the search with high-probability word matches, drastically reducing the search space.
3.  **Simulated Annealing with Massive Restarts**: A robust search algorithm designed to overcome local optima. It performs 50-100 random restarts with 15,000+ iterations each to find the global maximum of the fitness function.
4.  **Jamo Decomposition & Syllable Reconstruction**: Tools to handle the transformation between Hangeul syllables and Jamo compatibility characters.

## Performance
*   **Target Length**: 40-60 characters.
*   **Accuracy**: Achieved an average of ~30.8% accuracy on random corpus fragments, with peak performance over 77% on sentences with recognizable word patterns.
*   **Strength**: Extremely effective at identifying common words and phonological structures.
*   **Limitation**: Sparse statistical signals in very short texts can still lead to ambiguity that purely statistical models cannot always resolve.

## Files
*   `lean_solver.py`: Core high-intensity solver.
*   `train_kn_final.py`: KN model training script.
*   `kn_model.json`: Pre-trained 7-gram model.
*   `corpus.txt`: Cleaned Korean text corpus.
*   `validate_corpus.py`: Fragment-based validation suite.

---
*Proceeding to Phase 2: Neural Network Approach.*
