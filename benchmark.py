"""Fixed benchmark set for comparing solver versions fairly.

Generates N corpus fragments (jamo + space), encrypts each with a seeded
random substitution key, and saves everything to benchmark_set.json so
every solver version is evaluated on identical data.
"""
import json
import random
import sys
import io

if not isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from crack import decompose

JAMO_POOL = [chr(i) for i in range(0x3131, 0x3164)]


def get_jamo_sequence(text):
    return decompose(text)


def build(corpus_path="corpus.txt", out_path="benchmark_set.json",
          num_tests=100, min_len=30, max_len=200, seed=1234):
    """Build a benchmark with variable-length fragments (min_len..max_len jamo chars)."""
    rng = random.Random(seed)
    with open(corpus_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if len(l.strip()) > 10]

    cases = []
    while len(cases) < num_tests:
        line = rng.choice(lines)
        full_jamo = get_jamo_sequence(line)
        fragment_len = rng.randint(min_len, max_len)
        if len(full_jamo) <= fragment_len:
            fragment = full_jamo
        else:
            start = rng.randint(0, len(full_jamo) - fragment_len)
            fragment = full_jamo[start:start + fragment_len]
        clean = "".join(c for c in fragment if 0x3131 <= ord(c) <= 0x3163 or c == " ")
        clean = " ".join(clean.split())
        # Require at least half of min_len to be non-space jamo
        if len(clean.replace(" ", "")) < min_len // 2:
            continue

        unique = sorted(set(clean.replace(" ", "")))
        targets = list(JAMO_POOL)
        rng.shuffle(targets)
        key = {j: targets[i] for i, j in enumerate(unique)}
        ciphertext = "".join(key.get(c, c) for c in clean)
        cases.append({"plaintext": clean, "ciphertext": ciphertext,
                      "length": len(clean)})

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=1)
    print(f"Saved {len(cases)} cases to {out_path} (len range {min_len}-{max_len})")


def accuracy(plaintext, decrypted):
    n = min(len(plaintext), len(decrypted))
    matches = sum(1 for a, b in zip(plaintext[:n], decrypted[:n]) if a == b)
    return matches / len(plaintext)


if __name__ == "__main__":
    build()
