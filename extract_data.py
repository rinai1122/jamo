import os
import sys

# Korpora usually downloads to ~/Korpora. Let's find the nsmc files.
possible_paths = [
    os.path.join(os.path.expanduser("~"), "Korpora", "nsmc", "ratings_train.txt"),
    "ratings_train.txt"
]

def extract_text():
    found_path = None
    for p in possible_paths:
        if os.path.exists(p):
            found_path = p
            break
    
    if not found_path:
        print("Dataset not found. Please ensure nsmc is downloaded.")
        return

    print(f"Extracting from {found_path}...")
    with open(found_path, "r", encoding="utf-8") as f, open("corpus.txt", "w", encoding="utf-8") as out:
        next(f) # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                text = parts[1]
                if text:
                    out.write(text + "\n")
    print("Extraction complete: corpus.txt")

if __name__ == "__main__":
    extract_text()
