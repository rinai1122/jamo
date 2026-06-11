import sys
import io
from jamo import h2j, j2hcj

# Force UTF-8 for Windows console
if sys.platform == "win32":
    import os
    os.system("chcp 65001 > nul")

def get_jamo_sequence(text):
    return j2hcj(h2j(text))

if __name__ == "__main__":
    sample = "안녕하세요"
    # Using hex representation to be sure of what's happening
    jamos = get_jamo_sequence(sample)
    print(f"Jamo (repr): {repr(jamos)}")
    print(f"Jamo (hex): {' '.join(hex(ord(c)) for c in jamos)}")
