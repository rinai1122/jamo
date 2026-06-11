"""V4b: same 1200-step budget as the fair runs, better training recipe.

Changes vs train_compare.py v4:
- Pre-LN transformer (norm_first=True) + lr 3e-4 with linear warmup and
  cosine decay (post-LN at flat 1e-4 converges slowly).
- Data matches the benchmark distribution: fragments cleaned to
  jamo+space, runs of spaces collapsed, real PAD tokens instead of
  space-padding short lines.
"""
import sys
import io
import math
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim

if not isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from jamo import h2j, j2hcj
from nn_v4 import (
    RankTransformerCracker, canonicalize, encode_plain,
    OUT_VOCAB, OUT_PAD, PAD_ID, JAMO_CHARS,
)

SEQ_LEN = 50
BATCH = 64
STEPS = 1200
PEAK_LR = 3e-4
WARMUP = 100


class CleanDataGenerator:
    def __init__(self, corpus_path):
        with open(corpus_path, "r", encoding="utf-8") as f:
            self.lines = [l.strip() for l in f if len(l.strip()) > 5]

    def sample_pair(self, rng, seq_len=SEQ_LEN, min_len=25):
        while True:
            line = rng.choice(self.lines)
            jamo_seq = j2hcj(h2j(line))
            clean = "".join(
                c for c in jamo_seq if 0x3131 <= ord(c) <= 0x3163 or c == " "
            )
            clean = " ".join(clean.split())
            if len(clean.replace(" ", "")) < min_len:
                continue
            if len(clean) > seq_len:
                start = rng.randint(0, len(clean) - seq_len)
                clean = clean[start:start + seq_len]
            unique = sorted(set(clean.replace(" ", "")))
            targets = list(JAMO_CHARS)
            rng.shuffle(targets)
            key = {j: targets[i] for i, j in enumerate(unique)}
            cipher = "".join(key.get(c, c) for c in clean)
            return cipher, clean

    def generate_batch(self, rng, batch_size=BATCH, seq_len=SEQ_LEN):
        src_rows, tgt_rows = [], []
        for _ in range(batch_size):
            cipher, plain = self.sample_pair(rng, seq_len)
            src = canonicalize(cipher)[0]
            tgt = encode_plain(plain)
            src += [PAD_ID] * (seq_len - len(src))
            tgt += [OUT_PAD] * (seq_len - len(tgt))
            src_rows.append(src)
            tgt_rows.append(tgt)
        return (
            torch.tensor(src_rows).transpose(0, 1),
            torch.tensor(tgt_rows).transpose(0, 1),
        )


def main():
    torch.manual_seed(0)
    rng = random.Random(0)
    gen = CleanDataGenerator("corpus.txt")
    model = RankTransformerCracker(256, 4, 4, norm_first=True)
    criterion = nn.CrossEntropyLoss(ignore_index=OUT_PAD)
    optimizer = optim.Adam(model.parameters(), lr=PEAK_LR)

    def lr_at(step):
        if step < WARMUP:
            return PEAK_LR * (step + 1) / WARMUP
        t = (step - WARMUP) / (STEPS - WARMUP)
        return 1e-5 + 0.5 * (PEAK_LR - 1e-5) * (1 + math.cos(math.pi * t))

    print(f"Training v4b for {STEPS} steps (batch={BATCH}, seq={SEQ_LEN})")
    t0 = time.time()
    for step in range(STEPS):
        for g in optimizer.param_groups:
            g["lr"] = lr_at(step)
        model.train()
        src, tgt = gen.generate_batch(rng)
        optimizer.zero_grad()
        output = model(src)
        loss = criterion(output.view(-1, len(OUT_VOCAB)), tgt.reshape(-1))
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print(
                f"step {step}, loss {loss.item():.4f}, "
                f"{(time.time()-t0)/(step+1):.2f}s/step", flush=True,
            )

    torch.save(model.state_dict(), "jamo_nn_v4b.pth")
    print(f"Saved jamo_nn_v4b.pth ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
