"""Train V6: dual canonicalization, variable-length (30-200), deeper model.

Differences vs train_v5:
- Model is DualRankTransformerCracker (6 layers, 8 heads, dual input)
- Training fragments are sampled at random lengths 30-200 (match new eval)
- 6000 steps with warmup+cosine schedule
- Checkpoint every 1000 steps (jamo_nn_v6_ckptN.pth)
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
from nn_v6 import (
    DualRankTransformerCracker, canonicalize, encode_plain,
    OUT_VOCAB, OUT_PAD, PAD_ID, JAMO_CHARS,
)

MIN_LEN = 30
MAX_LEN = 200
BATCH = 32        # smaller batch to keep variable-length padding overhead low
STEPS = 6000
PEAK_LR = 3e-4
WARMUP = 200
CKPT_EVERY = 1000


class VarLenDataGenerator:
    def __init__(self, corpus_path):
        with open(corpus_path, "r", encoding="utf-8") as f:
            self.lines = [l.strip() for l in f if len(l.strip()) > 5]

    def sample_pair(self, rng, seq_len, min_jamo=15):
        while True:
            line = rng.choice(self.lines)
            jamo_seq = j2hcj(h2j(line))
            clean = "".join(
                c for c in jamo_seq if 0x3131 <= ord(c) <= 0x3163 or c == " "
            )
            clean = " ".join(clean.split())
            if len(clean.replace(" ", "")) < min_jamo:
                continue
            if len(clean) > seq_len:
                start = rng.randint(0, len(clean) - seq_len)
                clean = clean[start: start + seq_len]
            unique = sorted(set(clean.replace(" ", "")))
            targets = list(JAMO_CHARS)
            rng.shuffle(targets)
            key = {j: targets[i] for i, j in enumerate(unique)}
            cipher = "".join(key.get(c, c) for c in clean)
            return cipher, clean

    def generate_batch(self, rng, batch_size=BATCH):
        # Sample a shared max length for this batch (more efficient padding)
        seq_len = rng.randint(MIN_LEN, MAX_LEN)
        fo_rows, fr_rows, tgt_rows = [], [], []
        for _ in range(batch_size):
            cipher, plain = self.sample_pair(rng, seq_len)
            fo_ids, fr_ids, _ = canonicalize(cipher)
            tgt = encode_plain(plain)
            # Pad to seq_len
            pad = seq_len - len(fo_ids)
            fo_ids += [PAD_ID] * pad
            fr_ids += [PAD_ID] * pad
            tgt += [OUT_PAD] * pad
            fo_rows.append(fo_ids)
            fr_rows.append(fr_ids)
            tgt_rows.append(tgt)
        return (
            torch.tensor(fo_rows).transpose(0, 1),   # (S, B)
            torch.tensor(fr_rows).transpose(0, 1),
            torch.tensor(tgt_rows).transpose(0, 1),
        )


def main():
    torch.manual_seed(0)
    rng = random.Random(0)
    gen = VarLenDataGenerator("corpus.txt")
    model = DualRankTransformerCracker(d_model=256, nhead=8, num_layers=6,
                                       max_len=210, norm_first=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    criterion = nn.CrossEntropyLoss(ignore_index=OUT_PAD)
    optimizer = optim.Adam(model.parameters(), lr=PEAK_LR)

    def lr_at(step):
        if step < WARMUP:
            return PEAK_LR * (step + 1) / WARMUP
        t = (step - WARMUP) / max(STEPS - WARMUP, 1)
        return 1e-5 + 0.5 * (PEAK_LR - 1e-5) * (1 + math.cos(math.pi * t))

    print(f"Training v6 for {STEPS} steps (batch={BATCH}, len={MIN_LEN}-{MAX_LEN})")
    t0 = time.time()
    for step in range(STEPS):
        for g in optimizer.param_groups:
            g["lr"] = lr_at(step)
        model.train()
        fo_src, fr_src, tgt = gen.generate_batch(rng)
        optimizer.zero_grad()
        output = model(fo_src, fr_src)               # (S, B, out_vocab)
        loss = criterion(output.view(-1, len(OUT_VOCAB)), tgt.reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 100 == 0:
            elapsed = time.time() - t0
            print(
                f"step {step:4d}, loss {loss.item():.4f}, "
                f"lr {lr_at(step):.2e}, {elapsed/(step+1):.2f}s/step",
                flush=True,
            )

        if (step + 1) % CKPT_EVERY == 0:
            ck = f"jamo_nn_v6_ckpt{(step+1)//CKPT_EVERY}.pth"
            torch.save(model.state_dict(), ck)
            print(f"  checkpoint saved: {ck}", flush=True)

    torch.save(model.state_dict(), "jamo_nn_v6.pth")
    print(f"Saved jamo_nn_v6.pth ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
