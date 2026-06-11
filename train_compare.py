"""Train V3 (baseline) or V4 (improved) with an identical compute budget.

Usage: python train_compare.py v3|v4 [steps] [resume]
Same data generator, seq_len, batch size, optimizer, lr, d_model, layers.
Pass "resume" to continue training from the existing checkpoint.
"""
import sys
import io
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim

if not isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from nn_data_gen import DataGenerator

SEQ_LEN = 50
BATCH = 64
D_MODEL = 256
NHEAD = 4
LAYERS = 4
LR = 1e-4


def main():
    which = sys.argv[1]
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    torch.manual_seed(0)
    random.seed(0)

    gen = DataGenerator("corpus.txt")
    device = "cpu"

    if which == "v3":
        from nn_model_v3 import MetaTransformerCracker
        from nn_model import JamoTokenizer

        jamo_chars = [chr(i) for i in range(0x3131, 0x3164)] + [" "]
        tokenizer = JamoTokenizer(jamo_chars)
        model = MetaTransformerCracker(tokenizer.vocab_size, D_MODEL, NHEAD, LAYERS)
        pad = tokenizer.char2idx["<PAD>"]

        def make_batch(batch):
            src = torch.tensor([tokenizer.encode(c) for c, p in batch]).transpose(0, 1)
            tgt = torch.tensor([tokenizer.encode(p) for c, p in batch]).transpose(0, 1)
            return src, tgt

        out_vocab = tokenizer.vocab_size
        ckpt = "jamo_nn_v3_fair.pth"
    else:
        from nn_v4 import RankTransformerCracker, canonicalize, encode_plain, OUT_VOCAB, OUT_PAD

        model = RankTransformerCracker(D_MODEL, NHEAD, LAYERS)
        pad = OUT_PAD

        def make_batch(batch):
            src = torch.tensor([canonicalize(c)[0] for c, p in batch]).transpose(0, 1)
            tgt = torch.tensor([encode_plain(p) for c, p in batch]).transpose(0, 1)
            return src, tgt

        out_vocab = len(OUT_VOCAB)
        ckpt = "jamo_nn_v4_fair.pth"

    model = model.to(device)
    if len(sys.argv) > 3 and sys.argv[3] == "resume":
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"Resumed from {ckpt}")
    criterion = nn.CrossEntropyLoss(ignore_index=pad)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"Training {which} for {steps} steps (batch={BATCH}, seq={SEQ_LEN})")
    t0 = time.time()
    for step in range(steps):
        model.train()
        batch = gen.generate_batch(BATCH, SEQ_LEN)
        src, tgt = make_batch(batch)
        optimizer.zero_grad()
        output = model(src)
        loss = criterion(output.view(-1, out_vocab), tgt.reshape(-1))
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print(
                f"step {step}, loss {loss.item():.4f}, "
                f"{(time.time()-t0)/(step+1):.2f}s/step", flush=True,
            )

    torch.save(model.state_dict(), ckpt)
    print(f"Saved {ckpt} ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
