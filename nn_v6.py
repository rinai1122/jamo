"""NN v6: dual canonicalization + deeper model for variable-length texts.

Changes vs V5:
1. Dual input: each position gets BOTH first-occurrence rank AND
   frequency rank as separate embeddings (d_model//2 each, concat).
   V4 used freq-rank only; V5 used first-occurrence only.
   V6 gives the model both signals; attention learns to weight them.
2. Deeper: 6 layers, 8 heads (vs 4/4 in v4/v5).
3. max_len=210 to handle the new 30-200 char benchmark.
4. Pre-LN (norm_first=True) by default — v4b/v5 recipe that avoids
   gradient vanishing in deeper stacks.
"""
import math
import torch
import torch.nn as nn

PAD_ID = 0
SPACE_ID = 1
RANK_OFFSET = 2       # both rank channels share this offset
MAX_SYMBOLS = 64      # safe upper bound for Korean jamo

JAMO_CHARS = [chr(i) for i in range(0x3131, 0x3164)]
OUT_VOCAB = ["<PAD>", " "] + JAMO_CHARS
OUT_CHAR2IDX = {c: i for i, c in enumerate(OUT_VOCAB)}
OUT_PAD = 0


def canonicalize(ciphertext):
    """Return (fo_ids, fr_ids, symbols) where:
    - fo_ids[i] = RANK_OFFSET + first-occurrence rank of ciphertext[i]
    - fr_ids[i] = RANK_OFFSET + frequency rank of ciphertext[i]
    - symbols[r] = cipher char with first-occurrence rank r
    """
    # First-occurrence ranks
    fo_rank = {}
    fo_next = 0
    # Frequency counts
    counts = {}
    first_pos = {}
    for i, c in enumerate(ciphertext):
        if c == " ":
            continue
        counts[c] = counts.get(c, 0) + 1
        if c not in first_pos:
            first_pos[c] = i
        if c not in fo_rank:
            fo_rank[c] = fo_next
            fo_next += 1

    # Frequency ranks (0 = most frequent, ties broken by first occurrence)
    sorted_by_freq = sorted(counts, key=lambda c: (-counts[c], first_pos[c]))
    fr_rank = {c: i for i, c in enumerate(sorted_by_freq)}

    fo_ids, fr_ids = [], []
    for c in ciphertext:
        if c == " ":
            fo_ids.append(SPACE_ID)
            fr_ids.append(SPACE_ID)
        else:
            fo_ids.append(RANK_OFFSET + fo_rank[c])
            fr_ids.append(RANK_OFFSET + fr_rank[c])

    # symbols ordered by first-occurrence rank (for constrained decode)
    symbols = sorted(fo_rank, key=lambda c: fo_rank[c])
    return fo_ids, fr_ids, symbols


def encode_plain(plaintext):
    return [OUT_CHAR2IDX.get(c, OUT_PAD) for c in plaintext]


class DualRankTransformerCracker(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=6, max_len=210,
                 norm_first=True):
        super().__init__()
        self.d_model = d_model
        half = d_model // 2
        in_vocab = RANK_OFFSET + MAX_SYMBOLS  # shared vocab for both channels
        self.fo_emb = nn.Embedding(in_vocab, half, padding_idx=PAD_ID)
        self.fr_emb = nn.Embedding(in_vocab, half, padding_idx=PAD_ID)
        self.pos_encoder = nn.Parameter(torch.randn(max_len, 1, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=d_model * 4,
            norm_first=norm_first, batch_first=False,
        )
        final_norm = nn.LayerNorm(d_model)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers, norm=final_norm
        )
        self.out = nn.Linear(d_model, len(OUT_VOCAB))

    def forward(self, fo_src, fr_src):
        # fo_src, fr_src: (S, B)
        S, B = fo_src.shape
        fo = self.fo_emb(fo_src)  # (S, B, d//2)
        fr = self.fr_emb(fr_src)  # (S, B, d//2)
        emb = torch.cat([fo, fr], dim=-1) * math.sqrt(self.d_model)
        emb = emb + self.pos_encoder[:S, :, :]
        pad_mask = (fo_src == PAD_ID).transpose(0, 1)  # (B, S)
        output = self.transformer_encoder(emb, src_key_padding_mask=pad_mask)
        return self.out(output)  # (S, B, out_vocab)


@torch.no_grad()
def decode_constrained(model, ciphertext, device="cpu"):
    """Hungarian assignment over per-symbol aggregated logprobs."""
    from scipy.optimize import linear_sum_assignment

    fo_ids, fr_ids, symbols = canonicalize(ciphertext)
    fo_t = torch.tensor(fo_ids).unsqueeze(1).to(device)
    fr_t = torch.tensor(fr_ids).unsqueeze(1).to(device)
    logits = model(fo_t, fr_t).squeeze(1)  # (S, out_vocab)
    logprobs = torch.log_softmax(logits, dim=-1)

    jamo_idx = [OUT_CHAR2IDX[j] for j in JAMO_CHARS]
    score = torch.zeros(len(symbols), len(jamo_idx))
    sym_fo = {c: fo_ids[i] - RANK_OFFSET
              for i, c in enumerate(ciphertext) if c != " "}
    for pos, c in enumerate(ciphertext):
        if c == " ":
            continue
        r = sym_fo[c]
        if r < len(symbols):
            score[r] += logprobs[pos, jamo_idx].cpu()

    rows, cols = linear_sum_assignment(-score.numpy())
    key = {symbols[r]: JAMO_CHARS[c] for r, c in zip(rows, cols)}
    return "".join(key.get(c, c) for c in ciphertext)


@torch.no_grad()
def decode_greedy(model, ciphertext, device="cpu"):
    fo_ids, fr_ids, _ = canonicalize(ciphertext)
    fo_t = torch.tensor(fo_ids).unsqueeze(1).to(device)
    fr_t = torch.tensor(fr_ids).unsqueeze(1).to(device)
    logits = model(fo_t, fr_t).squeeze(1)
    pred = logits.argmax(dim=-1).tolist()
    return "".join(OUT_VOCAB[i] for i in pred)


@torch.no_grad()
def predict_key(model, ciphertext, device="cpu"):
    """Return the predicted key dict {cipher_char -> plain_jamo} via Hungarian."""
    from scipy.optimize import linear_sum_assignment

    fo_ids, fr_ids, symbols = canonicalize(ciphertext)
    fo_t = torch.tensor(fo_ids).unsqueeze(1).to(device)
    fr_t = torch.tensor(fr_ids).unsqueeze(1).to(device)
    logits = model(fo_t, fr_t).squeeze(1)
    logprobs = torch.log_softmax(logits, dim=-1)

    jamo_idx = [OUT_CHAR2IDX[j] for j in JAMO_CHARS]
    score = torch.zeros(len(symbols), len(jamo_idx))
    sym_fo = {c: fo_ids[i] - RANK_OFFSET
              for i, c in enumerate(ciphertext) if c != " "}
    for pos, c in enumerate(ciphertext):
        if c == " ":
            continue
        r = sym_fo[c]
        if r < len(symbols):
            score[r] += logprobs[pos, jamo_idx].cpu()

    rows, cols = linear_sum_assignment(-score.numpy())
    return {symbols[r]: JAMO_CHARS[c] for r, c in zip(rows, cols)}
