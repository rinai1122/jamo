"""V5 neural cracker.

Change vs V4: first-occurrence canonicalization instead of frequency-rank.

V4 assigns rank by descending frequency (most common symbol → rank 0).
V5 assigns rank by order of first appearance in the ciphertext:
  - first new cipher symbol seen → rank 0
  - second new symbol → rank 1
  - etc.

This preserves temporal structure — when each new symbol was introduced —
which is meaningful in Korean text: the first symbol in a sentence is almost
always an initial consonant (ㄱ–ㅎ range), and common initials like ㅅ, ㄴ, ㄱ
have a strong prior for rank-0. Frequency-rank discards this ordering.
"""
import math
import torch
import torch.nn as nn

PAD_ID = 0
SPACE_ID = 1
FIRST_RANK_ID = 2
MAX_SYMBOLS = 60

JAMO_CHARS = [chr(i) for i in range(0x3131, 0x3164)]
OUT_VOCAB = ["<PAD>", " "] + JAMO_CHARS
OUT_CHAR2IDX = {c: i for i, c in enumerate(OUT_VOCAB)}
OUT_PAD = 0


def canonicalize(ciphertext):
    """Relabel cipher symbols by order of first occurrence.

    Returns (input ids, symbols list where symbols[r] is the cipher char at rank r).
    """
    rank = {}
    next_rank = 0
    ids = []
    for c in ciphertext:
        if c == " ":
            ids.append(SPACE_ID)
        else:
            if c not in rank:
                rank[c] = next_rank
                next_rank += 1
            ids.append(FIRST_RANK_ID + rank[c])
    symbols = sorted(rank, key=lambda c: rank[c])
    return ids, symbols


def encode_plain(plaintext):
    return [OUT_CHAR2IDX.get(c, OUT_PAD) for c in plaintext]


class RankTransformerCracker(nn.Module):
    def __init__(self, d_model=256, nhead=4, num_layers=4, max_len=100,
                 norm_first=False):
        super().__init__()
        self.d_model = d_model
        in_vocab = FIRST_RANK_ID + MAX_SYMBOLS
        self.embedding = nn.Embedding(in_vocab, d_model, padding_idx=PAD_ID)
        self.pos_encoder = nn.Parameter(torch.randn(max_len, 1, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, norm_first=norm_first
        )
        final_norm = nn.LayerNorm(d_model) if norm_first else None
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers, norm=final_norm
        )
        self.out = nn.Linear(d_model, len(OUT_VOCAB))

    def forward(self, src):
        S, B = src.shape
        emb = self.embedding(src) * math.sqrt(self.d_model)
        emb = emb + self.pos_encoder[:S, :, :]
        pad_mask = (src == PAD_ID).transpose(0, 1)
        output = self.transformer_encoder(emb, src_key_padding_mask=pad_mask)
        return self.out(output)  # (S, B, out_vocab)


@torch.no_grad()
def decode_constrained(model, ciphertext, device="cpu"):
    """Recover the key via Hungarian assignment over aggregated logits."""
    from scipy.optimize import linear_sum_assignment

    ids, symbols = canonicalize(ciphertext)
    src = torch.tensor(ids).unsqueeze(1).to(device)
    logits = model(src).squeeze(1)  # (S, out_vocab)
    logprobs = torch.log_softmax(logits, dim=-1)

    jamo_idx = [OUT_CHAR2IDX[j] for j in JAMO_CHARS]
    score = torch.zeros(len(symbols), len(jamo_idx))
    for pos, c in enumerate(ciphertext):
        if c == " ":
            continue
        r = symbols.index(c)
        score[r] += logprobs[pos, jamo_idx]

    rows, cols = linear_sum_assignment(-score.numpy())
    key = {symbols[r]: JAMO_CHARS[c] for r, c in zip(rows, cols)}
    return "".join(key.get(c, c) for c in ciphertext)


@torch.no_grad()
def decode_greedy(model, ciphertext, device="cpu"):
    ids, _ = canonicalize(ciphertext)
    src = torch.tensor(ids).unsqueeze(1).to(device)
    logits = model(src).squeeze(1)
    pred = logits.argmax(dim=-1).tolist()
    return "".join(OUT_VOCAB[i] for i in pred)
