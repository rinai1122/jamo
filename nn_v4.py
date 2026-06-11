"""V4 neural cracker.

Fixes over V3 (MetaTransformerCracker):
1. Input canonicalization: cipher symbols are relabeled by frequency rank
   (ties broken by first occurrence) before embedding. The actual cipher
   alphabet is irrelevant under a random key, so raw symbol embeddings
   carry no transferable signal; rank ids do (e.g. "rank-0 symbol is
   usually ㅇ/ㅏ"). This replaces V3's frequency-context token.
2. Constraint-aware decoding: per-position logits are aggregated per
   cipher symbol and the key is recovered with a Hungarian assignment,
   enforcing that a substitution is consistent and injective. Greedy
   per-position argmax (V3) can output different jamos for the same
   cipher symbol, which no substitution cipher can produce.
"""
import math
import torch
import torch.nn as nn

PAD_ID = 0
SPACE_ID = 1
FIRST_RANK_ID = 2
MAX_SYMBOLS = 60  # rank ids FIRST_RANK_ID .. FIRST_RANK_ID+MAX_SYMBOLS-1

JAMO_CHARS = [chr(i) for i in range(0x3131, 0x3164)]
OUT_VOCAB = ["<PAD>", " "] + JAMO_CHARS
OUT_CHAR2IDX = {c: i for i, c in enumerate(OUT_VOCAB)}
OUT_PAD = 0


def canonicalize(ciphertext):
    """Relabel cipher symbols by (frequency desc, first occurrence asc).

    Returns (input ids, ordered list of symbols so rank r -> symbols[r]).
    """
    counts = {}
    first = {}
    for i, c in enumerate(ciphertext):
        if c == " ":
            continue
        counts[c] = counts.get(c, 0) + 1
        if c not in first:
            first[c] = i
    symbols = sorted(counts, key=lambda c: (-counts[c], first[c]))
    rank = {c: i for i, c in enumerate(symbols)}
    ids = [
        SPACE_ID if c == " " else FIRST_RANK_ID + rank[c] for c in ciphertext
    ]
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
        # Pre-LN encoders need a final norm before the classifier head
        final_norm = nn.LayerNorm(d_model) if norm_first else None
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers, norm=final_norm
        )
        self.out = nn.Linear(d_model, len(OUT_VOCAB))

    def forward(self, src):
        # src: (S, B) canonicalized ids
        S, B = src.shape
        emb = self.embedding(src) * math.sqrt(self.d_model)
        emb = emb + self.pos_encoder[:S, :, :]
        pad_mask = (src == PAD_ID).transpose(0, 1)  # (B, S)
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
    # score[r][j] = sum of logprobs of jamo j over positions of rank-r symbol
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
    """Per-position argmax (V3-style decoding) for comparison."""
    ids, _ = canonicalize(ciphertext)
    src = torch.tensor(ids).unsqueeze(1).to(device)
    logits = model(src).squeeze(1)
    pred = logits.argmax(dim=-1).tolist()
    return "".join(OUT_VOCAB[i] for i in pred)
