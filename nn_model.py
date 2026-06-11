import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4, dim_feedforward=512, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Using a simple Transformer Encoder-Decoder for Seq2Seq
        # For decryption, the mapping is symbol-to-symbol, but context matters.
        # We can use a Transformer Encoder followed by a linear layer if the lengths are fixed,
        # or a full Encoder-Decoder if we want more flexibility.
        # Let's start with a Transformer Encoder + Linear Layer for symbol-to-symbol mapping.
        
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Linear(d_model, vocab_size)

    def forward(self, src):
        # src shape: (seq_len, batch_size)
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        return output

class JamoTokenizer:
    def __init__(self, vocab_list):
        self.vocab = sorted(list(set(vocab_list))) + ['<PAD>', '<UNK>', '<SOS>', '<EOS>']
        self.char2idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx2char = {idx: char for idx, char in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)

    def encode(self, text):
        return [self.char2idx.get(char, self.char2idx['<UNK>']) for char in text]

    def decode(self, indices):
        return "".join([self.idx2char.get(idx, '<UNK>') for idx in indices])
